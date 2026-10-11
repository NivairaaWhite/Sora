#!/usr/bin/env python3
"""Sora chatbot: model-native chat templates, stable Gradio history, and optional Edge TTS."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DEFAULT_MODEL, SYSTEM_PROMPT, legacy_serialize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = ROOT / "models" / "sora-qwen-lora" / "final"
DEFAULT_VOICE = "en-US-AriaNeural"

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    PeftModel = None
    HAS_PEFT = False

try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    edge_tts = None
    HAS_TTS = False

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    gr = None
    HAS_GRADIO = False


def is_notebook() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # noqa: F821
    except NameError:
        return False


def enforce_dependencies() -> None:
    import importlib.util
    required = ("torch", "transformers")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing required packages: {', '.join(missing)}. Install requirements.txt first.")


def _safe_from_pretrained(model_name: str, kwargs: dict[str, Any]):
    dtype = kwargs.pop("dtype")
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_model(adapter_path: str | Path = DEFAULT_ADAPTER, base_model: str = DEFAULT_MODEL):
    """Load the matching base model and an adapter if present; reject mismatched adapters."""
    adapter_dir = Path(adapter_path).expanduser() if adapter_path else None
    adapter_config_path = adapter_dir / "adapter_config.json" if adapter_dir else None
    adapter_exists = bool(adapter_config_path and adapter_config_path.is_file())

    if adapter_exists:
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        recorded_base = config.get("base_model_name_or_path")
        if recorded_base and recorded_base != base_model:
            raise ValueError(
                "The selected LoRA adapter was trained on a different base model.\n"
                f"  Adapter base: {recorded_base}\n  Requested base: {base_model}\n"
                "Pass --base_model with the adapter's original base, or select the correct adapter folder."
            )
        if not HAS_PEFT:
            raise SystemExit("A LoRA adapter was found but peft is missing. Install requirements.txt again.")
    else:
        print(f"No trained adapter at {adapter_dir}; loading the base model only.")

    print(f"Loading tokenizer: {adapter_dir if adapter_exists else base_model}")
    tokenizer_source = str(adapter_dir) if adapter_exists else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except Exception:
        if tokenizer_source == base_model:
            raise
        print("Adapter tokenizer could not be loaded; falling back to the base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer needs a pad_token_id or eos_token_id.")

    use_cuda = torch.cuda.is_available()
    bf16_supported = bool(use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else torch.float32)
    model_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": False}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
    print(f"Loading base model: {base_model}")
    model = _safe_from_pretrained(base_model, model_kwargs)

    if adapter_exists:
        print(f"Loading LoRA adapter: {adapter_dir}")
        # Standard LoRA does not require torchao. Some Colab images include an
        # old torchao (for example 0.10.0), while newer PEFT raises instead of
        # ignoring it. Disable only PEFT's optional torchao dispatcher before
        # adapter injection, matching the training path in scripts/train.py.
        try:
            import peft.import_utils as peft_import_utils
            peft_import_utils.is_torchao_available = lambda: False
            try:
                import peft.tuners.lora.torchao as peft_torchao
                peft_torchao.is_torchao_available = lambda: False
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _clean_history(history: Any) -> list[dict[str, str]]:
    """Accept both Gradio messages and old (user, assistant) tuple histories."""
    result: list[dict[str, str]] = []
    if not isinstance(history, (list, tuple)):
        return result
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                result.append({"role": role, "content": content.strip()})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if isinstance(user_text, str) and user_text.strip():
                result.append({"role": "user", "content": user_text.strip()})
            if isinstance(assistant_text, str) and assistant_text.strip():
                result.append({"role": "assistant", "content": assistant_text.strip()})
    # Bound context length; keep complete recent turns where possible.
    return result[-12:]


def build_inputs(tokenizer, history: Any, user_input: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    messages.append({"role": "user", "content": user_input.strip()})
    if getattr(tokenizer, "chat_template", None):
        # Use the model's native template to match the format used during training.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The template already contains special tokens; do not add a second BOS/EOS layer.
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    legacy_messages = messages[1:]
    prompt = legacy_serialize(legacy_messages, include_system=True) + "<|assistant|>\n"
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True)


@torch.inference_mode()
def generate_reply(model, tokenizer, history: Any, user_input: str, max_new_tokens: int = 180) -> str:
    user_input = str(user_input or "").strip()
    if not user_input:
        return "I'm right here. What would you like to talk about?"
    inputs = build_inputs(tokenizer, history, user_input)
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    for special in ("<|im_end|>", "<|endoftext|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(special)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
                # Do not include unknown-token aliases as stop markers.
                if token_id != getattr(tokenizer, "unk_token_id", None):
                    eos_ids.append(token_id)
        except Exception:
            pass

    gen_kwargs = {
        "max_new_tokens": max(16, int(max_new_tokens)),
        "do_sample": True,
        "temperature": 0.70,
        "top_p": 0.90,
        "top_k": 40,
        "repetition_penalty": 1.10,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    outputs = model.generate(**inputs, **gen_kwargs)
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_tokens = outputs[0][prompt_len:]
    try:
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    except TypeError:
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Stop on leaked legacy role markers. They are ordinary text for the Qwen model,
    # but can occasionally be generated by older/custom-format models.
    for marker in ("<|user|>", "<|assistant|>", "<|endoftext|>"):
        if marker in reply:
            reply = reply.split(marker, 1)[0].strip()
    reply = reply.replace("\x00", "").strip()
    if not reply or not any(ch.isalnum() for ch in reply):
        return "I'm here with you. Could you tell me a little more?"
    return reply


async def _synthesize(text: str, voice: str, out_file: str) -> None:
    await edge_tts.Communicate(text, voice).save(out_file)


def create_audio(text: str, voice: str = DEFAULT_VOICE) -> str | None:
    """Return an MP3 path for Gradio; on failure let the text reply still succeed."""
    if not HAS_TTS:
        print("edge-tts is not installed; continuing without audio.")
        return None
    if not text.strip():
        return None
    try:
        with tempfile.NamedTemporaryFile(prefix="sora_", suffix=".mp3", delete=False) as handle:
            out_file = handle.name
        asyncio.run(_synthesize(text, voice, out_file))
        return out_file
    except Exception as exc:
        print(f"Voice generation failed (text answer is unaffected): {exc}")
        try:
            if "out_file" in locals() and os.path.exists(out_file):
                os.remove(out_file)
        except OSError:
            pass
        return None


def speak_cli(text: str, voice: str = DEFAULT_VOICE) -> None:
    path = create_audio(text, voice)
    if not path:
        return
    try:
        if is_notebook():
            from IPython.display import Audio, display
            display(Audio(path, autoplay=True))
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        elif sys.platform.startswith("linux"):
            if subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=False).returncode != 0:
                print(f"[Voice saved to {path}]")
        elif sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            print(f"[Voice saved to {path}]")
    except Exception as exc:
        print(f"Audio playback failed: {exc}. Audio file: {path}")


def _format_gradio_history(history: Any, use_messages: bool) -> list:
    normalized = _clean_history(history)
    if use_messages:
        return normalized
    pairs: list[tuple[str | None, str | None]] = []
    pending_user: str | None = None
    for item in normalized:
        if item["role"] == "user":
            if pending_user is not None:
                pairs.append((pending_user, None))
            pending_user = item["content"]
        else:
            pairs.append((pending_user, item["content"]))
            pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, None))
    return pairs


def run_cli(model, tokenizer, voice: bool = False, voice_id: str = DEFAULT_VOICE) -> None:
    history: list[dict[str, str]] = []
    opening = "Hey... you're here. I'm really glad. What's on your mind today?"
    print("\nSora — Your Older Sister (type 'clear' to reset, 'quit' to exit)")
    print(f"Sora: {opening}")
    if voice:
        speak_cli(opening, voice_id)
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSora: Take care of yourself, okay? I'll be right here whenever you need me.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "bye"}:
            print("Sora: Take care of yourself. Come back whenever you need to talk.")
            break
        if user_text.lower() == "clear":
            history = []
            print("Sora: Fresh start. I'm still here.")
            continue
        reply = generate_reply(model, tokenizer, history, user_text)
        print(f"\nSora: {reply}")
        history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])
        history = history[-12:]
        if voice:
            speak_cli(reply, voice_id)


def run_gradio(model, tokenizer, voice: bool = False, voice_id: str = DEFAULT_VOICE, share: bool = True) -> None:
    if not HAS_GRADIO:
        raise SystemExit("Gradio is not installed. Install requirements.txt first.")
    major = 0
    try:
        major = int(str(getattr(gr, "__version__", "0")).split(".", 1)[0])
    except (TypeError, ValueError):
        pass
    try:
        supports_type = "type" in inspect.signature(gr.Chatbot.__init__).parameters
    except (TypeError, ValueError):
        supports_type = False
    use_messages = major >= 5

    def respond(message, chat_history):
        user_text = str(message or "").strip()
        history = _clean_history(chat_history)
        if not user_text:
            return _format_gradio_history(chat_history, use_messages), None
        reply = generate_reply(model, tokenizer, history, user_text)
        if use_messages:
            out_history = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ]
        else:
            out_history = _format_gradio_history(history, use_messages) + [(user_text, reply)]
        audio_path = create_audio(reply, voice_id) if voice else None
        return out_history, audio_path

    def clear():
        return [], None

    with gr.Blocks(title="Sora — Your Older Sister") as demo:
        gr.Markdown(
            "# Sora — Your Older Sister\n"
            "*Created by Mahasvin S S (Snow White)*\n\n"
            "A warm, caring chatbot you can fine-tune yourself."
        )
        chatbot_kwargs = {"height": 450, "label": "Conversation"}
        if use_messages and supports_type:
            chatbot_kwargs["type"] = "messages"
        chatbot = gr.Chatbot(**chatbot_kwargs)
        with gr.Row():
            message = gr.Textbox(placeholder="Tell your big sister anything...", show_label=False, scale=4)
            send = gr.Button("Send", variant="primary", scale=1)
        audio = gr.Audio(label="Sora's voice", visible=voice, autoplay=True, type="filepath")
        clear_button = gr.Button("Clear conversation")

        def send_and_clear(msg, history):
            new_history, audio_file = respond(msg, history)
            return new_history, audio_file, ""

        send.click(send_and_clear, [message, chatbot], [chatbot, audio, message])
        message.submit(send_and_clear, [message, chatbot], [chatbot, audio, message])
        clear_button.click(clear, None, [chatbot, audio])

        opening = "Hey... you're here. I'm really glad. What's on your mind today?"
        if use_messages:
            initial = [{"role": "assistant", "content": opening}]
        else:
            initial = [(None, opening)]
        demo.load(lambda: (initial, None), None, [chatbot, audio])

    print(f"Launching Gradio {getattr(gr, '__version__', '?')} | messages format: {use_messages}")
    demo.launch(share=share, quiet=False)


def main() -> None:
    enforce_dependencies()
    parser = argparse.ArgumentParser(description="Chat with Sora")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", action="store_true", help="Generate voice for replies")
    parser.add_argument("--voice_id", default=DEFAULT_VOICE)
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--no-share", action="store_true", help="Do not create a public Gradio share URL")
    args = parser.parse_args()
    model, tokenizer = load_model(args.adapter, args.base_model)
    if args.gradio or is_notebook():
        run_gradio(model, tokenizer, args.voice, args.voice_id, share=not args.no_share)
    else:
        run_cli(model, tokenizer, args.voice, args.voice_id)


if __name__ == "__main__":
    main()
