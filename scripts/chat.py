#!/usr/bin/env python3
"""
Sora — Older Sister Chatbot
Created by Mahasvin S S (Snow White)

Run this after training (or it falls back to the base model).
Supports optional female voice via edge-tts.
Also supports a Gradio web UI (--gradio) that works well in Colab / notebooks.
"""

import argparse
import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional LoRA (only needed when a trained adapter exists)
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    PeftModel = None

# Optional TTS
try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    edge_tts = None

# Optional Gradio UI
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None

# Optional IPython (Colab / notebooks)
try:
    from IPython.display import Audio, display as ipy_display  # noqa: F401
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# ---------------------------------------------------------------------------
# Defaults (must match train.py defaults)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "microsoft/DialoGPT-small"
DEFAULT_ADAPTER = ROOT / "models" / "sora-lora" / "final"
DEFAULT_VOICE = "en-US-AriaNeural"

def _disable_torchao_probe():
    """Neutralise peft's torchao dispatcher so an old Colab torchao (0.10) cannot crash LoRA.

    peft stores dispatcher *function objects* in a list at import time. Patching the module
    attribute alone is not enough — we must also replace those list entries in-place.
    """
    import importlib
    import types

    def _false(*args, **kwargs):
        return False

    def _noop(*args, **kwargs):
        return None  # "not handled" → next dispatcher

    # 1) Make the availability check always False (never raise)
    for modname in (
        "peft.import_utils",
        "peft.tuners.lora.torchao",
        "peft.tuners.lora.model",
        "peft.tuners.lora",
        "peft",
    ):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "is_torchao_available"):
                setattr(mod, "is_torchao_available", _false)
            if hasattr(mod, "dispatch_torchao"):
                setattr(mod, "dispatch_torchao", _noop)
        except Exception:
            pass

    # 2) Rebuild dispatcher lists so they no longer contain the old torchao function
    try:
        from peft.tuners.lora import model as lora_model
        for attr in ("dispatchers", "DISPATCHERS", "_dispatchers"):
            if not hasattr(lora_model, attr):
                continue
            lst = getattr(lora_model, attr)
            if not isinstance(lst, (list, tuple)):
                continue
            cleaned = []
            for d in lst:
                name = getattr(d, "__name__", "") or ""
                mod = getattr(d, "__module__", "") or ""
                if "torchao" in name.lower() or "torchao" in mod.lower():
                    continue  # drop
                cleaned.append(d)
            # Also inject our noop at the front just in case
            setattr(lora_model, attr, cleaned)
    except Exception:
        pass

    # 3) Last resort: if torchao is importable but too old, hide the module
    try:
        import sys
        import torchao
        ver = getattr(torchao, "__version__", "0.0.0")
        parts = []
        for x in ver.split("."):
            try:
                parts.append(int(x))
            except ValueError:
                break
        if parts and parts[0] == 0 and (len(parts) < 2 or parts[1] < 16):
            # Pretend torchao is not installed for the rest of this process
            sys.modules["torchao"] = None
    except Exception:
        pass

def is_notebook() -> bool:
    """Detect Jupyter / Colab so we can auto-launch Gradio and avoid CLI hangs."""
    try:
        shell = get_ipython().__class__.__name__  # noqa: F821
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or Colab
        if shell == "TerminalInteractiveShell":
            return False  # Terminal IPython
        return False
    except NameError:
        return False


def enforce_dependencies():
    """Fail early with a clean message if core packages are missing."""
    # peft is only required when a trained adapter is present (handled at load time)
    required = {"torch", "transformers"}
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        sys.exit(
            f"Missing required modules: {missing}. "
            "Please install them (pip install -r requirements.txt --upgrade-strategy only-if-needed)."
        )


def load_model(adapter_path: str = None, base_model: str = BASE_MODEL):
    """Load base model + optional LoRA adapter with correct embedding size."""
    print("Loading model...")
    adapter_exists = adapter_path and Path(adapter_path).exists()

    # Prefer tokenizer from adapter (has special tokens + correct vocab size)
    tok_source = adapter_path if adapter_exists else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure special tokens exist even if loading from base
    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    tokenizer.add_special_tokens(special_tokens)

    _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _kwargs = dict(
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # transformers 5.x renamed torch_dtype -> dtype; support both
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=_dtype, **_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=_dtype, **_kwargs)

    # Bulletproof embedding resize: check first, resize only if needed, then verify
    # Must happen BEFORE loading the Peft adapter
    vocab_size = len(tokenizer)
    model_embeddings = model.get_input_embeddings().weight.shape[0]
    if model_embeddings != vocab_size:
        print(f"Resizing embeddings: {model_embeddings} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)
    assert model.get_input_embeddings().weight.shape[0] == vocab_size, (
        "Critical Error: Model embedding size still does not match tokenizer vocabulary."
    )
    print(f"Model embeddings aligned to {vocab_size} tokens.")

    if adapter_exists:
        if not HAS_PEFT:
            print("WARNING: Adapter found but 'peft' is not installed. Falling back to base model.")
            print("  Install with: pip install peft")
        else:
            print(f"Loading LoRA adapter from {adapter_path}")
            _disable_torchao_probe()
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # merge for faster inference
    else:
        print("No trained adapter found — using base model (still works, just less 'sister-like').")

    model.eval()
    return model, tokenizer


def build_prompt(history: list, user_input: str) -> str:
    """Build the conversation string in the same format used during training."""
    text = ""
    for role, content in history:
        if role == "user":
            text += f"<|user|>\n{content}\n"
        else:
            text += f"<|assistant|>\n{content}\n"
    text += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return text


@torch.no_grad()
def generate_reply(model, tokenizer, history, user_input, max_new_tokens=150):
    prompt = build_prompt(history, user_input)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.75,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract only the new assistant part
    if "<|assistant|>" in full:
        reply = full.split("<|assistant|>")[-1]
        reply = reply.split("<|user|>")[0].split("<|endoftext|>")[0].strip()
    else:
        reply = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    return reply or "I'm here, little one. Tell me more."


async def _synthesize(text: str, voice: str, out_file: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_file)


def speak(text: str, voice: str = DEFAULT_VOICE, out_file: str = None):
    """Synthesize speech and play it (works on desktop + Colab/notebooks).

    Gracefully degrades to text-only if edge-tts or playback fails.
    """
    try:
        if not HAS_TTS:
            print("\n[Audio Module Missing]: edge-tts not installed. Falling back to text-only.")
            return None

        if out_file is None:
            out_file = tempfile.mktemp(suffix=".mp3")

        asyncio.run(_synthesize(text, voice, out_file))
        print(f"[Voice saved → {out_file}]")

        # 1) Prefer IPython / Colab audio player (works in notebooks)
        if HAS_IPYTHON or is_notebook():
            try:
                from IPython.display import Audio, display as ipy_display
                ipy_display(Audio(out_file, autoplay=True))
                return out_file
            except Exception as e:
                print(f"\n[Audio Playback Failed]: {e}. Continuing with text-only.")

        # 2) Desktop fallbacks
        if sys.platform.startswith("linux"):
            os.system(
                f"ffplay -nodisp -autoexit -loglevel quiet {out_file} 2>/dev/null "
                f"|| aplay {out_file} 2>/dev/null || true"
            )
        elif sys.platform == "darwin":
            os.system(f"afplay {out_file}")
        elif sys.platform == "win32":
            os.system(f"start {out_file}")

        return out_file

    except ImportError as e:
        print(f"\n[Audio Module Missing]: {e}. Falling back to text-only.")
        return None
    except Exception as e:
        print(f"\n[Audio Playback Failed]: {e}. Continuing with text-only.")
        return None


def run_cli(model, tokenizer, voice: bool = False, voice_id: str = DEFAULT_VOICE):
    history = []
    print("\n" + "=" * 50)
    print("  Sora is online. Type 'quit' or 'exit' to leave.")
    print("  Type 'clear' to reset conversation.")
    print("  Created by Mahasvin S S (Snow White)")
    print("=" * 50 + "\n")

    opening = "Hey... you're here. I'm really glad. What's on your mind today?"
    print(f"Sora: {opening}")
    if voice:
        speak(opening, voice_id)

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSora: Take care of yourself, okay? I'll be right here whenever you need me.")
            break

        if not user:
            continue
        if user.lower() in {"quit", "exit", "bye"}:
            print("Sora: Goodbye for now... come back soon. I love you.")
            if voice:
                speak("Goodbye for now... come back soon. I love you.", voice_id)
            break
        if user.lower() == "clear":
            history = []
            print("Sora: Okay, fresh start. I'm still here.")
            continue

        reply = generate_reply(model, tokenizer, history, user)
        print(f"\nSora: {reply}")

        history.append(("user", user))
        history.append(("assistant", reply))

        # Keep history from growing forever
        if len(history) > 12:
            history = history[-12:]

        if voice:
            speak(reply, voice_id)


def run_gradio(model, tokenizer, voice: bool = False, voice_id: str = DEFAULT_VOICE):
    """Launch Gradio UI — auto-adapts to Gradio 4.x (tuples) and 5/6.x (messages)."""
    if not HAS_GRADIO:
        print("gradio is not installed. Run: pip install gradio")
        print("Falling back to CLI...")
        run_cli(model, tokenizer, voice, voice_id)
        return

    import inspect

    # Gradio 5+ uses role/content messages. Gradio 6 may no longer expose
    # the legacy `type` constructor argument, so detect the major version
    # independently and only pass `type="messages"` when it is supported.
    try:
        major = int(str(getattr(gr, "__version__", "0")).split(".", 1)[0])
    except (TypeError, ValueError):
        major = 0
    try:
        supports_type = "type" in inspect.signature(gr.Chatbot.__init__).parameters
    except (TypeError, ValueError):
        supports_type = False
    use_messages = major >= 5 or (major == 0 and supports_type)

    history_store = []  # list of (role, content) for the model

    def _format_history(pairs):
        """pairs = list of (user, assistant) or full messages list."""
        if use_messages:
            out = []
            for item in pairs:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    u, a = item
                    if u is not None:
                        out.append({"role": "user", "content": u})
                    if a is not None:
                        out.append({"role": "assistant", "content": a})
            return out
        else:
            # Gradio 4 tuples
            out = []
            for item in pairs:
                if isinstance(item, dict):
                    # convert message dicts back to pairs is harder; accumulate
                    pass
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    out.append(tuple(item))
            return out

    def respond(message, chat_history):
        nonlocal history_store
        if not message or not str(message).strip():
            return chat_history or ([] if use_messages else []), None

        user_text = str(message).strip()
        reply = generate_reply(model, tokenizer, history_store, user_text)
        history_store.append(("user", user_text))
        history_store.append(("assistant", reply))
        if len(history_store) > 12:
            history_store = history_store[-12:]

        chat_history = list(chat_history or [])
        if use_messages:
            # Ensure existing history is in messages format
            msgs = []
            for item in chat_history:
                if isinstance(item, dict) and "role" in item:
                    msgs.append({"role": item["role"], "content": item.get("content", "")})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    u, a = item
                    if u is not None:
                        msgs.append({"role": "user", "content": u})
                    if a is not None:
                        msgs.append({"role": "assistant", "content": a})
            msgs.append({"role": "user", "content": user_text})
            msgs.append({"role": "assistant", "content": reply})
            result_history = msgs
        else:
            result_history = chat_history + [(user_text, reply)]

        audio_path = None
        if voice and HAS_TTS:
            audio_path = speak(reply, voice_id)

        return result_history, audio_path

    def clear():
        nonlocal history_store
        history_store = []
        return [], None

    with gr.Blocks(title="Sora — Older Sister Chatbot") as demo:
        gr.Markdown(
            "# Sora — Your Older Sister\n"
            "*Created by Mahasvin S S (Snow White)*\n\n"
            "A warm, caring chatbot you can fine-tune yourself."
        )
        chat_kwargs = dict(height=450, label="Conversation")
        if use_messages and supports_type:
            chat_kwargs["type"] = "messages"
        try:
            chatbot = gr.Chatbot(**chat_kwargs)
        except TypeError:
            # Some Gradio versions removed the `type` argument. Keep the
            # messages payload on Gradio 5/6; its default format is messages.
            chat_kwargs.pop("type", None)
            chatbot = gr.Chatbot(**chat_kwargs)
            if major < 5:
                use_messages = False

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Tell your big sister anything...",
                show_label=False,
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        audio_out = gr.Audio(label="Sora's voice", visible=voice, autoplay=True)
        clear_btn = gr.Button("Clear conversation")

        send.click(respond, [msg, chatbot], [chatbot, audio_out]).then(
            lambda: "", None, msg
        )
        msg.submit(respond, [msg, chatbot], [chatbot, audio_out]).then(
            lambda: "", None, msg
        )
        clear_btn.click(clear, None, [chatbot, audio_out])

        opening = "Hey... you're here. I'm really glad. What's on your mind today?"
        if use_messages:
            open_val = [{"role": "assistant", "content": opening}]
        else:
            open_val = [(None, opening)]
        demo.load(lambda: (open_val, None), None, [chatbot, audio_out])

    print("Launching Gradio UI... (use the public link if sharing)")
    print(f"  Gradio {getattr(gr, '__version__', '?')} | messages format: {use_messages}")
    launch_kwargs = dict(share=True, quiet=False)
    try:
        if "theme" in inspect.signature(demo.launch).parameters:
            launch_kwargs["theme"] = gr.themes.Soft()
    except Exception:
        pass
    demo.launch(**launch_kwargs)



def main():
    enforce_dependencies()

    parser = argparse.ArgumentParser(description="Chat with Sora")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER))
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--voice", action="store_true", help="Speak replies with female voice")
    parser.add_argument("--voice_id", type=str, default=DEFAULT_VOICE)
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio web UI (recommended for Colab / notebooks)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base_model)

    # Auto-detect notebook / Colab: force Gradio to avoid CLI input hangs
    if is_notebook() and not args.gradio:
        print(
            "Notebook environment detected. "
            "Launching Gradio UI automatically to prevent hangs..."
        )
        run_gradio(model, tokenizer, args.voice, args.voice_id)
    elif args.gradio:
        run_gradio(model, tokenizer, args.voice, args.voice_id)
    else:
        run_cli(model, tokenizer, args.voice, args.voice_id)


if __name__ == "__main__":
    main()
