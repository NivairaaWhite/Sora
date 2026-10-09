#!/usr/bin/env python3
"""
Fine-tune a small causal LM to become Sora — your older-sister chatbot.
Created by Mahasvin S S (Snow White)

Uses LoRA so you can train even on a single consumer GPU (or CPU if patient).
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

import torch
import transformers
from packaging import version
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

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

def enforce_dependencies():
    """Fail early with a clean message if core packages are missing."""
    required = {"torch", "transformers", "peft", "datasets", "packaging"}
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        sys.exit(
            f"Missing required modules: {missing}. "
            "Please install them (pip install -r requirements.txt --upgrade-strategy only-if-needed)."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sora (older-sister chatbot)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Base model to fine-tune (small = faster, medium = smarter). "
             "Also try: Qwen/Qwen2.5-0.5B-Instruct, meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/sora-lora",
        help="Where to save the trained adapter",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load model in 4-bit (needs bitsandbytes + CUDA)",
    )
    return parser.parse_args()


def main():
    enforce_dependencies()
    args = parse_args()
    root = Path(__file__).parent.parent
    train_file = root / "data" / "train.jsonl"
    test_file = root / "data" / "test.jsonl"

    if not train_file.exists():
        print("Dataset not found. Run prepare_data.py first.")
        return

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # DialoGPT / many models need a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Special tokens for our chat format
    special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens. Vocab size now: {len(tokenizer)}")

    model_kwargs = {"trust_remote_code": True}
    if args.use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # transformers 5.x renamed torch_dtype -> dtype; support both
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=_dtype, **model_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=_dtype, **model_kwargs
        )

    # Bulletproof embedding resize: check first, resize only if needed, then verify
    vocab_size = len(tokenizer)
    model_embeddings = model.get_input_embeddings().weight.shape[0]
    if model_embeddings != vocab_size:
        print(f"Resizing embeddings: {model_embeddings} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)
    assert model.get_input_embeddings().weight.shape[0] == vocab_size, (
        "Critical Error: Model embedding size still does not match tokenizer vocabulary."
    )

    # LoRA config — very light, trains fast
    # Target modules adapt to common architectures
    model_name_lower = args.model_name.lower()
    if "gpt" in model_name_lower or "dialogpt" in model_name_lower:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif "llama" in model_name_lower or "qwen" in model_name_lower or "mistral" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    _disable_torchao_probe()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    raw = load_dataset(
        "json",
        data_files={"train": str(train_file), "test": str(test_file)},
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Version-agnostic TrainingArguments: warmup_steps is universally safe;
    # eval_strategy key changed in transformers 4.41.
    trans_version = version.parse(transformers.__version__)
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "warmup_steps": 50,  # safer than warmup_ratio across versions
        "logging_steps": 5,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "save_total_limit": 2,
    }
    if trans_version >= version.parse("4.41.0"):
        training_args_dict["eval_strategy"] = "epoch"
    else:
        training_args_dict["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save the LoRA adapter + tokenizer (with expanded vocab)
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining complete. Adapter + tokenizer saved to: {final_dir}")


if __name__ == "__main__":
    main()
