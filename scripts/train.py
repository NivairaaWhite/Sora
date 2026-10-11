#!/usr/bin/env python3
"""Assistant-only LoRA fine-tuning for Sora; default base is Qwen2.5-0.5B-Instruct."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import DEFAULT_MODEL
from sft_format import assistant_only_features

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Sora using assistant-only supervised loss")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Base model; defaults to Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="./models/sora-qwen-lora", help="Output adapter directory (new Qwen path by default)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", help="Optional QLoRA; requires CUDA + bitsandbytes")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Reduce activation memory; may train more slowly")
    return parser.parse_args()


class SFTPaddingCollator:
    """Dynamic padding that preserves assistant-only loss labels."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_masks.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Expected a {{'messages': [...]}} row at {path}:{line_no}. Run scripts/prepare_data.py again.")
            rows.append(row)
    return rows


def _target_modules(model) -> list[str]:
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = [name for name in candidates if name in names]
    if found:
        return found
    candidates = ["c_attn", "c_proj", "c_fc"]
    found = [name for name in candidates if name in names]
    if found:
        return found
    raise RuntimeError("Could not detect LoRA target modules for this architecture. Try Qwen/Qwen2.5-0.5B-Instruct.")


def _training_args(args, use_cuda: bool, use_bf16: bool, has_eval: bool) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    values: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "load_best_model_at_end": bool(has_eval),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "fp16": bool(use_cuda and not use_bf16),
        "bf16": bool(use_cuda and use_bf16),
    }
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    if strategy_key in params:
        values[strategy_key] = "epoch" if has_eval else "no"
    # Some installed versions may not expose optional parameters.
    values = {k: v for k, v in values.items() if k in params}
    return TrainingArguments(**values)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("epochs, batch_size, and gradient_accumulation_steps must all be >= 1")
    if args.max_length < 128:
        raise SystemExit("--max_length must be >= 128")
    if args.use_4bit and not torch.cuda.is_available():
        raise SystemExit("--use_4bit requires a CUDA GPU. For the default 0.5B model, try training without it.")

    train_path, test_path = ROOT / "data" / "train.jsonl", ROOT / "data" / "test.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Prepared data missing. Run: python scripts/prepare_data.py")
    raw_train, raw_eval = _load_jsonl(train_path), _load_jsonl(test_path)
    if not raw_train or not raw_eval:
        raise SystemExit("Train/test data is empty. Check data/sister_conversations.json.")

    print(f"Loading {args.model_name} (Transformers {transformers.__version__})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token nor eos_token; please use an instruction-tuned causal LM.")
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        print("WARNING: This tokenizer has no native chat template; using Sora's legacy role-marker format.")

    model_kwargs: dict[str, Any] = {"trust_remote_code": False}
    use_cuda = torch.cuda.is_available()
    bf16_supported = bool(use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else torch.float32)
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if not args.use_4bit and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    targets = _target_modules(model)
    print(f"LoRA target modules: {targets}")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    # Standard LoRA does not use torchao. PEFT versions bundled in some Colab
    # runtimes may detect an old preinstalled torchao and raise during dispatch.
    # Disable only the optional torchao LoRA dispatcher; normal LoRA remains enabled.
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

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def convert_rows(rows: list[dict], label: str) -> Dataset:
        raw = Dataset.from_list(rows)
        mapped = raw.map(
            lambda row: assistant_only_features(row["messages"], tokenizer, args.max_length) or {
                "input_ids": [], "attention_mask": [], "labels": []
            },
            remove_columns=raw.column_names,
            desc=f"Tokenizing {label} (assistant-only loss)",
        )
        mapped = mapped.filter(lambda row: len(row["input_ids"]) > 0, desc=f"Filtering invalid {label} rows")
        return mapped

    train_ds, eval_ds = convert_rows(raw_train, "training data"), convert_rows(raw_eval, "evaluation data")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise SystemExit("No valid assistant training targets found. Re-run prepare_data.py and inspect your source JSON.")
    print(f"Training examples: {len(train_ds)} | evaluation examples: {len(eval_ds)}")
    print("Loss is masked on system/user context and computed on assistant response tokens only.")

    training_args = _training_args(args, use_cuda, bf16_supported, has_eval=len(eval_ds) > 0)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": SFTPaddingCollator(tokenizer.pad_token_id),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print("\nStarting fine-tuning. Monitor both eval loss and generated answers; no fixed loss target is guaranteed.")
    train_result = trainer.train()
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # Disable cache only during checkpointed training; inference should use the normal cache.
    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = True
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    (final_dir / "training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=str), encoding="utf-8"
    )
    (final_dir / "training_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2, default=str), encoding="utf-8"
    )
    metadata = {
        "base_model": args.model_name,
        "training_objective": "assistant_only_causal_lm",
        "epochs_requested": args.epochs,
        "max_length": args.max_length,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
    }
    (final_dir / "sora_training_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Adapter and tokenizer saved to: {final_dir}")
    print("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
