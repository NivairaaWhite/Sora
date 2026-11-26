#!/usr/bin/env python3
"""Tokenizer-facing SFT formatting helpers; deliberately independent of Transformers imports."""
from __future__ import annotations

from typing import Any

from common import SYSTEM_PROMPT, legacy_serialize, normalize_messages


def as_list_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def apply_template_ids(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    """Use a native chat template when available, otherwise a consistent legacy format."""
    if getattr(tokenizer, "chat_template", None):
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        return as_list_ids(tokenized)
    text = legacy_serialize(messages, include_system=False)
    if add_generation_prompt:
        text += "<|assistant|>\n"
    else:
        text += tokenizer.eos_token or ""
    return as_list_ids(tokenizer(text, add_special_tokens=True)["input_ids"])


def assistant_only_features(messages_raw: Any, tokenizer, max_length: int) -> dict | None:
    """Tokenize one conversation; compute loss only on the final assistant response."""
    messages = normalize_messages(messages_raw)
    if not messages or messages[-1]["role"] != "assistant":
        return None
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    prefix_messages = full_messages[:-1]

    if getattr(tokenizer, "chat_template", None):
        full_ids = apply_template_ids(tokenizer, full_messages, add_generation_prompt=False)
        prefix_ids = apply_template_ids(tokenizer, prefix_messages, add_generation_prompt=True)
    else:
        # Legacy architectures receive the same persona and string markers in train/inference.
        full_text = SYSTEM_PROMPT.strip() + "\n\n" + legacy_serialize(messages, include_system=False)
        prefix_text = SYSTEM_PROMPT.strip() + "\n\n" + legacy_serialize(messages[:-1], include_system=False) + "<|assistant|>\n"
        full_ids = as_list_ids(tokenizer(full_text + (tokenizer.eos_token or ""), add_special_tokens=True)["input_ids"])
        prefix_ids = as_list_ids(tokenizer(prefix_text, add_special_tokens=True)["input_ids"])

    common = 0
    for left, right in zip(full_ids, prefix_ids):
        if left != right:
            break
        common += 1
    if len(full_ids) < 2 or common >= len(full_ids):
        return None

    # Preserve the final part of long examples; if truncation starts after the prompt,
    # all remaining tokens are assistant targets rather than user/prompt tokens.
    start = max(0, len(full_ids) - max_length)
    input_ids = full_ids[start:]
    masked_prefix = max(0, common - start)
    labels = [-100] * min(masked_prefix, len(input_ids)) + input_ids[min(masked_prefix, len(input_ids)):]
    if not any(label != -100 for label in labels):
        return None
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
