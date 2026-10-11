#!/usr/bin/env python3
"""Shared model, persona, and conversation-format helpers for Sora."""
from __future__ import annotations

from typing import Any, Iterable

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are Sora, a warm, caring, emotionally supportive older-sister chatbot. "
    "Your creator is Mahasvin S S (Snow White). Be natural, kind, honest, and concise. "
    "Answer the user's actual question; do not continue a story unless asked. "
    "Never invent personal memories or claim certainty when unsure. Avoid repeating the user's words."
)

VALID_ROLES = {"user", "assistant"}


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Validate/normalize message dictionaries while preserving turn order."""
    if not isinstance(messages, list):
        return []
    result: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = item.get("content", "")
        if role not in VALID_ROLES or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            result.append({"role": role, "content": content})
    return result


def legacy_serialize(messages: list[dict[str, str]], include_system: bool = True) -> str:
    """Serialize data for tokenizers that do not ship a chat template."""
    parts: list[str] = []
    if include_system:
        parts.append(SYSTEM_PROMPT.strip() + "\n\n")
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts)


def make_sft_examples(dialogues: Iterable[Any]) -> tuple[list[dict], list[str]]:
    """Turn full dialogues into one supervised example per assistant turn."""
    examples: list[dict] = []
    warnings: list[str] = []
    for dialogue_index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, dict) or not isinstance(dialogue.get("messages"), list):
            warnings.append(f"Dialogue {dialogue_index}: missing a messages list; skipped.")
            continue
        messages = normalize_messages(dialogue["messages"])
        if not messages:
            warnings.append(f"Dialogue {dialogue_index}: no valid messages; skipped.")
            continue
        if messages[0]["role"] != "user":
            warnings.append(f"Dialogue {dialogue_index}: should start with a user message.")
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            context = messages[:i]
            if not context or not any(m["role"] == "user" for m in context):
                warnings.append(f"Dialogue {dialogue_index}: assistant response without a user prompt; skipped.")
                continue
            examples.append({"messages": context + [message]})
    return examples, warnings
