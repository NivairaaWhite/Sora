#!/usr/bin/env python3
"""
Prepare the sister conversation dataset for causal language modeling.
Created by Mahasvin S S (Snow White)

Converts multi-turn dialogues into a single text format that the model can learn from.
"""

import json
import os
from pathlib import Path
from datasets import Dataset

def format_conversation(messages):
    """Turn a list of role/content dicts into a single training string."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            text += f"<|user|>\n{content}\n"
        else:
            text += f"<|assistant|>\n{content}\n"
    text += "<|endoftext|>"
    return text

def main():
    data_path = Path(__file__).parent.parent / "data" / "sister_conversations.json"
    output_dir = Path(__file__).parent.parent / "data"

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts = [format_conversation(item["messages"]) for item in raw]

    # Create a simple HF Dataset
    ds = Dataset.from_dict({"text": texts})
    ds = ds.train_test_split(test_size=0.1, seed=42)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    ds["train"].to_json(train_path)
    ds["test"].to_json(test_path)

    print(f"Prepared {len(ds['train'])} training examples → {train_path}")
    print(f"Prepared {len(ds['test'])} test examples → {test_path}")
    print("\nExample training text:")
    print(ds["train"][0]["text"][:400] + "...")

if __name__ == "__main__":
    main()
