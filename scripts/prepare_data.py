#!/usr/bin/env python3
"""Validate and split Sora's conversation JSON into structured SFT JSONL."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import make_sft_examples, normalize_messages

ROOT = Path(__file__).resolve().parent.parent


def load_dialogues(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Dataset root must be a JSON list of {messages: [...]} objects.")
    return payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare Sora conversation examples")
    parser.add_argument("--input", default=str(ROOT / "data" / "sister_conversations.json"))
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.05 <= args.test_ratio <= 0.30:
        raise SystemExit("--test_ratio must be between 0.05 and 0.30")

    input_path = Path(args.input)
    dialogues = load_dialogues(input_path)
    # Split original dialogues BEFORE expanding turns so turns from the same conversation
    # cannot leak into both train and test.
    good_dialogues: list[dict] = []
    skipped = 0
    for idx, item in enumerate(dialogues):
        if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
            print(f"WARNING: Dialogue {idx} has no valid messages list; skipped.")
            skipped += 1
            continue
        clean = normalize_messages(item["messages"])
        if not clean or not any(m["role"] == "user" for m in clean) or not any(m["role"] == "assistant" for m in clean):
            print(f"WARNING: Dialogue {idx} lacks a user/assistant pair; skipped.")
            skipped += 1
            continue
        good_dialogues.append({"messages": clean})

    if len(good_dialogues) < 2:
        raise SystemExit("Need at least 2 valid dialogues to create train/test splits.")

    rng = random.Random(args.seed)
    rng.shuffle(good_dialogues)
    test_count = max(1, round(len(good_dialogues) * args.test_ratio))
    test_dialogues = good_dialogues[:test_count]
    train_dialogues = good_dialogues[test_count:]
    train_examples, train_warnings = make_sft_examples(train_dialogues)
    test_examples, test_warnings = make_sft_examples(test_dialogues)
    warnings = train_warnings + test_warnings
    for warning in warnings[:20]:
        print("WARNING:", warning)
    if len(warnings) > 20:
        print(f"WARNING: {len(warnings) - 20} more data-quality warnings omitted.")
    if not train_examples or not test_examples:
        raise SystemExit("Could not create non-empty train and test examples. Check the JSON data.")

    data_dir = ROOT / "data"
    write_jsonl(data_dir / "train.jsonl", train_examples)
    write_jsonl(data_dir / "test.jsonl", test_examples)
    print(f"Valid source dialogues: {len(good_dialogues)} (skipped {skipped})")
    print(f"Train: {len(train_examples)} assistant-turn examples from {len(train_dialogues)} dialogues")
    print(f"Test:  {len(test_examples)} assistant-turn examples from {len(test_dialogues)} dialogues")
    print(f"Wrote structured JSONL to {data_dir / 'train.jsonl'} and {data_dir / 'test.jsonl'}")
    print("Example row:")
    print(json.dumps(train_examples[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
