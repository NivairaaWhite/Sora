#!/usr/bin/env python3
"""Run a small fixed prompt suite and save Sora's replies for manual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chat import DEFAULT_ADAPTER, generate_reply, load_model
from common import DEFAULT_MODEL

PROMPTS = [
    "Hello, Sora!",
    "Who created you?",
    "I'm anxious about my exams. Can you help me make a small plan?",
    "I had an argument with my friend. What should I do?",
    "Can you explain death in a thoughtful, gentle way?",
    "Thank you, Sister.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a repeatable smoke-test report for Sora")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="./evaluation_results.json")
    args = parser.parse_args()
    model, tokenizer = load_model(args.adapter, args.base_model)
    results = []
    for prompt in PROMPTS:
        reply = generate_reply(model, tokenizer, [], prompt)
        results.append({"prompt": prompt, "reply": reply})
        print(f"\nUSER: {prompt}\nSORA: {reply}\n" + "-" * 68)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved review report to {out.resolve()}")
    print("Read the answers manually; a successful run does not itself prove answer quality.")


if __name__ == "__main__":
    main()
