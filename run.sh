#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt --upgrade-strategy only-if-needed
python scripts/prepare_data.py
if [ ! -f models/sora-qwen-lora/final/adapter_config.json ]; then
  echo "No trained Qwen adapter found. Train it first with:"
  echo "  python scripts/train.py --epochs 5 --batch_size 2 --gradient_accumulation_steps 2 --lr 2e-4"
  echo "You may still test the un-fine-tuned base model using: python scripts/chat.py --gradio"
  exit 2
fi
python scripts/chat.py --gradio "$@"
