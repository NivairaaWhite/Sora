#!/bin/bash
# Simple launcher for Sora

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt --upgrade-strategy only-if-needed
else
    source venv/bin/activate
fi

# Prepare data if needed
if [ ! -f "data/train.jsonl" ]; then
    echo "Preparing dataset..."
    python scripts/prepare_data.py
fi

# Check if model exists
if [ -d "models/sora-lora/final" ]; then
    echo "Trained model found. Starting chat..."
    python scripts/chat.py "$@"
else
    echo "No trained model yet."
    echo "Would you like to train now? (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        python scripts/train.py --epochs 3 --batch_size 1
        python scripts/chat.py "$@"
    else
        echo "Starting with base model (not yet fine-tuned)..."
        python scripts/chat.py "$@"
    fi
fi
