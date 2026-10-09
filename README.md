# Sora — Trainable Older-Sister AI Chatbot

**Created by Mahasvin S S (Snow White)**

A complete, from-scratch trainable chatbot that becomes a warm, caring older sister.  
You can fine-tune it on your own conversations and talk to it with a soft female voice.

## Features

- Fine-tunes a small causal language model (DialoGPT-small by default) with LoRA
- Custom chat format (`<|user|>` / `<|assistant|>`)
- Expanded conversation dataset (~97 dialogues) in `data/sister_conversations.json`
- Female voice via `edge-tts` (high-quality neural voices)
- **Gradio web UI** (`--gradio`) — works great in Google Colab / notebooks
- Colab-friendly audio playback via `IPython.display.Audio` (uses Colab's built-in IPython — **no restart required**)
- Works on CPU or GPU
- Fully self-contained — no external API keys required
- Proper embedding resize so custom tokens never cause shape mismatches
- Install carefully pinned so Colab does **not** force a runtime restart

## Project Structure

```
sora_chatbot/
├── data/
│   └── sister_conversations.json   ← your training dialogues
├── models/                         ← trained adapters go here
├── scripts/
│   ├── prepare_data.py             ← converts JSON → train/test jsonl
│   ├── train.py                    ← fine-tunes the model
│   └── chat.py                     ← talk to Sora (+ optional voice / Gradio)
├── requirements.txt
├── run.sh
└── README.md
```

## Quick Start (Local)

### 1. Install dependencies

```bash
cd sora_chatbot
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt --upgrade-strategy only-if-needed
```

### 2. Prepare the dataset

```bash
python scripts/prepare_data.py
```

This creates `data/train.jsonl` and `data/test.jsonl`.

### 3. Train (fine-tune)

**On GPU (recommended):**
```bash
python scripts/train.py --epochs 5 --batch_size 4 --use_4bit
```

**On CPU (slower but works):**
```bash
python scripts/train.py --epochs 3 --batch_size 1
```

**Try a stronger base model (if you have GPU memory):**
```bash
python scripts/train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch_size 2 --use_4bit
# or
python scripts/train.py --model_name microsoft/DialoGPT-medium --epochs 3
```

After training you will have:
```
models/sora-lora/final/
```

### 4. Chat with Sora

```bash
# Text only (CLI)
python scripts/chat.py

# With female voice
python scripts/chat.py --voice

# Gradio web UI (recommended for Colab / notebooks)
python scripts/chat.py --gradio

# Gradio + voice
python scripts/chat.py --gradio --voice
```

Available high-quality female voices (change with `--voice_id`):
- `en-US-AriaNeural` (default – warm & natural)
- `en-US-JennyNeural`
- `en-GB-SoniaNeural`
- `en-AU-NatashaNeural`
- `ja-JP-NanamiNeural` (Japanese)
- and many more (run `edge-tts --list-voices` to see all)

## Google Colab — No Restart Required

The previous install upgraded `ipython` / `traitlets` / `psutil` (already imported by Colab), which triggered the restart warning.  
This version **never installs or upgrades those packages**.

Copy-paste these cells:

**Cell 1 – Install (safe, no restart)**
```python
# Upload the sora_chatbot folder first (or unzip it), then:
%cd /content/sora_chatbot   # or wherever you put it

!pip install -q -r requirements.txt --upgrade-strategy only-if-needed
# only-if-needed = leave existing transformers / huggingface-hub / gradio / diffusers alone
# (we deliberately omit ipython / traitlets / psutil so Colab does not ask for a restart)
```

**Cell 2 – Prepare data**
```python
!python scripts/prepare_data.py
```

**Cell 3 – Train (pick one)**
```python
# GPU (recommended)
!python scripts/train.py --epochs 4 --batch_size 4 --use_4bit

# or CPU (slower)
# !python scripts/train.py --epochs 3 --batch_size 1
```

**Cell 4 – Chat (Gradio + optional voice)**
```python
!python scripts/chat.py --gradio --voice
```

The Gradio link will appear. Voice plays automatically inside the notebook via Colab’s built-in IPython (no restart needed).

If you already ran an old install that upgraded IPython, just ignore the old warning — the new install no longer touches those packages.

## Adding Your Own Training Data

Edit `data/sister_conversations.json`.  
Each entry is a multi-turn conversation:

```json
{
  "messages": [
    {"role": "user", "content": "Your message here"},
    {"role": "assistant", "content": "Sora's warm sister reply"}
  ]
}
```

You can have as many turns as you want inside one conversation.  
After editing, re-run:

```bash
python scripts/prepare_data.py
python scripts/train.py
```

## Fixes Applied (v3 → v4)

| Issue | Fix |
|-------|-----|
| Colab “You must restart the runtime” | Removed `ipython` (and friends) from requirements; install instructions never upgrade already-imported Colab packages |
| Dependency conflicts (google-colab / diffusers / gradio) | No upper pins. Install with `--upgrade-strategy only-if-needed` so existing packages stay untouched |
| Gradio 6 chat format crash | Detect Gradio major version separately from constructor signature; Gradio 5/6 always receive `{"role","content"}` message dictionaries, including when `type` was removed from `Chatbot` |
| peft / torchao version crash | Monkey-patch `is_torchao_available()` so standard LoRA works without upgrading torchao |
| `torch_dtype` deprecation (transformers 5) | Try `dtype=` first, fall back to `torch_dtype=` |
| Audio / TTS crashes | Full try/except around edge-tts + playback; falls back to text-only |
| Missing dependencies mid-run | `enforce_dependencies()` exits early with a clear install hint |
| Embedding / vocab size mismatch | Pre-flight check + resize + assert in both train.py and chat.py |
| Colab CLI hangs | Auto-detects Jupyter/Colab and forces Gradio UI |
| TrainingArguments version breakage | Dynamic `eval_strategy` / `evaluation_strategy` + safe `warmup_steps` |
| Small dataset | ~97 high-quality sister dialogues included |

## Troubleshooting: Gradio says “Data incompatible with messages format”

If the log reports Gradio 5 or 6, the chatbot must return a list of dictionaries such as `{"role": "user", "content": "Hello"}`. Do not force tuple history just because `gr.Chatbot.__init__` no longer has a `type` parameter; newer Gradio versions use messages by default. The current `scripts/chat.py` detects this separately. Restart the Python/Colab runtime after replacing the script, then launch `python scripts/chat.py --gradio` again.

## Tips for Better Results

1. **More data = better sister**  
   Aim for 100+ good dialogues. The sample now has ~97 to get you started.

2. **Style consistency**  
   Keep the assistant replies in the same warm, slightly teasing, protective tone.

3. **Stronger base models**  
   If you have a GPU with enough VRAM, try:
   - `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`
   - `meta-llama/Llama-3.2-1B` (requires HF token for gated models)
   - `microsoft/DialoGPT-medium`

4. **Voice**  
   edge-tts needs an internet connection the first time it downloads a voice. After that it works offline for that voice.

## Requirements

- Python 3.10+
- ~2–4 GB RAM for DialoGPT-small
- Optional: NVIDIA GPU with CUDA for faster training

## License

Do whatever you want with this code. It’s yours.

---

**Created by Mahasvin S S (Snow White)**  
Made so you can have your own older sister that you trained yourself.
