# Sora — Your Older Sister (Improved Training Edition)

Created by **Mahasvin S S (Snow White)**.

This edition fixes the training/inference format mismatch and upgrades the recommended base model from DialoGPT-small to **Qwen2.5-0.5B-Instruct**. A ready-to-run notebook is included at `notebooks/Sora_Colab_Training.ipynb`. It includes the project code and curated conversation data; it does **not** contain pretrained model weights or a newly trained LoRA adapter. Training downloads the public base model and creates the adapter in your runtime.

## Main changes in this edition

- **Instruction-tuned base model:** defaults to `Qwen/Qwen2.5-0.5B-Instruct` rather than an older non-instruction-tuned dialogue model.
- **One matching format in train and chat:** uses the tokenizer's native `apply_chat_template` whenever the model provides one. This avoids inventing custom role tokens for Qwen.
- **Assistant-only supervised loss:** prompt/system/user tokens are masked with label `-100`; the loss is computed on the assistant target response, not on the whole transcript.
- **More useful examples:** added varied factual, identity, clarification, practical-advice, and supportive responses; updated a few dataset examples to avoid unrealistic promises and improve transparency about being an AI.
- **Safer split:** source dialogues are split before expanding assistant turns, preventing turns from the same source dialogue leaking into both train and test.
- **More useful optimization defaults:** LoRA rank 8, learning rate `2e-4`, 3% warm-up ratio, gradient accumulation, dynamic padding, epoch evaluation, and best-checkpoint selection by evaluation loss.
- **Quantization is optional:** the default 0.5B model is trained without 4-bit quantization. `--use_4bit` enables QLoRA when CUDA and `bitsandbytes` are available.
- **Adapter mismatch protection:** the chat app verifies the model recorded in `adapter_config.json`, and uses a separate `models/sora-qwen-lora/` path so the old DialoGPT adapter is not overwritten.
- **Generation and UI improvements:** model-native prompt formatting, controlled sampling, new-token-only decoding, robust Gradio 5/6 messages, conversation history normalization, and best-effort Edge TTS that will not discard the text reply.
- **Smoke-test script:** run a fixed prompt suite and save answers for manual quality review.

A small fine-tuning set teaches style and some response patterns; it cannot turn a 0.5B model into an expert or guarantee factual accuracy. Always judge the generated answers as well as the loss.

## Recommended Google Colab workflow

Choose **Runtime → Change runtime type → GPU** before training. This workflow deliberately uses a distinct adapter directory so it leaves an old `models/sora-lora/final` adapter untouched.

### 1. Upload and extract the ZIP

Upload `sora_model_improved.zip` to Colab (or place it in `/content/`), then run:

```python
!unzip -o /content/sora_model_improved.zip -d /content/
%cd /content/Sora
```

If you extracted it somewhere else, `cd` into the `Sora` folder containing `scripts/` and `data/`.

### 2. Install dependencies

```python
!pip install -r requirements.txt --upgrade-strategy only-if-needed
```

This intentionally avoids installing/upgrading IPython, traitlets, and psutil, which can trigger Colab runtime restart prompts. If an existing notebook has incompatible pinned packages, install requirements in a clean runtime rather than repeatedly upgrading packages mid-session.

### 3. Prepare and validate the dataset

```python
!python scripts/prepare_data.py
```

The script validates source records and creates structured `data/train.jsonl` and `data/test.jsonl`. Run it again after editing `data/sister_conversations.json`.

### 4. Fine-tune the Qwen model

Recommended first run on a Colab GPU:

```python
!python scripts/train.py --epochs 5 --batch_size 2 --gradient_accumulation_steps 2 --lr 2e-4
```

The default base model is `Qwen/Qwen2.5-0.5B-Instruct`. Training starts with full-precision weights in fp16/bf16 as supported by the GPU; you do **not** need `--use_4bit` for the default 0.5B model on a typical T4. The resulting LoRA adapter is saved to:

```text
models/sora-qwen-lora/final/
```

For tight GPU memory, first try `--batch_size 1`. If you specifically want QLoRA and have a compatible CUDA runtime, install `bitsandbytes` and add `--use_4bit`:

```python
!pip install bitsandbytes
!python scripts/train.py --epochs 5 --batch_size 2 --gradient_accumulation_steps 2 --lr 2e-4 --use_4bit
```

Do not run multiple trainers simultaneously. Keep the browser tab/session active until training finishes; save or download the adapter if you need it after the Colab runtime is reset.

### 5. Evaluate actual responses

```python
!python scripts/evaluate.py
```

This prints six consistent prompts and saves `evaluation_results.json`. Read the answers manually. A falling loss is useful, but coherent, relevant replies and a reasonable evaluation curve matter more than hitting one magic loss number.

### 6. Launch Gradio and voice

```python
!python scripts/chat.py --gradio --voice
```

The default voice is `en-US-AriaNeural`; change it with `--voice_id en-US-JennyNeural`. The Gradio share URL is public to people who have the link while it is active. For local/private-only launch, use `--no-share`.

## Run locally

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade-strategy only-if-needed
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py
python scripts/chat.py --gradio
```

## Using another base model

Use the **same** `--model_name` in training and `--base_model` at inference. Example:

```bash
python scripts/train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --output_dir ./models/sora-qwen-lora --epochs 5
python scripts/chat.py --base_model Qwen/Qwen2.5-0.5B-Instruct --adapter ./models/sora-qwen-lora/final --gradio
```

Do not load the old `microsoft/DialoGPT-small` adapter with the new Qwen base. The chat program intentionally stops with a clear error when it detects a base-model mismatch.

## Project layout

```text
Sora/
├── data/
│   ├── sister_conversations.json
│   ├── train.jsonl                 # generated by prepare_data.py
│   └── test.jsonl                  # generated by prepare_data.py
├── scripts/
│   ├── common.py                   # shared persona + serialization
│   ├── prepare_data.py             # validate / split / structure examples
│   ├── train.py                    # assistant-only LoRA training
│   ├── chat.py                     # CLI + Gradio + Edge TTS
│   └── evaluate.py                 # fixed prompt smoke test
├── tests/
├── notebooks/
│   └── Sora_Colab_Training.ipynb
├── requirements.txt
└── README.md
```

## Troubleshooting

**Colab PEFT / `torchao` version error:** this project disables PEFT's optional torchao LoRA dispatcher during standard LoRA setup, because this training path does not use torchao. If you are running an older copy of the project, re-extract this fixed ZIP or run `!pip uninstall -y torchao` and restart the runtime before training.

**`apply_chat_template` returns `input_ids` / tokenizer conversion error:** the formatting helper accepts list and dictionary/BatchEncoding outputs from different Transformers versions. If you still see the old traceback, Colab is likely using a stale extracted copy; re-extract the fixed ZIP and rerun the notebook from the project directory.


**Gibberish or fragments:** check that `train.py` and `chat.py` use the same base model and that `models/sora-qwen-lora/final/adapter_config.json` exists. Re-run data preparation, train the Qwen adapter from scratch, then run the same evaluation prompts. Do not reuse an adapter trained on the previous DialoGPT architecture.

**Old Gradio messages-format error:** this edition emits `{"role": ..., "content": ...}` records on Gradio 5/6 and `(user, assistant)` pairs on Gradio 4. Replace the script actually being launched and restart the runtime after replacement.

**Training loss does not fall:** inspect the data JSONL and the printed count of assistant-only examples. Confirm the dataset was regenerated from the structured source file. Watch `eval_loss`; stop if it consistently worsens. The train set is still small, and memorization/overfitting is possible.

**Out of GPU memory:** use `--batch_size 1`; lower `--max_length` to 512; or use `--gradient_checkpointing`. QLoRA is optional, not the first fix for this small model.

**Hugging Face anonymous warning:** this is usually a rate-limit/download warning, not a training correctness error. In Colab, store a Hugging Face token in Colab Secrets if authenticated downloads are needed. Never paste a token into a public notebook or share it with the ZIP.

**TTS failed:** text chat should still work. Check internet connectivity and install `edge-tts`; TTS is a separate service and is not needed to train or evaluate the model.

## Important limitations

- This archive contains source code/data, not trained weights. Run fine-tuning to generate your personalized adapter.
- With roughly 120 short conversations, the dataset is suitable for a prototype. It will not reliably cover every topic. Keep expanding it with accurate, varied, reviewed conversations.
- `evaluation_results.json` is a manual-review aid, not an automated benchmark or a guarantee of safety/factuality.
- Avoid sharing notebooks containing credentials or private conversations. The Gradio `--share` link is public to anyone with the link until it expires.

## License note

The project scripts and sample dialogue data are provided for personal experimentation. Check the base model's model card and license before redistributing trained adapters or deploying commercially.

---
**Sora — Your Older Sister**  
Created by Mahasvin S S (Snow White).
