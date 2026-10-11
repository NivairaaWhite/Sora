# Changelog

## Improved training edition

- Switched default model to `Qwen/Qwen2.5-0.5B-Instruct` and gave it an independent adapter directory.
- Added model-native chat templating in both training and inference.
- Added assistant-only loss masking, dynamic padding and a prefix-alignment safeguard.
- Removed the fixed 50-step warm-up; introduced a short ratio-based warm-up and sensible LoRA defaults.
- Added optional, correctly prepared QLoRA path; kept 4-bit quantization off by default.
- Expanded and curated dataset examples and made the data preparation step structured, validated, reproducible and leakage-aware.
- Added strict model/adapter compatibility checking, controlled generation, Gradio 5/6 payload handling, and best-effort Edge TTS.
- Added repeatable response smoke tests, saved training metadata/metrics, unit tests, and a Colab notebook.
