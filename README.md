# Dia-Spanish: Text-to-Dialogue for Spanish Conversations

> Spanish adaptation of [NARI Labs Dia](https://github.com/nari-labs/dia) — a 1.6B text-to-dialogue model that generates highly realistic multi-speaker conversations from transcripts.

**D8 Labs** · Spanish Text-to-Dialogue

---

## 🎯 Goal

Build a **Spanish text-to-dialogue model** that takes transcripts like:

```
[S1] ¡Hola! ¿Cómo te fue en la reunión? [S2] ¡Uf, súper bien! Estaba nerviosa pero al final todo salió perfecto. (risas) [S1] ¡Qué bueno! Me alegro mucho.
```

And generates **natural Spanish conversational audio** with proper intonation, emotion, and nonverbal cues (laughs, sighs, gasps).

Dia is English-only today. We're changing that.

---

## 🧠 Why This Works

Dia uses **byte-level UTF-8 encoding** — no tokenizer, no vocab bottleneck. Spanish characters (ñ, á, é, ü) are already representable. The architecture is language-agnostic. The **only missing ingredient is Spanish training data**.

| Component | Status | Notes |
|-----------|--------|-------|
| Model architecture | ✅ Language-agnostic | Byte-level encoding handles UTF-8 natively |
| Nonverbal tags | ✅ Language-agnostic | `(laughs)`, `(suspira)`, `(tose)` — model learns from data |
| Spanish training data | ❌ **Missing** | This is the core challenge |
| Spanish evaluation | ❌ **Missing** | Need speaker diarization + ASR for quality metrics |
| DAC audio codec | ✅ Language-agnostic | Descript Audio Codec is universal |

---

## 📋 Project Plan

### Phase 1: Data Collection & Preparation (Weeks 1-2)

- **Spanish podcast corpus** — scrape public Spanish-language podcasts with transcripts
- **Spanish YouTube** — conversational content with auto-generated captions
- **Spanish radio/TV interviews** — existing datasets (Common Voice, VoxPopuli, etc.)
- **Synthetic conversations** — generate Spanish dialog scripts with LLM (GPT-4, Claude)
- **Format**: Stereo WAV pairs (S1/S2) with transcript aligned per segment
- **Target**: 500-2000 hours of Spanish dialogue audio

Key data sources:
- [Mozilla Common Voice Spanish](https://commonvoice.mozilla.org/es)
- [VoxPopuli](https://github.com/facebookresearch/voxpopuli)
- [LibriSpeech Español](https://www.openslr.org/61/)
- [Google FLEURS Spanish](https://huggingface.co/datasets/google/fleurs)
- [TEDx Spanish Corpus](https://www.openslr.org/107/)
- [MagicHub SpCSC](https://www.magichub.com/datasets/) (used in moshi-spanish-finetuned)

### Phase 2: Data Preparation (Weeks 2-3)

Turn raw podcast audio into speaker-tagged training chunks:
- Transcribe + diarize with **WhisperX** (word-level timestamps + speakers)
- Group words into 3–17 s chunks with `[S1]`/`[S2]` tags → `chunk_manifest.jsonl`
- Cut + resample audio (16k → 44.1 kHz mono) on Modal
- Package into a HuggingFace `datasets.Dataset` with an `Audio` feature

### Phase 3: Fine-Tuning (Weeks 3-4)

- Fine-tune `nari-labs/Dia-1.6B` on the Spanish dataset
- Full fine-tune in bf16 with an 8-bit AdamW optimizer
- Channel-weighted cross-entropy over the 9 DAC codebooks
- Train on Modal (A100 80GB)
- Spanish-specific nonverbal tags: `(risas)`, `(suspiros)`, `(tos)`, `(gemidos)`

### Phase 4: Evaluation & Release

- ASR-based evaluation: generate audio → transcribe → compare with input transcript
- Speaker consistency metrics
- Naturalness MOS (Mean Opinion Score) with native Spanish speakers
- Release weights to HuggingFace

---

## 📁 Project Structure

```
dia-spanish/
├── dia/                    # Original Dia model code (upstream: nari-labs/dia)
│   ├── config.py
│   ├── layers.py
│   ├── model.py
│   ├── audio.py
│   └── state.py
├── data/
│   ├── raw/                # Raw Spanish audio + transcripts
│   ├── processed/          # Aligned, formatted training data
│   └── scripts/            # Data processing scripts
├── scripts/
│   ├── collect_spanish_podcasts.py
│   ├── collect_spanish_youtube.py
│   ├── generate_spanish_dialogues.py   # LLM-powered synthetic dialogues
│   ├── prepare_training_data.py
│   └── evaluate_es.py                 # Spanish evaluation pipeline
├── config/
│   └── spanish_finetune.yaml           # Training config
├── docs/
│   ├── DATA_SOURCES.md                 # Spanish audio data sources
│   └── Dia-Spanish-Report.pdf          # Architecture, pipeline & data report
├── app.py                  # Gradio UI (from upstream)
├── cli.py                  # CLI interface (from upstream)
├── example/                # Example scripts (from upstream)
└── pyproject.toml
```

---

## 🚀 Quick Start

### Setup

```bash
cd dia-spanish
python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
```

### Generate Spanish dialogue (synthetic)

```bash
python scripts/generate_spanish_dialogues.py --num-samples 100 --output data/raw/synthetic/spanish_dialogues.jsonl
```

### Prepare training data

```bash
python scripts/prepare_training_data.py --input data/raw/synthetic/spanish_dialogues.jsonl --output data/processed/spanish_dialogues_train.jsonl
```

### Fine-tune on Modal

```bash
# Smoke test (3,000 samples, 1 epoch, ~1 hour on L40S)
modal run train_dia_es.py --max-samples 3000 --epochs 1

# Full run (43,219 samples, 2 epochs, ~24h on A100-80GB)
modal run train_dia_es.py --max-samples 0 --epochs 2 --gpu a100-80gb
```

---

## 🔗 Related Projects

- [moshi-spanish-finetuned](https://github.com/D8-laboratory/moshi-spanish-finetuned) — Our Spanish PersonaPlex/Moshi fine-tune for Certeza STS
- [nari-labs/dia](https://github.com/nari-labs/dia) — Original Dia model (upstream)

---

## 📜 License

Upstream Dia code: Apache 2.0 (© NARI Labs)
Our additions: Apache 2.0 (© D8 Labs)
