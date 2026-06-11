# Spanish Audio Data Sources for Dia Training

## Free / Open Datasets

### 1. Mozilla Common Voice Spanish
- **URL**: https://commonvoice.mozilla.org/es
- **Size**: ~2,000+ hours validated Spanish speech
- **License**: CC0
- **Notes**: Single-speaker read speech, not dialogue.

### 2. VoxPopuli
- **URL**: https://github.com/facebookresearch/voxpopuli
- **Size**: ~1,500 hours Spanish
- **License**: CC0
- **Notes**: Formal EU parliament speech.

### 3. Google FLEURS (Spanish)
- **URL**: https://huggingface.co/datasets/google/fleurs
- **Size**: ~12 hours Spanish
- **License**: CC-BY-4.0
- **Notes**: Small but high quality, multiple Spanish variants.

### 4. LibriSpeech Español
- **URL**: https://www.openslr.org/61/
- **Size**: ~240 hours
- **License**: CC-BY-4.0

### 5. MagicHub SpCSC (Spanish Conversational Speech Corpus)
- **URL**: https://www.magichub.com/datasets/
- **Size**: ~200+ hours conversational
- **Notes**: **Already used in moshi-spanish-finetuned.** Best fit for Dia.

### 6. TEDx Spanish Corpus
- **URL**: https://www.openslr.org/107/
- **Size**: ~180 hours
- **License**: CC-BY-NC-SA

## Podcast / Radio Sources (Requires scraping)

### 7. Spanish Podcasts
- **Platforms**: iVoox, Spotify, YouTube
- **Tool**: `yt-dlp` + Whisper for transcription

### 8. YouTube Conversational Content
- **Search**: "entrevista español", "podcast español"
- **Tool**: `yt-dlp` for audio, Whisper for transcription

## Synthetic Data Generation

### 9. LLM-Generated Spanish Dialogues
- **Script**: `scripts/generate_spanish_dialogues.py`
- **Format**: `[S1] text [S2] text (risas)` with nonverbal tags
- **TTS for audio**: Cartesia, ElevenLabs for audio generation from synthetic transcripts

## Nonverbal Tags (Spanish)

| English | Spanish |
|---------|---------|
| (laughs) | (risas) |
| (sighs) | (suspira) |
| (coughs) | (tose) |
| (gasps) | (jadea) |
| (groans) | (gime) |
| (screams) | (grita) |
| (singing) | (cantando) |
| (humming) | (tarareando) |
| (whistles) | (silba) |
| (sneezes) | (estornuda) |
| (mumbles) | (murmura) |
