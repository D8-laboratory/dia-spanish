"""WhisperX transcription + speaker diarization for Spanish podcasts.

Processes all RSS-sourced WAV files on the Modal volume:
  - Whisper large-v3 transcription (Spanish)
  - Word-level forced alignment
  - Speaker diarization via pyannote (requires HF token)

Output: JSON files with speaker-labeled segments + word timestamps.

Usage:
  modal run transcribe_whisperx.py
  modal run transcribe_whisperx.py --max-episodes 5   # test run
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import modal

# ── Config ────────────────────────────────────────────────────────────────────

MINUTES = 60
HOURS = 60 * MINUTES

VOL_MOUNT = Path("/vol")
DATA_DIR = VOL_MOUNT / "data"
PODCAST_DIR = DATA_DIR / "podcasts_rss"
AUDIO_DIR = PODCAST_DIR / "audio"
TRANSCRIPTS_DIR = PODCAST_DIR / "transcripts"
MANIFESTS_DIR = PODCAST_DIR / "manifests"
MANIFEST_FILE = MANIFESTS_DIR / "podcast_episodes.jsonl"

app = modal.App("dia-spanish-whisperx")

volume = modal.Volume.from_name("dia-spanish-vol")

# ── Image ─────────────────────────────────────────────────────────────────────

whisperx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "build-essential")
    .pip_install(
        "whisperx>=3.1.1",
        "torch>=2.1",
        "torchaudio>=2.1",
        "faster-whisper>=1.0",
        "pyannote.audio>=3.1",
    )
    # Model downloads happen at runtime where HF secret is available
    # (pyannote models are gated — need HF token from Modal secret)
)


# ── Transcription Function ───────────────────────────────────────────────────

@app.function(
    image=whisperx_image,
    volumes={str(VOL_MOUNT): volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    gpu="A10G",
    cpu=4,
    memory=8192,
    timeout=2 * HOURS,
    max_containers=10,
)
def transcribe_episode(episode_id: str) -> Optional[dict]:
    """Transcribe a single episode with WhisperX: ASR + alignment + diarization."""
    import whisperx
    import torch

    audio_path = AUDIO_DIR / f"{episode_id}.wav"
    transcript_path = TRANSCRIPTS_DIR / f"{episode_id}.json"

    if not audio_path.exists():
        print(f"⚠️  Audio not found: {audio_path}")
        return None

    # Skip if already done
    if transcript_path.exists():
        print(f"⏭️  Already transcribed: {episode_id}")
        with open(transcript_path) as f:
            return json.load(f)

    print(f"🎙️  Transcribing: {episode_id}")

    try:
        return _transcribe_inner(episode_id, audio_path, transcript_path)
    except Exception as e:
        import torch
        # Clear CUDA cache so the container can be reused for the next episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ❌ FAILED ({type(e).__name__}): {e}")
        return None


def _transcribe_inner(episode_id: str, audio_path, transcript_path) -> dict:
    """Actual transcription logic. Raises on failure so caller can handle + clear VRAM."""
    import whisperx
    import torch
    import os
    import inspect

    # 1. Load audio
    audio = whisperx.load_audio(str(audio_path))

    # 2. Transcribe with Whisper large-v3 (batch_size=8 for VRAM headroom)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(
        "large-v3",
        device,
        language="es",
        compute_type="float16" if device == "cuda" else "int8",
        vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
    )

    result = model.transcribe(audio, batch_size=8, language="es")
    # Free ASR model VRAM before loading alignment + diarization models
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Align (word-level timestamps)
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code="es", device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"  ⚠️  Alignment failed: {e}, using segment-level timestamps")

    # 4. Speaker diarization
    # NOTE: WhisperX renamed `use_auth_token` → `token` around v3.1.
    # Probe the signature and pass whichever the installed version accepts.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from whisperx.diarize import DiarizationPipeline
        sig_params = set(inspect.signature(DiarizationPipeline.__init__).parameters)
        kw = {"device": device}
        if "token" in sig_params:
            kw["token"] = hf_token
        elif "use_auth_token" in sig_params:
            kw["use_auth_token"] = hf_token
        elif "auth_token" in sig_params:
            kw["auth_token"] = hf_token
        diarize_model = DiarizationPipeline(**kw)
        diarize_segments = diarize_model(
            str(audio_path),
            min_speakers=2,
            max_speakers=4,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        print(f"  ⚠️  Diarization failed: {e}, using unlabeled speakers")

    # 5. Save transcript
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    transcript_data = {
        "episode_id": episode_id,
        "language": "es",
        "num_segments": len(result.get("segments", [])),
        "segments": [],
    }

    for seg in result.get("segments", []):
        entry = {
            "start": float(seg.get("start", 0) or 0),
            "end": float(seg.get("end", 0) or 0),
            "text": str(seg.get("text", "")).strip(),
            "speaker": str(seg.get("speaker", "SPEAKER_UNKNOWN")),
        }
        # Include word-level timestamps if available
        if "words" in seg:
            entry["words"] = [
                {
                    "word": str(w.get("word", "")),
                    "start": float(w.get("start", 0) or 0),
                    "end": float(w.get("end", 0) or 0),
                    "score": float(w.get("score", 0) or 0),
                }
                for w in seg["words"]
            ]
        transcript_data["segments"].append(entry)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    volume.commit()

    duration_min = sum(
        s["end"] - s["start"] for s in transcript_data["segments"]
    ) / 60
    speakers = set(s["speaker"] for s in transcript_data["segments"])
    print(
        f"  ✅ {len(transcript_data['segments'])} segments, "
        f"{duration_min:.0f} min, {len(speakers)} speakers"
    )

    # Return only a small summary — the full transcript is already on the volume.
    # Returning large dicts via .map() triggers Modal blob transport errors.
    return {
        "episode_id": episode_id,
        "num_segments": len(transcript_data["segments"]),
        "duration_min": round(duration_min),
        "num_speakers": len(speakers),
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={str(VOL_MOUNT): volume},
    timeout=5 * MINUTES,
)
def get_stats() -> dict:
    """Get transcription progress and list of episode IDs to process."""
    if not AUDIO_DIR.exists():
        return {"error": "No audio directory", "todo_ids": []}

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    transcripts = list(TRANSCRIPTS_DIR.glob("*.json")) if TRANSCRIPTS_DIR.exists() else []

    done_ids = {t.stem for t in transcripts}
    remaining = [w.stem for w in wav_files if w.stem not in done_ids]

    return {
        "total_episodes": len(wav_files),
        "transcribed": len(transcripts),
        "remaining": len(remaining),
        "todo_ids": remaining,  # full list for entrypoint to use
    }


# ── Local Entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(max_episodes: int = 0):
    """Transcribe all RSS podcast episodes with WhisperX."""
    import time

    print("=" * 60)
    print("🎙️  Dia-Spanish WhisperX Transcription Pipeline")
    print("=" * 60)

    # Get stats + episode IDs from remote (volume-mounted context)
    stats = get_stats.remote()
    print(f"\n📊 Progress: {stats['transcribed']}/{stats['total_episodes']} transcribed")
    print(f"   Remaining: {stats['remaining']}")

    if stats["remaining"] == 0:
        print("\n✅ All episodes already transcribed!")
        return

    todo = stats["todo_ids"]

    if max_episodes > 0:
        todo = todo[:max_episodes]
        print(f"\n🔢 Limited to {max_episodes} episodes")

    print(f"\n🚀 Transcribing {len(todo)} episodes on A10G GPUs (max 10 concurrent)...")
    print(f"   (resume-safe: skips the {stats['transcribed']} already done)\n")

    start = time.time()
    successful = 0
    failed = []
    for i, result in enumerate(transcribe_episode.map(todo, order_outputs=True), 1):
        elapsed = time.time() - start
        if result is not None:
            n_seg = result.get("num_segments", 0)
            successful += 1
            print(f"  [{i}/{len(todo)}] ✅ {result['episode_id']} ({n_seg} segs) "
                  f"— {elapsed/60:.1f} min elapsed")
        else:
            failed.append(todo[i - 1])
            print(f"  [{i}/{len(todo)}] ❌ {todo[i-1]} failed — {elapsed/60:.1f} min elapsed")

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"✅ Transcribed: {successful}/{len(todo)}")
    if failed:
        print(f"❌ Failed: {len(failed)} — {failed}")
    print(f"⏱️  Total time: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # Final stats
    final = get_stats.remote()
    print(f"\n📊 Final: {final['transcribed']}/{final['total_episodes']} episodes transcribed")
