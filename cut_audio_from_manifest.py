"""
Cut audio chunks for Dia fine-tuning from the diarized transcript manifest.

Reads chunk_manifest.jsonl (text + [start,end] per chunk), slices the source
WAVs on the Modal volume, resamples 16kHz → 44.1kHz mono, and writes individual
chunk WAVs + a metadata JSONL for downstream HF dataset packaging.

Usage:
  modal run cut_audio_from_manifest.py --limit 3          # smoke test
  modal run cut_audio_from_manifest.py                     # full run (80 episodes)
"""

from __future__ import annotations

import json
import math
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
CHUNKS_DIR = DATA_DIR / "chunks"           # output: individual chunk WAVs
MANIFEST_VOL_PATH = PODCAST_DIR / "manifests" / "chunk_manifest.jsonl"
META_VOL_PATH = CHUNKS_DIR / "_metadata.jsonl"

SOURCE_SR = 16000   # source WAVs are 16kHz mono (WhisperX downsampled)
TARGET_SR = 44100   # Dia expects 44.1kHz

app = modal.App("dia-spanish-chunker")

volume = modal.Volume.from_name("dia-spanish-vol")

# Image: soundfile for random-access WAV reads, torchaudio for resampling
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "soundfile>=0.12",
        "torch>=2.1",
        "torchaudio>=2.1",
        "numpy>=1.24",
    )
)


# ── Chunking function ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={str(VOL_MOUNT): volume},
    cpu=2,
    memory=4096,
    timeout=30 * MINUTES,
    max_containers=10,
)
def cut_episode_chunks(episode_id: str) -> dict:
    """Cut all chunks belonging to one episode from its source WAV.

    Reads the manifest from the volume, filters for this episode's chunks,
    then for each: seek → read → resample → save as 44.1kHz mono WAV.
    Resume-safe: skips chunks whose output WAV already exists.
    """
    import soundfile as sf
    import torch
    import torchaudio
    import numpy as np

    volume.reload()  # see latest manifest + any committed chunks

    # ── Load manifest, filter for this episode ──
    if not MANIFEST_VOL_PATH.exists():
        return {"episode_id": episode_id, "error": "manifest not found on volume"}
    chunks = []
    with MANIFEST_VOL_PATH.open() as f:
        for line in f:
            c = json.loads(line)
            if c["episode_id"] == episode_id:
                chunks.append(c)
    if not chunks:
        return {"episode_id": episode_id, "error": "no chunks for episode"}

    # ── Find source audio ──
    audio_path = AUDIO_DIR / f"{episode_id}.wav"
    if not audio_path.exists():
        return {"episode_id": episode_id, "error": f"audio not found: {audio_path.name}"}

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-build resampler (reused across chunks for same episode)
    resampler = torchaudio.transforms.Resample(
        orig_freq=SOURCE_SR, new_freq=TARGET_SR,
        dtype=torch.float32,
    )

    # Open SoundFile once for random access
    written = 0
    skipped = 0
    errors = 0

    with sf.SoundFile(str(audio_path), mode="r") as sf_src:
        src_sr = sf_src.samplerate
        src_ch = sf_src.channels
        if src_sr != SOURCE_SR:
            # Adjust resampler if source differs from expected
            resampler = torchaudio.transforms.Resample(
                orig_freq=src_sr, new_freq=TARGET_SR, dtype=torch.float32,
            )

        for c in chunks:
            out_path = CHUNKS_DIR / f"{c['chunk_id']}.wav"

            # Resume-safety: skip if already exists and non-empty
            if out_path.exists() and out_path.stat().st_size > 44:
                skipped += 1
                continue

            try:
                start_frame = int(c["start"] * src_sr)
                end_frame = int(c["end"] * src_sr)
                n_frames = max(1, end_frame - start_frame)

                sf_src.seek(start_frame)
                audio = sf_src.read(n_frames)  # shape (n_frames,) mono or (n_frames, n_ch)

                # Ensure mono
                if audio.ndim > 1:
                    audio = audio[:, 0]

                # Resample 16k → 44.1k
                t = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
                t_resampled = resampler(t).squeeze(0).numpy()     # (T',)

                # Peak-normalize (VN README mentions peak-normalization)
                peak = np.abs(t_resampled).max()
                if peak > 0:
                    t_resampled = t_resampled / peak * 0.99

                # Write as 44.1kHz mono PCM_16
                sf.write(str(out_path), t_resampled, TARGET_SR, subtype="PCM_16")
                written += 1
            except Exception as e:
                print(f"  ✗ {c['chunk_id']}: {e}")
                errors += 1

    volume.commit()

    return {
        "episode_id": episode_id,
        "total_chunks": len(chunks),
        "written": written,
        "skipped": skipped,
        "errors": errors,
    }


# ── Metadata builder (run once after all cutting is done) ─────────────────────

@app.function(
    image=image,
    volumes={str(VOL_MOUNT): volume},
    timeout=10 * MINUTES,
)
def build_metadata() -> dict:
    """Scan existing chunk WAVs, cross-reference manifest, write clean _metadata.jsonl."""
    volume.reload()
    if not MANIFEST_VOL_PATH.exists():
        return {"error": "manifest not found"}

    all_chunks = []
    with MANIFEST_VOL_PATH.open() as f:
        for line in f:
            all_chunks.append(json.loads(line))

    found = 0
    missing = 0
    meta = []
    for c in all_chunks:
        wav_path = CHUNKS_DIR / f"{c['chunk_id']}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 44:
            meta.append({
                "chunk_id": c["chunk_id"],
                "episode_id": c["episode_id"],
                "text": c["text"],
                "language": "es",
                "duration": round(c["duration"], 3),
                "speakers": c.get("speakers", []),
                "num_turns": c.get("num_turns", 1),
                "audio_file": f"{c['chunk_id']}.wav",
            })
            found += 1
        else:
            missing += 1

    with META_VOL_PATH.open("w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    volume.commit()
    return {"found": found, "missing": missing, "total_manifest": len(all_chunks)}


# ── Orchestrator ──────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={str(VOL_MOUNT): volume},
    timeout=10 * MINUTES,
)
def get_episode_ids_from_manifest() -> list[str]:
    """Read the manifest from the volume and return unique episode IDs."""
    volume.reload()
    if not MANIFEST_VOL_PATH.exists():
        return []
    ids = set()
    with MANIFEST_VOL_PATH.open() as f:
        for line in f:
            c = json.loads(line)
            ids.add(c["episode_id"])
    return sorted(ids)


@app.local_entrypoint()
def main(limit: int = 0):
    """Run the chunker. Set --limit N to process only first N episodes."""
    # Upload manifest to volume first
    local_manifest = Path("results/chunk_manifest.jsonl")
    if not local_manifest.exists():
        raise FileNotFoundError(f"{local_manifest} not found — run chunk_for_training.py first")

    print("📤 Uploading manifest to volume...")
    manifest_content = local_manifest.read_text(encoding="utf-8")
    _upload_manifest.remote(manifest_content)

    episode_ids = get_episode_ids_from_manifest.remote()
    if limit:
        episode_ids = episode_ids[:limit]

    print(f"🎙️  Processing {len(episode_ids)} episodes...")
    print(f"   Source: {SOURCE_SR}Hz → Target: {TARGET_SR}Hz mono")
    print()

    results = list(cut_episode_chunks.map(episode_ids))

    total_written = sum(r.get("written", 0) for r in results)
    total_skipped = sum(r.get("skipped", 0) for r in results)
    total_errors = sum(r.get("errors", 0) for r in results)
    errors_list = [r for r in results if r.get("error")]

    print(f"\n{'='*60}")
    print(f"✅ Cutting done. {total_written} written, {total_skipped} skipped, {total_errors} errors")
    if errors_list:
        print(f"⚠️  {len(errors_list)} episodes had errors:")
        for r in errors_list:
            print(f"   {r['episode_id']}: {r['error']}")
    print(f"{'='*60}")

    # Build clean metadata index
    print("\n📋 Building metadata index...")
    meta_result = build_metadata.remote()
    print(f"   Metadata: {meta_result['found']} found, {meta_result['missing']} missing, "
          f"{meta_result['total_manifest']} total in manifest")


@app.function(
    image=image,
    volumes={str(VOL_MOUNT): volume},
    timeout=5 * MINUTES,
)
def _upload_manifest(content: str = ""):
    """Write the manifest content to the volume."""
    volume.reload()
    MANIFEST_VOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Reset metadata too (start fresh)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_VOL_PATH.write_text(content, encoding="utf-8")
    n_lines = content.count("\n") if content else 0
    volume.commit()
    print(f"   Manifest uploaded: {len(content)} bytes, {n_lines} chunks")


if __name__ == "__main__":
    pass  # Use `modal run` entrypoint
