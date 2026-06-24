"""
Package cut audio chunks into a HuggingFace datasets.Dataset.

Reads the chunk WAVs + _metadata.jsonl from the Modal volume, builds a proper
HF Dataset with Audio feature (matching cosrigel/vn_tts_medium_clean schema),
and either saves to disk on the volume or pushes to the HF Hub.

Usage:
  modal run build_hf_dataset.py --mode disk          # save arrow to volume
  modal run build_hf_dataset.py --mode hub --repo USERNAME/spanish-podcasts-dia
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MINUTES = 60

VOL_MOUNT = Path("/vol")
DATA_DIR = VOL_MOUNT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
META_PATH = CHUNKS_DIR / "_metadata.jsonl"
HF_DATASET_DIR = DATA_DIR / "hf_dataset"

app = modal.App("dia-spanish-hf-builder")

volume = modal.Volume.from_name("dia-spanish-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "datasets>=2.16,<3.0",
        "soundfile>=0.12",
        "torch>=2.1",
        "numpy>=1.24",
    )
)


@app.function(
    image=image,
    volumes={str(VOL_MOUNT): volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    cpu=4,
    memory=16384,
    timeout=30 * MINUTES,
)
def build_dataset(mode: str = "disk", repo_name: str = "") -> dict:
    """Build HF dataset from chunk WAVs + metadata on the volume.

    Stores audio paths as strings (NOT Audio feature) — avoids slow 43k-file
    encoding. The training dataset loader reads WAVs from these paths at runtime.
    For push_to_hub, audio must be embedded separately (TODO).
    """
    volume.reload()

    import numpy as np
    from datasets import Dataset, Features, Value, Sequence

    # ── Load metadata ──
    if not META_PATH.exists():
        return {"error": f"metadata not found: {META_PATH}"}

    meta = []
    with META_PATH.open() as f:
        for line in f:
            meta.append(json.loads(line))

    print(f"Loaded {len(meta):,} metadata entries")

    # ── Build lists ──
    audio_paths = []
    texts = []
    languages = []
    durations = []
    speakers_list = []
    num_turns_list = []
    episode_ids = []
    chunk_ids = []

    missing = 0
    for m in meta:
        wav_path = str(CHUNKS_DIR / m["audio_file"])
        p = Path(wav_path)
        if not p.exists() or p.stat().st_size <= 44:
            missing += 1
            continue
        audio_paths.append(wav_path)
        texts.append(m["text"])
        languages.append(m["language"])
        durations.append(m["duration"])
        speakers_list.append(m.get("speakers", []))
        num_turns_list.append(m.get("num_turns", 1))
        episode_ids.append(m["episode_id"])
        chunk_ids.append(m["chunk_id"])

    print(f"Valid chunks: {len(audio_paths):,} (skipped {missing} missing/empty)")
    if not audio_paths:
        return {"error": "no valid chunks found"}

    # ── Define schema (audio_path as string, NOT Audio feature) ──
    features = Features({
        "audio_path": Value("string"),
        "text": Value("string"),
        "language": Value("string"),
        "duration": Value("float64"),
        "speakers": Sequence(Value("string")),
        "num_turns": Value("int64"),
        "episode_id": Value("string"),
        "chunk_id": Value("string"),
    })

    ds = Dataset.from_dict({
        "audio_path": audio_paths,
        "text": texts,
        "language": languages,
        "duration": durations,
        "speakers": speakers_list,
        "num_turns": num_turns_list,
        "episode_id": episode_ids,
        "chunk_id": chunk_ids,
    }, features=features)

    print(f"Dataset built: {len(ds):,} samples")
    print(f"Features: {ds.features}")

    # ── Stats ──
    total_hours = sum(durations) / 3600
    multi_turn = sum(1 for n in num_turns_list if n >= 2)
    stats = {
        "num_samples": len(ds),
        "total_hours": round(total_hours, 1),
        "multi_turn": multi_turn,
        "multi_turn_pct": round(100 * multi_turn / len(ds), 1),
        "avg_duration": round(sum(durations) / len(durations), 2),
    }
    print(f"Stats: {stats}")

    # ── Output ──
    if mode == "disk":
        print(f"\n💾 Saving to disk: {HF_DATASET_DIR}")
        HF_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(HF_DATASET_DIR))
        volume.commit()
        # Check size
        import subprocess
        result = subprocess.run(
            ["du", "-sh", str(HF_DATASET_DIR)], capture_output=True, text=True
        )
        size = result.stdout.strip()
        return {**stats, "saved_to": str(HF_DATASET_DIR), "disk_size": size}

    elif mode == "hub":
        if not repo_name:
            return {"error": "--repo required for hub mode"}
        print(f"\n📤 Pushing to HF Hub: {repo_name}")
        ds.push_to_hub(repo_name, private=True)
        return {**stats, "pushed_to": repo_name}

    else:
        return {"error": f"unknown mode: {mode}"}


@app.local_entrypoint()
def main(mode: str = "disk", repo: str = ""):
    result = build_dataset.remote(mode=mode, repo_name=repo)
    print(f"\n{'='*60}")
    print("RESULT:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")
