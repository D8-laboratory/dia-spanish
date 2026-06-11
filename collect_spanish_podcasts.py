"""Download 100+ hours of Spanish podcasts on Modal for Dia training data.

Uses yt-dlp to download audio from YouTube Spanish podcast channels,
then transcribes with Whisper for aligned text+audio pairs.

Usage:
  # Dry run (list episodes found, no download)
  modal run collect_spanish_podcasts.py --dry-run

  # Download all (100 episodes, ~100 hours)
  modal run collect_spanish_podcasts.py --num-episodes 100

  # Download and transcribe with Whisper
  modal run collect_spanish_podcasts.py --num-episodes 100 --transcribe

  # Download to local machine instead of Modal volume
  modal run collect_spanish_podcasts.py --num-episodes 10 --download-local 5
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

# ── App & Volume ──────────────────────────────────────────────────────────────

APP_NAME = "dia-spanish-collector"

MINUTES = 60
HOURS = 60 * MINUTES

VOL_MOUNT = Path("/vol")
DATA_DIR = VOL_MOUNT / "data"
PODCAST_DIR = DATA_DIR / "podcasts"
AUDIO_DIR = PODCAST_DIR / "audio"
TRANSCRIPTS_DIR = PODCAST_DIR / "transcripts"
MANIFESTS_DIR = PODCAST_DIR / "manifests"
MANIFEST_FILE = MANIFESTS_DIR / "podcast_episodes.jsonl"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name("dia-spanish-vol", create_if_missing=True)

# ── Image ─────────────────────────────────────────────────────────────────────

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "curl", "unzip")
    .run_commands(
        # Install deno (yt-dlp needs JS runtime for YouTube)
        "curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh",
    )
    .pip_install(
        "yt-dlp>=2024.12.0",
        "tqdm",
    )
)

transcribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "yt-dlp>=2024.12.0",
        "openai-whisper>=20240930",
        "tqdm",
    )
)

# ── Spanish Podcast Sources ──────────────────────────────────────────────────
# Each source is a YouTube channel URL with episode count target.
# We prioritize: conversational, interview, and dialogue formats.
# Duration filter: 20min - 4hr (sweet spot for 1-hour podcasts)

PODCAST_SOURCES = [
    # ── Channel-based (known working URLs) ──
    {
        "name": "The Wild Project",
        "url": "https://www.youtube.com/@TheWildProject",
        "target": 20,
        "region": "españa",
        "format": "interview",
        "notes": "Long-form interviews, 2-4 hours. Excellent conversational Spanish (Spain).",
    },
    {
        "name": "Entiende tu Mente",
        "url": "https://www.youtube.com/@entiendetumente",
        "target": 15,
        "region": "neutral",
        "format": "conversation",
        "notes": "Psychology conversations. Clean, educational Spanish.",
    },
    # ── Search-based discovery (most reliable) ──
    # Each query returns diverse results from across the Spanish-speaking world
    {
        "name": "podcast entrevista español",
        "query": "podcast entrevista español larga duración",
        "target": 15,
        "region": "mixed",
        "format": "interview",
    },
    {
        "name": "podcast conversación español",
        "query": "podcast conversación español entrevista completa",
        "target": 15,
        "region": "mixed",
        "format": "conversation",
    },
    {
        "name": "podcast colombiano entrevista",
        "query": "podcast colombiano entrevista larga",
        "target": 10,
        "region": "colombia",
        "format": "interview",
    },
    {
        "name": "podcast mexicano charla",
        "query": "podcast mexicano entrevista completa",
        "target": 10,
        "region": "mexico",
        "format": "conversation",
    },
    {
        "name": "podcast argentino",
        "query": "podcast argentino entrevista larga duración",
        "target": 10,
        "region": "argentina",
        "format": "interview",
    },
    {
        "name": "entrevista podcast español",
        "query": "entrevista podcast programa completo español",
        "target": 15,
        "region": "mixed",
        "format": "interview",
    },
    {
        "name": "Radio Ambulante",
        "query": "radio ambulante latinoamericano historia completa",
        "target": 10,
        "region": "latam",
        "format": "narrative",
    },
    {
        "name": "podstickers español",
        "query": "podcast español humor conversación completa",
        "target": 10,
        "region": "mixed",
        "format": "conversation",
    },
]


def _ensure_dirs():
    """Create directory structure on volume."""
    for d in [AUDIO_DIR, TRANSCRIPTS_DIR, MANIFESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _download_audio(
    video_url: str,
    output_dir: Path,
    episode_id: str,
) -> Optional[dict]:
    """Download audio from a YouTube video as 16kHz mono WAV.
    
    Returns metadata dict or None on failure."""
    out_path = output_dir / f"{episode_id}.wav"

    if out_path.exists():
        return {"path": str(out_path), "skipped": True}

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",  # extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(output_dir / f"{episode_id}.%(ext)s"),
        "--no-write-comments",
        "--no-write-description",
        "--write-info-json",
        "--write-auto-sub",
        "--sub-lang", "es",
        "--sub-format", "vtt",
        "--restrict-filenames",
        "--retries", "3",
        "--fragment-retries", "3",
        video_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        print(f"  ⚠ Download failed: {result.stderr[:200]}")
        return None

    # Read info json
    info_path = output_dir / f"{episode_id}.info.json"
    info = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

    # Check file exists (yt-dlp may rename)
    actual_wav = output_dir / f"{episode_id}.wav"
    if not actual_wav.exists():
        # Find the actual file
        candidates = list(output_dir.glob(f"{episode_id}*.wav"))
        if not candidates:
            return None
        actual_wav = candidates[0]

    return {
        "path": str(actual_wav),
        "title": info.get("title", ""),
        "duration": info.get("duration", 0),
        "channel": info.get("channel", ""),
        "upload_date": info.get("upload_date", ""),
        "description": info.get("description", "")[:500],
        "skipped": False,
    }


def _list_channel_episodes(
    source: dict,
    max_episodes: int,
    min_duration: int = 1200,  # 20 minutes
    max_duration: int = 14400,  # 4 hours
) -> list[dict]:
    """List episodes from a YouTube channel or search query, filtered by duration."""
    # Determine if this is a channel URL or search query
    if "query" in source:
        target = f"ytsearch{max_episodes * 3}:{source['query']}"
    else:
        target = source["url"]

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(max_episodes * 3),
        "--no-warnings",
        target,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  ⚠ Failed to list: {result.stderr[:200]}")
        return []

    episodes = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Skip entries without video IDs (e.g., channel metadata)
        if not entry.get("id") or entry.get("_type") == "playlist":
            continue

        duration = entry.get("duration", 0) or 0
        if min_duration <= duration <= max_duration:
            video_id = entry.get("id", "")
            episodes.append({
                "id": video_id,
                "title": entry.get("title", ""),
                "duration": duration,
                "channel": entry.get("channel", ""),
                "url": f"https://youtube.com/watch?v={video_id}",
            })

    # Sort by duration descending (prefer longer episodes)
    episodes.sort(key=lambda x: x["duration"], reverse=True)

    return episodes[:max_episodes]


# ── Modal Functions ───────────────────────────────────────────────────────────


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=12 * HOURS,
    cpu=2,
    memory=2048,
)
def discover_episodes(
    num_episodes: int = 100,
    dry_run: bool = False,
) -> list[dict]:
    """Discover Spanish podcast episodes across all sources."""
    all_episodes = []

    for source in PODCAST_SOURCES:
        label = source.get("query") or source.get("url", "")
        print(f"\n📡 Scanning: {source['name']} ({label})")
        episodes = _list_channel_episodes(source, source["target"])

        for ep in episodes:
            ep["source_name"] = source["name"]
            ep["source_region"] = source["region"]
            ep["source_format"] = source["format"]

        all_episodes.extend(episodes)
        total_hrs = sum(e["duration"] for e in episodes) / 3600
        print(f"   Found {len(episodes)} episodes ({total_hrs:.1f} hours)")

    # Deduplicate by video ID
    seen = set()
    unique = []
    for ep in all_episodes:
        if ep["id"] not in seen:
            seen.add(ep["id"])
            unique.append(ep)

    # Trim to requested count
    unique = unique[:num_episodes]

    total_hours = sum(e["duration"] for e in unique) / 3600
    print(f"\n{'='*60}")
    print(f"Total: {len(unique)} episodes, {total_hours:.1f} hours")
    print(f"{'='*60}")

    if dry_run:
        for i, ep in enumerate(unique, 1):
            hrs = ep["duration"] / 3600
            print(f"  {i:3d}. [{ep['source_region']:8s}] {ep['source_name']:25s} | {hrs:.1f}h | {ep['title'][:60]}")
        return unique

    return unique


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=2 * HOURS,
    cpu=2,
    memory=2048,
)
def download_episode(episode: dict) -> Optional[dict]:
    """Download a single podcast episode."""
    _ensure_dirs()

    source = episode["source_name"].replace(" ", "_").lower()
    ep_id = f"{source}_{episode['id']}"
    video_url = episode.get("url") or f"https://youtube.com/watch?v={episode['id']}"

    print(f"⬇️  Downloading: {episode['title'][:60]}...")
    meta = _download_audio(video_url, AUDIO_DIR, ep_id)

    if meta is None:
        print(f"  ❌ Failed: {episode['title'][:60]}")
        return None

    # Build manifest entry
    entry = {
        **episode,
        "episode_id": ep_id,
        "audio_path": meta["path"],
        "downloaded_at": datetime.utcnow().isoformat(),
        "skipped": meta.get("skipped", False),
    }

    if not meta.get("skipped"):
        print(f"  ✅ {meta.get('title', '')[:50]} ({entry.get('duration', 0)/60:.0f} min)")
    
    return entry


@app.function(
    image=transcribe_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=4 * HOURS,
    gpu="T4",
    cpu=4,
    memory=8192,
)
def transcribe_episode(episode: dict) -> Optional[dict]:
    """Transcribe a podcast episode using Whisper (Spanish)."""
    import whisper

    audio_path = Path(episode["audio_path"])
    if not audio_path.exists():
        print(f"  ⚠ Audio not found: {audio_path}")
        return None

    # Check if transcript already exists
    transcript_path = TRANSCRIPTS_DIR / f"{episode['episode_id']}.json"
    if transcript_path.exists():
        print(f"  ⏭️  Transcript exists: {episode['episode_id']}")
        with open(transcript_path) as f:
            return json.load(f)

    print(f"🎙️  Transcribing: {episode.get('title', '')[:50]}...")
    model = whisper.load_model("large-v3")

    result = model.transcribe(
        str(audio_path),
        language="es",
        verbose=False,
        word_timestamps=True,
    )

    # Save transcript
    transcript_data = {
        "episode_id": episode["episode_id"],
        "text": result["text"],
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
        "language": result.get("language", "es"),
    }

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    duration = episode.get("duration", 0)
    print(f"  ✅ {len(result['segments'])} segments ({duration/60:.0f} min)")

    return transcript_data


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=10 * MINUTES,
)
def save_manifest(episodes: list[dict]) -> str:
    """Save episode manifest to JSONL on volume."""
    _ensure_dirs()

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    volume.commit()
    total_hrs = sum(e.get("duration", 0) for e in episodes if e) / 3600
    successful = sum(1 for e in episodes if e is not None)
    print(f"\n📋 Manifest saved: {successful} episodes, {total_hrs:.1f} hours")
    print(f"   Path: {MANIFEST_FILE}")
    return str(MANIFEST_FILE)


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=30 * MINUTES,
)
def download_samples(num_samples: int = 3) -> list[str]:
    """Download a few sample files locally for inspection."""
    import base64

    if not MANIFEST_FILE.exists():
        print("No manifest found. Run download first.")
        return []

    samples = []
    with open(MANIFEST_FILE) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            ep = json.loads(line)
            audio_path = Path(ep["audio_path"])
            if audio_path.exists():
                size_mb = audio_path.stat().st_size / 1024 / 1024
                samples.append(f"{audio_path.name} ({size_mb:.1f} MB)")
            else:
                samples.append(f"{audio_path.name} (NOT FOUND)")

    return samples


# ── Local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    num_episodes: int = 100,
    dry_run: bool = False,
    transcribe: bool = False,
    download_local: int = 0,
):
    """Download Spanish podcasts for Dia training data.

    Args:
        num_episodes: Number of episodes to download (default 100)
        dry_run: List episodes without downloading
        transcribe: Also transcribe with Whisper (costs GPU time)
        download_local: Download N sample files locally for inspection
    """
    print("=" * 60)
    print("🎙️  Dia-Spanish Podcast Collector")
    print("=" * 60)

    # Phase 1: Discover episodes
    print(f"\n📡 Phase 1: Discovering episodes (target: {num_episodes})...")
    episodes = discover_episodes.remote(num_episodes=num_episodes, dry_run=dry_run)

    if dry_run:
        print(f"\nDry run complete. {len(episodes)} episodes found.")
        return

    if not episodes:
        print("No episodes found!")
        return

    # Phase 2: Download audio
    print(f"\n⬇️  Phase 2: Downloading {len(episodes)} episodes...")
    results = []
    for chunk_start in range(0, len(episodes), 10):
        chunk = episodes[chunk_start:chunk_start + 10]
        chunk_results = download_episode.map(chunk)
        results.extend(chunk_results)
        print(f"  Progress: {min(chunk_start + 10, len(episodes))}/{len(episodes)}")

    successful = [r for r in results if r is not None]
    failed = len(episodes) - len(successful)
    total_hrs = sum(r.get("duration", 0) for r in successful) / 3600

    print(f"\n✅ Downloaded: {len(successful)}/{len(episodes)} episodes ({total_hrs:.1f} hours)")
    if failed:
        print(f"❌ Failed: {failed}")

    # Save manifest
    save_manifest.remote(successful)

    # Phase 3: Transcribe (optional)
    if transcribe and successful:
        print(f"\n🎙️  Phase 3: Transcribing {len(successful)} episodes with Whisper...")
        trans_results = transcribe_episode.map(successful)
        trans_ok = sum(1 for r in trans_results if r is not None)
        print(f"✅ Transcribed: {trans_ok}/{len(successful)}")

    # Download samples locally
    if download_local > 0:
        print(f"\n📦 Downloading {download_local} samples locally...")
        samples = download_samples.remote(num_samples=download_local)
        for s in samples:
            print(f"  {s}")

    print("\n🎉 Done! Data stored on Modal volume: dia-spanish-vol")
    print("   Access with: modal volume get dia-spanish-vol data/podcasts .")
