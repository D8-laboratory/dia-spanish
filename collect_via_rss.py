"""Download 100+ hours of Spanish podcasts via RSS feeds on Modal.

RSS feeds give direct MP3 links — no YouTube bot protection issues.
Much faster and more reliable than yt-dlp on cloud IPs.

Usage:
  # Dry run (list episodes found, no download)
  modal run collect_via_rss.py --dry-run

  # Download 100 episodes (~100 hours)
  modal run collect_via_rss.py --num-episodes 100

  # Download and transcribe with Whisper
  modal run collect_via_rss.py --num-episodes 100 --transcribe
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import urlparse

import modal

# ── App & Volume ──────────────────────────────────────────────────────────────

APP_NAME = "dia-spanish-rss-collector"

MINUTES = 60
HOURS = 60 * MINUTES

VOL_MOUNT = Path("/vol")
DATA_DIR = VOL_MOUNT / "data"
PODCAST_DIR = DATA_DIR / "podcasts_rss"
AUDIO_DIR = PODCAST_DIR / "audio"
TRANSCRIPTS_DIR = PODCAST_DIR / "transcripts"
MANIFESTS_DIR = PODCAST_DIR / "manifests"
MANIFEST_FILE = MANIFESTS_DIR / "podcast_episodes.jsonl"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name("dia-spanish-vol", create_if_missing=True)

# ── Image ─────────────────────────────────────────────────────────────────────

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl")
    .pip_install("tqdm", "requests")
)

transcribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl")
    .pip_install(
        "openai-whisper>=20240930",
        "tqdm",
        "requests",
    )
)

# ── Spanish Podcast RSS Sources ──────────────────────────────────────────────
# Curated via Apple Podcasts API + manual verification.
# Prioritize: long-form interview/conversation (1-4 hrs), natural Spanish.

PODCAST_FEEDS = [
    # ── Spain ──
    {
        "name": "The Wild Project",
        "rss": "https://feeds.megaphone.fm/TWIP9771253765",
        "region": "españa",
        "format": "interview",
        "target": 20,
        "notes": "Jordi Wild — long-form interviews 2-4hrs, excellent Spain Spanish",
    },
    {
        "name": "Aladetres",
        "rss": "https://feeds.megaphone.fm/HOT9252650758",
        "region": "españa",
        "format": "interview",
        "target": 10,
        "notes": "Deep conversations and interviews",
    },
    # ── Mexico ──
    {
        "name": "Cracks Podcast",
        "rss": "https://rss.buzzsprout.com/1682368.rss",
        "region": "mexico",
        "format": "interview",
        "target": 10,
        "notes": "Oso Trava — long interviews with Mexican personalities",
    },
    {
        "name": "No Hay Tos",
        "rss": "https://feeds.simplecast.com/UrwSRKXH",
        "region": "mexico",
        "format": "conversation",
        "target": 10,
        "notes": "Real Mexican Spanish conversations",
    },
    # ── Colombia / LatAm ──
    {
        "name": "La Entrevista con Yordi Rosado",
        "rss": "https://feeds.simplecast.com/E9wKnFyb",
        "region": "mexico",
        "format": "interview",
        "target": 10,
        "notes": "Deep emotional interviews, Mexican Spanish",
    },
    # ── Pan-LatAm ──
    {
        "name": "Radio Ambulante",
        "rss": "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/b3c9b6e7-72ba-45c4-aff9-b1e7012d213b/092b66a8-4329-4183-bb12-b1e7012d216f/podcast.rss",
        "region": "latam",
        "format": "narrative",
        "target": 15,
        "notes": "Stories from across Latin America, high production quality",
    },
    {
        "name": "Entiende Tu Mente",
        "rss": "https://feeds.acast.com/public/shows/69779e4cf4b515342e3b249c",
        "region": "neutral",
        "format": "conversation",
        "target": 15,
        "notes": "Psychology conversations, clean educational Spanish",
    },
    # ── More interviews ──
    {
        "name": "La Entrevista Profunda",
        "rss": "https://anchor.fm/s/61823adc/podcast/rss",
        "region": "mixed",
        "format": "interview",
        "target": 10,
        "notes": "Long-form deep interviews in Spanish",
    },
]


# ── RSS Parsing ───────────────────────────────────────────────────────────────

def _parse_duration(text: str) -> int:
    """Parse duration from RSS (could be seconds, MM:SS, or HH:MM:SS)."""
    if not text:
        return 0
    text = text.strip()
    # Already seconds
    try:
        return int(text)
    except ValueError:
        pass
    # HH:MM:SS or MM:SS
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0


def _fetch_feed(feed_url: str) -> list[dict]:
    """Parse a podcast RSS feed and return episode list."""
    req = Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        tree = ET.parse(resp)

    root = tree.getroot()
    # Handle RSS namespaces
    ns = {
        "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
        "content": "http://purl.org/rss/1.0/modules/content/",
    }

    episodes = []
    for item in root.iter("item"):
        title = item.findtext("title", "")
        pub_date = item.findtext("pubDate", "")

        # Get audio URL from <enclosure>
        enc = item.find("enclosure")
        audio_url = enc.get("url", "") if enc is not None else ""
        audio_type = enc.get("type", "") if enc is not None else ""

        if not audio_url or not audio_type.startswith("audio/"):
            # Some feeds use <link> instead
            link = item.findtext("link", "")
            if link and any(ext in link.lower() for ext in [".mp3", ".m4a", ".wav", ".ogg"]):
                audio_url = link
            else:
                continue

        # Duration
        duration_el = item.find("itunes:duration", ns)
        duration = _parse_duration(duration_el.text if duration_el is not None else "")

        # If no itunes:duration, try to estimate from file size (rough)
        # Skip episodes we can't determine duration for
        if duration == 0:
            length = int(enc.get("length", 0)) if enc is not None else 0
            if length > 0:
                # Rough estimate: 1MB ≈ 1min at ~128kbps
                duration = int(length / (128 * 1000 / 8))

        description = item.findtext("description", "") or item.findtext("content:encoded", "", ns) or ""
        # Strip HTML
        description = re.sub(r"<[^>]+>", "", description)[:500]

        # Generate stable ID from audio URL
        ep_id = hashlib.md5(audio_url.encode()).hexdigest()[:12]

        episodes.append({
            "id": ep_id,
            "title": title,
            "audio_url": audio_url,
            "audio_type": audio_type,
            "duration": duration,
            "pub_date": pub_date,
            "description": description,
        })

    # Sort newest first
    return episodes


def _ensure_dirs():
    for d in [AUDIO_DIR, TRANSCRIPTS_DIR, MANIFESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _download_episode_audio(episode: dict, output_dir: Path) -> Optional[dict]:
    """Download podcast audio via curl (direct MP3 link from RSS)."""
    ep_id = episode["episode_id"]
    # Determine extension
    url_path = urlparse(episode["audio_url"]).path
    ext = Path(url_path).suffix or ".mp3"
    if ext not in (".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"):
        ext = ".mp3"

    raw_path = output_dir / f"{ep_id}{ext}"
    wav_path = output_dir / f"{ep_id}.wav"

    # Skip if already converted
    if wav_path.exists():
        return {"path": str(wav_path), "skipped": True}

    # Download raw audio
    if not raw_path.exists():
        cmd = [
            "curl", "-L", "-s", "-S",
            "-o", str(raw_path),
            "--max-time", "600",
            "--retry", "3",
            episode["audio_url"],
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=700)
        if result.returncode != 0:
            print(f"  ⚠ curl failed: {result.stderr[:150]}")
            if raw_path.exists():
                raw_path.unlink()
            return None

        # Check file is not empty or HTML error page
        if raw_path.exists() and raw_path.stat().st_size < 10000:
            print(f"  ⚠ File too small ({raw_path.stat().st_size} bytes), probably error page")
            raw_path.unlink()
            return None

    # Convert to 16kHz mono WAV with ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", str(raw_path),
        "-ar", "16000", "-ac", "1",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ⚠ ffmpeg failed: {result.stderr[:150]}")
        return None

    # Remove raw file to save space
    if raw_path.exists() and raw_path != wav_path:
        raw_path.unlink()

    return {"path": str(wav_path), "skipped": False}


# ── Modal Functions ───────────────────────────────────────────────────────────


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=30 * MINUTES,
    cpu=2,
    memory=2048,
)
def discover_episodes(num_episodes: int = 100, dry_run: bool = False) -> list[dict]:
    """Discover episodes from RSS feeds."""
    import requests

    all_episodes = []

    for source in PODCAST_FEEDS:
        print(f"\n📡 Fetching: {source['name']}")
        try:
            episodes = _fetch_feed(source["rss"])
        except Exception as e:
            print(f"  ⚠ Feed error: {e}")
            continue

        # Filter by duration (20 min - 6 hours)
        filtered = [ep for ep in episodes if 1200 <= ep["duration"] <= 21600]

        # Sort longest first, take top N
        filtered.sort(key=lambda x: x["duration"], reverse=True)
        selected = filtered[:source["target"]]

        for ep in selected:
            ep["source_name"] = source["name"]
            ep["source_region"] = source["region"]
            ep["source_format"] = source["format"]
            ep["episode_id"] = f"{source['name'].lower().replace(' ', '_')}_{ep['id']}"

        total_hrs = sum(e["duration"] for e in selected) / 3600
        print(f"   Found {len(selected)} episodes ({total_hrs:.1f} hours)")
        all_episodes.extend(selected)

    # Deduplicate by audio URL
    seen_urls = set()
    unique = []
    for ep in all_episodes:
        if ep["audio_url"] not in seen_urls:
            seen_urls.add(ep["audio_url"])
            unique.append(ep)

    # Sort longest first, trim to count
    unique.sort(key=lambda x: x["duration"], reverse=True)
    unique = unique[:num_episodes]

    total_hours = sum(e["duration"] for e in unique) / 3600
    print(f"\n{'='*60}")
    print(f"Total: {len(unique)} episodes, {total_hours:.1f} hours")
    print(f"{'='*60}")

    if dry_run:
        for i, ep in enumerate(unique, 1):
            hrs = ep["duration"] / 3600
            print(f"  {i:3d}. [{ep['source_region']:8s}] {ep['source_name']:30s} | {hrs:.1f}h | {ep['title'][:55]}")
        return unique

    return unique


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=20 * MINUTES,
    cpu=2,
    memory=2048,
)
def download_episode(episode: dict) -> Optional[dict]:
    """Download a single podcast episode from RSS."""
    _ensure_dirs()

    print(f"⬇️  {episode['title'][:55]}...")
    result = _download_episode_audio(episode, AUDIO_DIR)

    if result is None:
        print(f"  ❌ Failed: {episode['title'][:55]}")
        return None

    entry = {
        **episode,
        "local_path": result["path"],
        "downloaded_at": datetime.utcnow().isoformat(),
        "skipped": result.get("skipped", False),
    }

    if not result.get("skipped"):
        dur = episode.get("duration", 0)
        print(f"  ✅ {episode['title'][:50]} ({dur/60:.0f} min)")

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

    audio_path = Path(episode["local_path"])
    if not audio_path.exists():
        print(f"  ⚠ Audio not found: {audio_path}")
        return None

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

    print(f"  ✅ {len(result['segments'])} segments ({episode.get('duration', 0)/60:.0f} min)")
    return transcript_data


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=10 * MINUTES,
)
def save_manifest(episodes: list[dict]) -> str:
    _ensure_dirs()

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    volume.commit()
    total_hrs = sum(e.get("duration", 0) for e in episodes if e) / 3600
    successful = sum(1 for e in episodes if e is not None)
    print(f"\n📋 Manifest: {successful} episodes, {total_hrs:.1f} hours → {MANIFEST_FILE}")
    return str(MANIFEST_FILE)


@app.function(
    image=download_image,
    volumes={str(VOL_MOUNT): volume},
    timeout=30 * MINUTES,
)
def get_stats() -> dict:
    """Get download statistics from volume."""
    if not MANIFEST_FILE.exists():
        return {"error": "No manifest found"}

    episodes = []
    with open(MANIFEST_FILE) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    total_hrs = sum(e.get("duration", 0) for e in episodes) / 3600

    # Check which audio files exist
    audio_files = list(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []
    total_size_gb = sum(f.stat().st_size for f in audio_files) / 1024**3

    return {
        "episodes": len(episodes),
        "total_hours": round(total_hrs, 1),
        "audio_files": len(audio_files),
        "total_size_gb": round(total_size_gb, 2),
    }


# ── Local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    num_episodes: int = 100,
    dry_run: bool = False,
    transcribe: bool = False,
):
    print("=" * 60)
    print("🎙️  Dia-Spanish RSS Podcast Collector")
    print("=" * 60)

    # Phase 1: Discover
    print(f"\n📡 Phase 1: Discovering episodes (target: {num_episodes})...")
    episodes = discover_episodes.remote(num_episodes=num_episodes, dry_run=dry_run)

    if dry_run:
        print(f"\nDry run complete. {len(episodes)} episodes found.")
        return

    if not episodes:
        print("No episodes found!")
        return

    # Phase 2: Download
    print(f"\n⬇️  Phase 2: Downloading {len(episodes)} episodes via RSS...")
    results = []
    batch_size = 10
    for chunk_start in range(0, len(episodes), batch_size):
        chunk = episodes[chunk_start:chunk_start + batch_size]
        chunk_results = download_episode.map(chunk)
        results.extend(chunk_results)
        print(f"  Progress: {min(chunk_start + batch_size, len(episodes))}/{len(episodes)}")

    successful = [r for r in results if r is not None]
    failed = len(episodes) - len(successful)
    total_hrs = sum(r.get("duration", 0) for r in successful) / 3600

    print(f"\n✅ Downloaded: {len(successful)}/{len(episodes)} ({total_hrs:.1f} hours)")
    if failed:
        print(f"❌ Failed: {failed}")

    # Save manifest
    save_manifest.remote(successful)

    # Phase 3: Transcribe
    if transcribe and successful:
        print(f"\n🎙️  Phase 3: Transcribing {len(successful)} episodes...")
        trans_results = transcribe_episode.map(successful)
        trans_ok = sum(1 for r in trans_results if r is not None)
        print(f"✅ Transcribed: {trans_ok}/{len(successful)}")

    # Final stats
    stats = get_stats.remote()
    print(f"\n📊 Volume stats: {stats}")
    print("\n🎉 Done! Retrieve with: modal volume get dia-spanish-vol data/podcasts_rss .")
