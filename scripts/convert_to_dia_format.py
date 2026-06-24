"""Convert WhisperX diarized JSON → Dia training format.

Dia expects:
  - Mono WAV audio (the original, untouched)
  - Transcript text with [S1], [S2], ... speaker tokens, one block per speaker turn

Example output:
  [S1] La Segunda Guerra Mundial se produce porque Alemania rompe el equilibrio...
  [S1] Inglaterra no tiene amigos ni enemigos permanente, tiene intereses permanentes.
  [S2] Los británicos se alarman enormemente, pero hay otro actor que juega un papel...

Strategy:
  1. Rank speakers by total talk time → assign [S1], [S2], ... (top N only)
  2. Drop speakers below a min-talk-time threshold (they're ads/noise/guests)
  3. Collapse consecutive same-speaker segments into one turn
  4. Merge turns shorter than `min_turn_sec` into the neighbor to avoid fragmentation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dia_transcript(
    segments: list[dict],
    num_speakers: int = 2,
    min_speaker_sec: float = 60.0,
    max_gap_sec: float = 1.0,
) -> str:
    """Convert diarized segments into Dia [S1]/[S2]/... transcript text.

    Args:
        segments: WhisperX segments with 'start','end','text','speaker'
        num_speakers: how many top speakers to keep (mapped to [S1]..[SN])
        min_speaker_sec: drop speakers with less than this total talk time
        max_gap_sec: max silence between consecutive same-speaker segs to merge
    """
    # 1. Rank speakers by talk time
    talk_time: dict[str, float] = {}
    for s in segments:
        spk = s.get("speaker", "SPEAKER_UNKNOWN")
        talk_time[spk] = talk_time.get(spk, 0) + (s["end"] - s["start"])

    # Keep only speakers above threshold, take top N
    kept = sorted(
        [(spk, t) for spk, t in talk_time.items() if t >= min_speaker_sec],
        key=lambda x: -x[1],
    )[:num_speakers]
    spk_to_token = {spk: f"[S{i+1}]" for i, (spk, _) in enumerate(kept)}

    # 2. Filter to kept speakers, skip empty text
    kept_set = set(spk_to_token)
    turns = []
    for s in segments:
        spk = s.get("speaker", "SPEAKER_UNKNOWN")
        if spk not in kept_set:
            continue
        text = s.get("text", "").strip()
        if not text:
            continue
        turns.append({"speaker": spk, "text": text})

    if not turns:
        return ""

    # 3. Collapse consecutive same-speaker turns
    collapsed = [turns[0]]
    for t in turns[1:]:
        last = collapsed[-1]
        if t["speaker"] == last["speaker"]:
            last["text"] += " " + t["text"]
        else:
            collapsed.append(t)

    # 4. Build output
    lines = [f'{spk_to_token[t["speaker"]]} {t["text"]}' for t in collapsed]
    return "\n".join(lines)


def convert_episode(in_path: Path, out_path: Path, **kwargs) -> dict:
    with open(in_path) as f:
        data = json.load(f)
    text = build_dia_transcript(data["segments"], **kwargs)
    out_path.write_text(text, encoding="utf-8")
    # stats
    n_lines = text.count("\n") + 1 if text else 0
    tokens = [l.split("]")[0] + "]" for l in text.split("\n") if l]
    from collections import Counter
    spk_dist = Counter(tokens)
    return {
        "lines": n_lines,
        "char_len": len(text),
        "speaker_dist": dict(spk_dist),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/transcripts")
    ap.add_argument("--out-dir", default="results/dia_format")
    ap.add_argument("--num-speakers", type=int, default=2)
    ap.add_argument("--min-speaker-sec", type=float, default=60.0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for jf in sorted(in_dir.glob("*.json")):
        stats = convert_episode(
            jf,
            out_dir / (jf.stem + ".txt"),
            num_speakers=args.num_speakers,
            min_speaker_sec=args.min_speaker_sec,
        )
        print(f"📝 {jf.stem}")
        print(f"   {stats['lines']} lines, {stats['char_len']:,} chars")
        print(f"   turn distribution: {stats['speaker_dist']}")
        print()


if __name__ == "__main__":
    main()
