"""
Chunk diarized transcripts into 3-17s training clips for Dia fine-tuning.

Input:  results/transcripts/*.json  (WhisperX output with speaker diarization)
Output: results/chunk_manifest.jsonl  (one line per chunk: text + [start,end] + metadata)

This is a TEXT/TIMESTAMP-ONLY pass. No audio is touched. The companion Modal
job (cut_audio_from_manifest.py) reads this manifest and slices WAVs.

Speaker strategy: [S1]/[S2] generic (Dia original style).
  - Rank speakers per episode by total talk time.
  - Keep top-N (default 2) as [S1]..[SN]. Drop the rest.
  - Within each chunk, speakers are labeled by their per-episode rank,
    so voices vary across episodes (accepted tradeoff of generic tagging).
"""

import argparse
import json
import statistics
from pathlib import Path


def rank_speakers(segments, min_speaker_sec, num_speakers):
    """Return {raw_speaker: '[SN]'} for the top-N speakers by talk time."""
    talk = {}
    for s in segments:
        spk = s.get("speaker")
        if not spk:
            continue
        talk[spk] = talk.get(spk, 0.0) + (s["end"] - s["start"])
    ranked = [spk for spk, sec in sorted(talk.items(), key=lambda x: -x[1])
              if sec >= min_speaker_sec]
    ranked = ranked[:num_speakers]
    return {spk: f"[S{i+1}]" for i, spk in enumerate(ranked)}, ranked


def split_long_segment(seg, max_sec):
    """Split a segment longer than max_sec at word boundaries.

    Returns a list of pseudo-segments (dicts with start/end/text/speaker),
    each <= max_sec. Falls back to [seg] if no word timestamps available.
    """
    words = seg.get("words") or []
    if len(words) < 2:
        return [seg]
    spk = seg["speaker"]
    out = []
    cur_words = []
    cur_start = words[0].get("start", seg["start"])
    for w in words:
        w_end = w.get("end", seg["end"])
        if cur_words and (w_end - cur_start) > max_sec:
            out.append({
                "start": cur_start,
                "end": cur_words[-1].get("end", cur_start),
                "text": " ".join(x.get("word", "").strip() for x in cur_words),
                "speaker": spk,
            })
            cur_start = w.get("start", w_end)
            cur_words = [w]
        else:
            cur_words.append(w)
    if cur_words:
        out.append({
            "start": cur_start,
            "end": cur_words[-1].get("end", seg["end"]),
            "text": " ".join(x.get("word", "").strip() for x in cur_words),
            "speaker": spk,
        })
    return out or [seg]


def chunk_episode(segments, spk_map, min_sec, max_sec, target_sec, max_gap):
    """Group consecutive segments into chunks of [min_sec, max_sec].

    Bias toward multi-turn dialogue: when a segment is from a *different*
    speaker than the current chunk's last speaker, allow the chunk to grow
    up to max_sec (not just target_sec) so we capture the exchange.
    """
    # Keep only segments from mapped speakers; split any over-long ones.
    segs = []
    for s in segments:
        if s.get("speaker") not in spk_map:
            continue
        if (s["end"] - s["start"]) > max_sec:
            segs.extend(split_long_segment(s, max_sec))
        else:
            segs.append(s)
    if not segs:
        return []

    chunks = []
    cur = {"segs": [], "start": None, "end": None, "last_spk": None}

    def flush():
        if not cur["segs"]:
            return
        dur = cur["end"] - cur["start"]
        if dur >= min_sec:
            chunks.append(list(cur["segs"]))
        cur["segs"].clear()
        cur["start"] = None
        cur["end"] = None
        cur["last_spk"] = None

    for s in segs:
        prospective_dur = (s["end"] - cur["start"]) if cur["start"] is not None else (s["end"] - s["start"])
        gap = (s["start"] - cur["end"]) if cur["end"] is not None else 0.0
        cross_speaker = cur["last_spk"] is not None and s["speaker"] != cur["last_spk"]

        must_flush = False
        if cur["segs"]:
            if prospective_dur > max_sec:
                must_flush = True
            elif gap > max_gap:
                must_flush = True
            # If same speaker and already past target, close to avoid
            # giant single-turn monologues. If cross-speaker, keep going
            # up to max_sec to capture the dialogue.
            elif not cross_speaker and (cur["end"] - cur["start"]) >= target_sec:
                must_flush = True

        if must_flush:
            flush()

        if not cur["segs"]:
            cur["start"] = s["start"]
        cur["end"] = s["end"]
        cur["last_spk"] = s["speaker"]
        cur["segs"].append(s)

        # Hard close at max_sec (guaranteed by split, but be safe).
        if cur["end"] - cur["start"] >= max_sec - 0.05:
            flush()

    flush()
    return chunks


def chunk_to_text(chunk_segs, spk_map):
    """Collapse consecutive same-speaker segments; emit '[SN] text ...'."""
    turns = []
    for s in chunk_segs:
        tag = spk_map[s["speaker"]]
        text = s["text"].strip()
        if not text:
            continue
        if turns and turns[-1][0] == tag:
            turns[-1] = (tag, turns[-1][1] + " " + text)
        else:
            turns.append((tag, text))
    if not turns:
        return None
    return " ".join(f"{tag} {text}" for tag, text in turns)


def process_episode(json_path, min_speaker_sec, num_speakers,
                    min_sec, max_sec, target_sec, max_gap):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    episode_id = data.get("episode_id", Path(json_path).stem)
    segments = data.get("segments", [])
    if not segments:
        return [], {"episode_id": episode_id, "error": "no segments"}

    spk_map, kept = rank_speakers(segments, min_speaker_sec, num_speakers)
    raw_chunks = chunk_episode(segments, spk_map,
                               min_sec, max_sec, target_sec, max_gap)

    chunks = []
    for i, csegs in enumerate(raw_chunks):
        text = chunk_to_text(csegs, spk_map)
        if not text:
            continue
        start = csegs[0]["start"]
        end = csegs[-1]["end"]
        dur = end - start
        spkrs = sorted({spk_map[s["speaker"]] for s in csegs})
        chunks.append({
            "chunk_id": f"{episode_id}_{i:04d}",
            "episode_id": episode_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(dur, 3),
            "text": text,
            "speakers": spkrs,
            "num_turns": text.count("[S"),
        })

    stats = {
        "episode_id": episode_id,
        "num_segments": len(segments),
        "speakers_kept": [f"{spk}={spk_map[spk]}" for spk in kept],
        "num_chunks": len(chunks),
        "total_chunk_sec": round(sum(c["duration"] for c in chunks), 1),
        "avg_chunk_sec": round(statistics.mean(c["duration"] for c in chunks), 2)
                         if chunks else 0,
    }
    return chunks, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-dir", default="results/transcripts")
    ap.add_argument("--output", default="results/chunk_manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N episodes (0 = all)")
    ap.add_argument("--min-speaker-sec", type=float, default=60.0)
    ap.add_argument("--num-speakers", type=int, default=2)
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=17.0)
    ap.add_argument("--target-sec", type=float, default=12.0)
    ap.add_argument("--max-gap", type=float, default=1.5)
    args = ap.parse_args()

    tdir = Path(args.transcripts_dir)
    files = sorted(tdir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_stats = []
    n_chunks = 0
    with out.open("w", encoding="utf-8") as fout:
        for f in files:
            chunks, stats = process_episode(
                f, args.min_speaker_sec, args.num_speakers,
                args.min_sec, args.max_sec, args.target_sec, args.max_gap,
            )
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
            all_stats.append(stats)
            n_chunks += stats.get("num_chunks", 0)

    # Aggregate
    durs = [c["duration"] for c in chunks] if False else None  # placeholder
    print(f"\n{'='*60}")
    print(f"Episodes processed: {len(files)}")
    print(f"Total chunks: {n_chunks}")
    print(f"Manifest: {out}")
    print(f"{'='*60}")
    print(f"{'episode':40s} {'segs':>5s} {'chunks':>6s} {'min':>5s} {'avg':>5s} {'max':>5s}  speakers")
    print("-" * 80)
    for st in all_stats:
        kept = ",".join(st.get("speakers_kept", []))
        print(f"{st['episode_id'][:40]:40s} {st['num_segments']:5d} "
              f"{st['num_chunks']:6d} {'-':>5s} {st['avg_chunk_sec']:5.1f} {'-':>5s}  {kept}")


if __name__ == "__main__":
    main()
