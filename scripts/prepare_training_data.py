"""
Prepare Spanish dialogue data for Dia fine-tuning.

Converts various data sources into the format expected by Dia training:
- Audio: stereo WAV (left=S1, right=S2), 44.1kHz
- Transcript: JSONL with [S1]/[S2] tags and nonverbal cues
- Metadata: duration, domain, region, speaker info
"""

import argparse
import json
from pathlib import Path


def prepare_synthetic_dialogues(input_path: str, output_path: str):
    """Convert synthetic dialogue JSONL to Dia training format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line.strip())

            # Validate transcript format
            transcript = data.get("transcript", "")
            if "[S1]" not in transcript and "[S2]" not in transcript:
                continue

            # Convert to Dia training format
            training_sample = {
                "id": data["id"],
                "transcript": transcript,
                "metadata": {
                    "domain": data.get("domain", "unknown"),
                    "region": data.get("region", "neutral"),
                    "source": "synthetic",
                    "num_turns": transcript.count("[S"),
                    "has_nonverbal": any(
                        tag in transcript
                        for tag in ["(risas)", "(suspira)", "(tose)", "(grita)", "(cantando)"]
                    ),
                },
            }

            fout.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} dialogues → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Spanish dialogue data for Dia training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--source", type=str, default="synthetic",
                       choices=["synthetic", "magichub", "podcast", "common_voice"])
    args = parser.parse_args()

    if args.source == "synthetic":
        prepare_synthetic_dialogues(args.input, args.output)
    else:
        print(f"Source type '{args.source}' preparation not yet implemented")
        print("TODO: Add converters for each data source")


if __name__ == "__main__":
    main()
