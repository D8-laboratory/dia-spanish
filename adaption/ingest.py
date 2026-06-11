"""
Adaption Labs integration: ingest Spanish dialogue data, run adaptation, export results.

Uses the Adaption Python SDK to:
1. Upload Spanish dialogue datasets
2. Run data adaptation (dedup, quality scoring, rephrasing)
3. Export training-ready data

Requires: ADAPTION_API_KEY environment variable
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ingest Spanish dialogues into Adaption Labs")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--name", type=str, default=None, help="Dataset name in Adaption")
    parser.add_argument("--adapt", action="store_true", help="Run adaptation after upload")
    parser.add_argument("--download", type=str, default=None, help="Download adapted data to path")
    parser.add_argument("--estimate-only", action="store_true", help="Estimate cost without running")
    args = parser.parse_args()

    try:
        from adaption import Adaption, DatasetTimeout
    except ImportError:
        print("Error: Install the Adaption SDK: pip install adaption")
        sys.exit(1)

    api_key = os.environ.get("ADAPTION_API_KEY")
    if not api_key:
        print("Error: Set ADAPTION_API_KEY environment variable")
        print("  Get your key at: https://adaptionlabs.ai/app/settings?tab=api_keys")
        sys.exit(1)

    client = Adaption(api_key=api_key)

    # Step 1: Upload dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"Uploading {dataset_path}...")
    name = args.name or dataset_path.stem
    result = client.datasets.upload_file(str(dataset_path), name=name)
    dataset_id = result.dataset_id
    print(f"Dataset uploaded: {dataset_id} (name: {name})")

    if not args.adapt and not args.estimate_only:
        print(f"\nDataset ID: {dataset_id}")
        print("Use --adapt to run adaptation, or --estimate-only to check cost")
        return

    # Step 2: Estimate or run adaptation
    print("\nRunning adaptation estimate...")
    estimate = client.datasets.run(
        dataset_id,
        column_mapping={"prompt": "transcript"},
        estimate=True,
    )
    print(f"Estimated credits: {estimate.estimated_credits_consumed}")

    if args.estimate_only:
        return

    if args.adapt:
        print("\nStarting adaptation run...")
        run = client.datasets.run(
            dataset_id,
            column_mapping={"prompt": "transcript"},
            recipe_specification={
                "recipes": {
                    "deduplication": True,
                    "prompt_rephrase": True,
                    "reasoning_traces": True,
                },
            },
        )
        print(f"Run ID: {run.run_id}")
        print(f"Estimated credits: {run.estimated_credits_consumed}")

        # Wait for completion
        print("Waiting for completion...")
        try:
            status = client.datasets.wait_for_completion(dataset_id, timeout=3600)
            print(f"Status: {status.status}")
            if status.error:
                print(f"Error: {status.error.message}")
                sys.exit(1)
        except DatasetTimeout as e:
            print(f"Timeout after {e.timeout}s (status: {e.last_status})")
            print("Check status manually in the Adaption dashboard")
            sys.exit(1)

        # Download results
        if args.download:
            download_path = Path(args.download)
            download_path.parent.mkdir(parents=True, exist_ok=True)
            url = client.datasets.download(dataset_id)
            print(f"Download URL: {url}")
            print(f"Download manually or use: curl -o {download_path} '{url}'")


if __name__ == "__main__":
    main()
