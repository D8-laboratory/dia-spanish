# Adaption Labs Integration Guide

## Setup

```bash
pip install adaption
export ADAPTION_API_KEY="pt_live_..."
```

Get your key at: https://adaptionlabs.ai/app/settings?tab=api_keys

## Workflow

```bash
# Upload + estimate cost
python adaption/ingest.py --dataset data/processed/spanish_dialogues_train.jsonl --estimate-only

# Upload + run adaptation + download
python adaption/ingest.py \
  --dataset data/processed/spanish_dialogues_train.jsonl \
  --adapt \
  --download data/processed/adapted/spanish_dialogues_adapted.jsonl
```

## Budget

- **Credits**: $1,000 USD
- **Strategy**: Estimate small batches first, then scale

## AutoScientist Challenge

- **Category**: Language (Part 1: June 8 - July 5)
- **Requirements**: Release dataset + weights to HF + Kaggle
