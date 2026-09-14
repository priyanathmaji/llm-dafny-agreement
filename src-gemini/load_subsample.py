"""
Single source of truth for "which documents are in the frozen tier
subsample" -- every script (scanner, tier-filtered ablation, etc.)
should import this instead of touching clean_evaluation_pool.jsonl
or any random-sampling logic directly.
"""
import json

def load_subsample(path: str):
    """Returns list of full pool records (dicts) from a frozen
    subsample file written by save_tier_subsample.py."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def load_subsample_doc_ids(path: str):
    return {r["doc_id"] for r in load_subsample(path)}