"""Publish the self-built evaluation sets to a Hugging Face dataset repo.

Usage:
  python scripts/push_eval_to_hf.py --repo-id <user>/amz-ops-eval --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    files = {
        "compliance": "compliance_cases.jsonl",
        "compliance_holdout": "compliance_holdout.jsonl",
        "retrieval": "retrieval_cases.jsonl",
        "agent": "agent_cases.jsonl",
    }
    dataset = DatasetDict(
        {
            split: Dataset.from_list(load_jsonl(ROOT / "data" / "eval" / name))
            for split, name in files.items()
        }
    )
    dataset.push_to_hub(args.repo_id, token=args.token, private=args.private)
    print(f"pushed {sum(len(split) for split in dataset.values())} rows to {args.repo_id}")


if __name__ == "__main__":
    main()
