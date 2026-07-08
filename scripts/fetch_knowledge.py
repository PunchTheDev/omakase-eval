#!/usr/bin/env python3
"""Scale the knowledge holdout pool from MMLU-Pro (hard, multi-domain, verified).

The committed data/knowledge.pool.jsonl is a small hand-verified seed so the
mechanism runs out of the box. For a real competition, expand the pool with
MMLU-Pro (~12k graduate-level MCQ, non-gated on HF). Run by the maintainer:

    pip install datasets
    python scripts/fetch_knowledge.py --out data/knowledge.pool.jsonl --n 4000

Then bump the round's gate seed so the new pool is live. The pool is public;
the per-round subset stays secret via the seed (see datasets.sample_jsonl).
GPQA-Diamond can be added the same way once you accept its HF terms.
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--dataset", default="TIGER-Lab/MMLU-Pro")
    args = ap.parse_args()

    from datasets import load_dataset  # heavy dep; maintainer-only

    ds = load_dataset(args.dataset, split="test")
    written = 0
    with open(args.out, "w") as f:
        for row in ds:
            if written >= args.n:
                break
            opts = row.get("options") or []
            ans_idx = row.get("answer_index")
            if not opts or ans_idx is None or ans_idx >= len(opts):
                continue
            task = {
                "id": f"k-knowledge-{written:05d}",
                "suite": "knowledge",
                "prompt": row["question"].strip(),
                "options": [str(o) for o in opts],
                "answer": str(opts[ans_idx]),
                "meta": {"domain": row.get("category", "mmlu-pro")},
            }
            f.write(json.dumps(task) + "\n")
            written += 1
    print(f"wrote {written} knowledge items → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
