"""Real-benchmark loader.

The procedural suites (suites.py) are the un-memorizable synthetic core — real,
reproducible from (split, seed), nothing to download or leak. This module loads
*fixed* benchmark instances from JSONL for the production gate suite (hidden
splits of GPQA-D / MMLU-Pro / LiveCodeBench, fresh-harvested GitHub tasks). Both
produce the same Task objects the engine scores, so the harness never changes.

JSONL schema (one object per line):
    {"id","suite","prompt","options":[...]|[],"answer":"<canonical>"}
"""
from __future__ import annotations

import json

from .suites import SUITES, Task


def load_jsonl(path: str) -> list[Task]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            tasks.append(Task(o["id"], o["suite"], o["prompt"],
                              tuple(o.get("options", [])), o["answer"],
                              meta=o.get("meta", {})))
    return tasks


def sample_jsonl(path: str, split: str, seed: int, n: int) -> list[Task]:
    """Hidden-subset holdout: draw n items from a public pool by (split, seed).

    The pool file is public, but *which* items a round scores is a secret
    function of the gate seed — so acing a round requires handling the whole
    distribution, not memorizing a fixed slice (the Kaggle private-leaderboard
    property). dev and gate draw disjoint-ish subsets; retired gate seeds are
    published so yesterday's holdout becomes training data.
    """
    import hashlib
    import random

    pool = load_jsonl(path)
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{path}|{split}|{seed}".encode()).digest()[:8], "big"))
    picked = rng.sample(pool, min(n, len(pool)))
    # re-id so the split/seed provenance is in the task id (grading is content-based)
    return [Task(f"{split}-{t.suite}-{i:04d}", t.suite, t.prompt, t.options, t.answer, meta=t.meta)
            for i, t in enumerate(picked)]


def load_split(source: dict, split: str, seed: int, per_suite: int = 40) -> list[Task]:
    """Unified entry point. source = {'kind':'procedural'} or {'kind':'jsonl','path':...}.

    The maintainer's run config names the source; miners self-score against the
    procedural dev source, the canonical rerun against the configured gate source.
    """
    if source.get("kind", "procedural") == "procedural":
        from .suites import generate_split

        return generate_split(split, seed, per_suite)
    tasks = load_jsonl(source["path"])
    bad = [t.suite for t in tasks if t.suite not in SUITES]
    if bad:
        raise ValueError(f"unknown suites in dataset: {sorted(set(bad))}")
    return tasks
