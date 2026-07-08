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
