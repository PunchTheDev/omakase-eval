"""Round composition — the Kaggle public-dev / private-gate structure.

A round config declares how a split is built from sources. Two splits per round:

- **dev** (public): fixed seed, published — miners self-score freely.
- **gate** (private): a rotation seed Punch keeps secret until after scoring,
  bumped every round so nothing can be memorized. Same distribution as dev.

Sources compose in tiers (least → most un-gameable is the other way, but all
are refreshable):
  1. procedural  — generated from (split, seed); infinite, verifiable, zero leak
  2. jsonl       — hidden holdout of hard curated knowledge (grad STEM, expert
                   domains); the private slice is never published until retired

The maintainer tunes the mix + counts so the best single worker lands in the
55–75% band (max routing headroom). Grading is always objective (see suites.py).
"""
from __future__ import annotations

import json

from . import datasets, suites


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_split(config: dict, split: str, seed: int) -> list[suites.Task]:
    """Compose a split from the round config's sources. `split` ∈ {dev, gate}."""
    tasks: list[suites.Task] = []
    for src in config["sources"]:
        if src["kind"] == "procedural":
            tasks += suites.generate_split(
                split, seed, per_suite=src.get("per_suite", 40),
                suites=tuple(src.get("suites", suites.SUITES)))
        elif src["kind"] == "jsonl":
            # hidden-subset holdout: sample n items from a public pool by (split, seed).
            # The pool is public; the per-round subset is a secret function of the seed.
            tasks += datasets.sample_jsonl(src["pool"], split, seed, src.get("count", 100))
        else:
            raise ValueError(f"unknown source kind {src['kind']!r}")
    return tasks


def descriptor(config: dict) -> dict:
    """A public, leak-free description of the suite for the dashboard /benchmarks page.

    Names sources, task types, counts and grading — never any gate instance or
    answer. Describing the task *types* fully is deliberate: a contributor should
    be able to train against the real distribution, since only the instances (and
    the seed that produces them) are secret.
    """
    rows = []
    for src in config["sources"]:
        if src["kind"] == "procedural":
            for s in src.get("suites", suites.SUITES):
                rows.append({"suite": s, "description": suites.DESCRIPTIONS.get(s, ""),
                             "source": "procedural", "graded": "objective",
                             "per_split": src.get("per_suite", 40)})
        else:
            rows.append({"suite": src.get("name", "knowledge-holdout"),
                         "description": "general-knowledge MCQ drawn as a secret per-round subset "
                                        "of a published question pool",
                         "source": "hidden-holdout",
                         "graded": "objective", "per_split": src.get("count", "—")})
    return {
        "name": config.get("name", "gate"),
        "structure": "public dev split (self-score) + private gate split (scores the crown, rotated "
                     "each round) — same suite mix, generators, and grading on both; only the seed differs",
        "suites": rows,
    }
