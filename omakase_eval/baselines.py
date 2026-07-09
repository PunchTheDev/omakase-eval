"""Solo baselines and the oracle ceiling.

Run every pool worker alone over a split. best-single = the strongest solo
worker (the bar every router must clear); oracle = per-task best (the routing
headroom ceiling). Recomputed once per (pool, suite, split) version and cached.

Every cached artifact is *stamped* — split, seed fingerprint, pool version,
suite version — and refuses to be used out of context. A baseline scored on the
public dev split cannot judge a run scored on a rotated gate split: the task ids
are disjoint, so a paired comparison against it is not merely inaccurate, it is
meaningless. Loudly failing beats silently zipping two unrelated vectors.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import engine, score, suites
from .routers import SingleWorkerRouter
from .workers import Pool


class StaleBaseline(RuntimeError):
    """A cached baseline does not match the split/pool/suite being scored."""


@dataclass(frozen=True)
class Baselines:
    split: str
    seed: int | None  # None on private splits — never commit a gate seed
    solo: dict[str, list[bool]]  # worker -> correctness vector (task-id order)
    solo_axes: dict[str, dict]
    best_single: str
    best_single_results: list[dict]  # serialized TaskResults for paired judging
    oracle_accuracy: float
    best_worker_per_task: dict[str, str]
    seed_fingerprint: str = ""
    pool_version: str = ""
    suite_version: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)


def path(runs_dir: str, split: str) -> str:
    return os.path.join(runs_dir, f"baselines.{split}.json")


PUBLIC_SPLIT = "dev"


def compute(pool: Pool, split: str, seed: int, budget: engine.Budget = engine.Budget()) -> Baselines:
    tasks = suites.generate_split(split, seed)
    ordered = sorted(tasks, key=lambda t: t.id)
    solo_results: dict[str, list[engine.TaskResult]] = {}
    for worker in pool.workers:
        solo_results[worker] = engine.run_split(SingleWorkerRouter(worker), tasks, pool, seed, split, budget)

    solo = {w: score.correctness_vector(rs) for w, rs in solo_results.items()}
    solo_axes = {w: score.axes(rs).__dict__ for w, rs in solo_results.items()}
    best = max(solo_axes, key=lambda w: solo_axes[w]["accuracy"])

    per_task_best: dict[str, str] = {}
    oracle_hits = 0
    for i, task in enumerate(ordered):
        winners = [w for w in solo if solo[w][i]]
        if winners:
            oracle_hits += 1
            # strongest correct worker in this task's suite, cost as tiebreak
            per_task_best[task.id] = max(
                winners,
                key=lambda w: (solo_axes[w]["per_suite"][task.suite], -pool.workers[w].cost_per_1k),
            )

    return Baselines(
        split=split,
        seed=seed if split == PUBLIC_SPLIT else None,
        solo=solo,
        solo_axes=solo_axes,
        best_single=best,
        best_single_results=[r.__dict__ | {"steps": []} for r in solo_results[best]],
        oracle_accuracy=oracle_hits / len(ordered),
        best_worker_per_task=per_task_best,
        seed_fingerprint=suites.split_fingerprint(split, seed),
        pool_version=pool.version,
        suite_version=suites.SUITE_VERSION,
    )


def load(p: str) -> Baselines:
    with open(p) as f:
        blob = json.load(f)
    known = set(Baselines.__dataclass_fields__)
    return Baselines(**{k: v for k, v in blob.items() if k in known})


def load_for(runs_dir: str, split: str, seed: int, pool: Pool) -> Baselines:
    """Load the baseline for this exact round, or refuse."""
    p = path(runs_dir, split)
    if not os.path.exists(p):
        raise StaleBaseline(
            f"no baseline for split {split!r} at {p} — compute one on the trusted host "
            f"(`omakase-eval baselines --split {split}`) before judging submissions."
        )
    base = load(p)
    want = suites.split_fingerprint(split, seed)
    if base.split != split:
        raise StaleBaseline(f"baseline is for split {base.split!r}, scoring {split!r}")
    if base.seed_fingerprint and base.seed_fingerprint != want:
        raise StaleBaseline(
            f"baseline for {split!r} was scored on a different seed (fingerprint mismatch) — "
            "the split rotated; recompute the baseline before judging."
        )
    if base.pool_version and base.pool_version != pool.version:
        raise StaleBaseline(
            f"baseline was scored against pool {base.pool_version!r}, running {pool.version!r} — "
            "the pool changed; recompute the baseline."
        )
    if base.suite_version and base.suite_version != suites.SUITE_VERSION:
        raise StaleBaseline(
            f"baseline was scored under {base.suite_version!r}, running {suites.SUITE_VERSION!r} — "
            "the generators changed; recompute the baseline."
        )
    return base


def deserialize_results(rows: list[dict]) -> list[engine.TaskResult]:
    return [engine.TaskResult(**{**row, "steps": []}) for row in rows]


# -- king-of-the-hill incumbent ---------------------------------------------
# The crown is taken by beating the *current champion*, not just the best single
# worker. The champion's scored results are cached here (like Harness's main
# baseline); at genesis there is no champion, so the best-single floor is used.

def champion_path(runs_dir: str) -> str:
    return os.path.join(runs_dir, "champion-baseline.json")


def load_incumbent(runs_dir: str, floor_results: list[engine.TaskResult],
                   split: str, seed: int) -> list[engine.TaskResult]:
    p = champion_path(runs_dir)
    if not os.path.exists(p):
        return floor_results  # genesis: must beat the best single worker
    with open(p) as f:
        cached = json.load(f)
    want = suites.split_fingerprint(split, seed)
    if cached.get("seed_fingerprint", "") != want or cached.get("split") != split:
        raise StaleBaseline(
            "the champion's cached results were scored on a different split/seed — the round "
            "rotated. Re-run the reigning champion on the current split (rebaseline) before "
            "judging challengers; pairing across splits would compare disjoint tasks."
        )
    return deserialize_results(cached["results"])


def write_champion(runs_dir: str, results: list[engine.TaskResult], split: str, seed: int) -> None:
    os.makedirs(runs_dir, exist_ok=True)
    rows = [r.__dict__ | {"steps": []} for r in results]
    accuracy = sum(r.correct for r in results) / len(results)
    with open(champion_path(runs_dir), "w") as f:
        json.dump({"split": split,
                   "seed": seed if split == PUBLIC_SPLIT else None,  # never commit a gate seed
                   "seed_fingerprint": suites.split_fingerprint(split, seed),
                   "accuracy": accuracy, "results": rows}, f)
