"""Solo baselines and the oracle ceiling.

Run every pool worker alone over a split. best-single = the strongest solo
worker (the bar every router must clear); oracle = per-task best (the routing
headroom ceiling). Recomputed once per (pool, suite) version and cached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import engine, score, suites
from .routers import SingleWorkerRouter
from .workers import Pool


@dataclass(frozen=True)
class Baselines:
    split: str
    seed: int
    solo: dict[str, list[bool]]  # worker -> correctness vector (task-id order)
    solo_axes: dict[str, dict]
    best_single: str
    best_single_results: list[dict]  # serialized TaskResults for paired judging
    oracle_accuracy: float
    best_worker_per_task: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1)


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
        seed=seed,
        solo=solo,
        solo_axes=solo_axes,
        best_single=best,
        best_single_results=[r.__dict__ | {"steps": []} for r in solo_results[best]],
        oracle_accuracy=oracle_hits / len(ordered),
        best_worker_per_task=per_task_best,
    )


def load(path: str) -> Baselines:
    with open(path) as f:
        return Baselines(**json.load(f))


def deserialize_results(rows: list[dict]) -> list[engine.TaskResult]:
    return [engine.TaskResult(**{**row, "steps": []}) for row in rows]
