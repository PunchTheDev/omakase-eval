"""Showcase runner — 'open-weights stack vs. the labs'.

Runs the champion router and every configured contender (lab models solo, best
single pool worker) over the same split and emits one blob: the dashboard's
/vs-labs bar chart. In dev the labs are mock stand-ins; in production they are
real API contenders and the runs carry attestation references.
"""
from __future__ import annotations

from . import engine, score, suites
from .routers import SingleWorkerRouter
from .workers import Pool


def run(champion_router, champion_name: str, contenders: Pool, pool: Pool,
        split: str, seed: int, run_task=engine.run_task) -> dict:
    """run_task defaults to the reference engine; pass the champion harness's to showcase the full stack."""
    tasks = suites.generate_split(split, seed)
    stack = [run_task(champion_router, t, pool, seed, split) for t in tasks]
    bars = {champion_name: _axes(stack)}
    for worker_id in contenders.workers:
        results = engine.run_split(SingleWorkerRouter(worker_id), tasks, contenders, seed, split)
        bars[worker_id] = _axes(results)
    return {"split": split, "seed": seed, "n_tasks": len(tasks), "contenders": bars}


def _axes(results) -> dict:
    a = score.axes(results)
    return {"accuracy": round(a.accuracy, 4), "per_suite": {k: round(v, 4) for k, v in a.per_suite.items()},
            "cost_per_task": round(a.cost_per_task, 4)}
