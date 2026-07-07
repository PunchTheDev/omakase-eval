"""Composite scoring and verdicts.

Axes: accuracy (gated, paired vs. baseline), cost and latency (tolerance-banded
Pareto guards). The uplift that matters is against the best single worker —
a router that only beats other routers adds nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import stats
from .engine import TaskResult

SUITE_WEIGHTS = {"reasoning": 1.0, "math": 1.0, "code_qa": 1.0}
COST_TOLERANCE = 1.10  # candidate may cost up to +10% vs. baseline
LATENCY_TOLERANCE = 1.25


@dataclass(frozen=True)
class Axes:
    accuracy: float
    per_suite: dict[str, float]
    cost_per_task: float
    latency_p50_ms: float


def axes(results: list[TaskResult]) -> Axes:
    by_suite: dict[str, list[TaskResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)
    per_suite = {s: sum(r.correct for r in rs) / len(rs) for s, rs in by_suite.items()}
    total_w = sum(SUITE_WEIGHTS.get(s, 1.0) for s in per_suite)
    accuracy = sum(per_suite[s] * SUITE_WEIGHTS.get(s, 1.0) for s in per_suite) / total_w
    latencies = sorted(r.latency_ms for r in results)
    return Axes(
        accuracy=accuracy,
        per_suite=per_suite,
        cost_per_task=sum(r.cost for r in results) / len(results),
        latency_p50_ms=latencies[len(latencies) // 2],
    )


def correctness_vector(results: list[TaskResult]) -> list[bool]:
    return [r.correct for r in sorted(results, key=lambda r: r.task_id)]


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str
    comparison: stats.Comparison
    candidate: Axes
    baseline: Axes
    oracle_capture: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def judge(candidate: list[TaskResult], baseline: list[TaskResult],
          oracle_accuracy: float | None = None) -> Verdict:
    """Pass iff accuracy improves with significance and cost/latency stay in band."""
    cand_axes, base_axes = axes(candidate), axes(baseline)
    cmp_ = stats.compare(correctness_vector(candidate), correctness_vector(baseline))

    capture = None
    if oracle_accuracy is not None and oracle_accuracy > base_axes.accuracy:
        capture = (cand_axes.accuracy - base_axes.accuracy) / (oracle_accuracy - base_axes.accuracy)

    if not cmp_.significant:
        return Verdict(False, "accuracy gain not significant", cmp_, cand_axes, base_axes, capture)
    if cand_axes.cost_per_task > base_axes.cost_per_task * COST_TOLERANCE:
        return Verdict(False, "cost regression beyond tolerance", cmp_, cand_axes, base_axes, capture)
    if cand_axes.latency_p50_ms > base_axes.latency_p50_ms * LATENCY_TOLERANCE:
        return Verdict(False, "latency regression beyond tolerance", cmp_, cand_axes, base_axes, capture)
    return Verdict(True, "significant accuracy gain within cost/latency bands", cmp_, cand_axes, base_axes, capture)
