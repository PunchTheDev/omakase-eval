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
COST_TOLERANCE = 1.10  # candidate worker-token cost may rise up to +10% vs. incumbent


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


def paired_correctness(candidate: list[TaskResult], baseline: list[TaskResult]) -> tuple[list[bool], list[bool]]:
    """Align candidate and baseline by task_id before pairing.

    McNemar + the paired bootstrap are only meaningful when position i of both
    vectors is the *same* task. Sorting each side independently (as
    `correctness_vector` does) silently pairs disjoint ids when the two runs
    were scored on different splits/seeds — e.g. a rotated gate split vs. the dev
    baseline. Refuse that instead of returning a garbage verdict: pair on the
    shared id set and demand the sets match.
    """
    cand = {r.task_id: r.correct for r in candidate}
    base = {r.task_id: r.correct for r in baseline}
    if cand.keys() != base.keys():
        unmatched = sorted(cand.keys() ^ base.keys())
        raise ValueError(
            f"paired comparison requires identical task ids on both sides; "
            f"{len(unmatched)} unmatched (e.g. {unmatched[:3]}). Candidate and "
            "baseline must be scored on the same split+seed — refusing to zip "
            "disjoint vectors positionally."
        )
    ids = sorted(cand)
    return [cand[i] for i in ids], [base[i] for i in ids]


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
          oracle_accuracy: float | None = None, gate_cost: bool = True) -> Verdict:
    """Pass iff accuracy improves with significance and cost stays in band.

    `gate_cost` compares candidate cost vs. the incumbent — fair when the
    incumbent is itself a router (a reigning champion). At genesis the incumbent
    is a *single* worker (the accuracy floor), so cost-gating would punish
    routing to specialists for accuracy — the whole point of the pool. Punch
    sets gate_cost=False there and the first champion is judged on accuracy
    alone. Latency is reported but never gated (transport jitter on a served pool).
    """
    cand_axes, base_axes = axes(candidate), axes(baseline)
    cand_vec, base_vec = paired_correctness(candidate, baseline)
    cmp_ = stats.compare(cand_vec, base_vec)

    capture = None
    if oracle_accuracy is not None and oracle_accuracy > base_axes.accuracy:
        capture = (cand_axes.accuracy - base_axes.accuracy) / (oracle_accuracy - base_axes.accuracy)

    if not cmp_.significant:
        return Verdict(False, "accuracy gain not significant", cmp_, cand_axes, base_axes, capture)
    if gate_cost and cand_axes.cost_per_task > base_axes.cost_per_task * COST_TOLERANCE:
        return Verdict(False, "cost regression beyond tolerance", cmp_, cand_axes, base_axes, capture)
    return Verdict(True, "significant accuracy gain within cost band", cmp_, cand_axes, base_axes, capture)
