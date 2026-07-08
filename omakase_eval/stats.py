"""Paired statistics for verdicts.

Candidate and baseline run the identical instances with identical seeds, so we
compare paired correctness vectors — McNemar's exact test plus a paired
bootstrap CI. Pairing is the single highest-leverage statistical choice in the
system: it cuts the minimum detectable effect ~2-3x vs. independent runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def mcnemar_exact(candidate: list[bool], baseline: list[bool]) -> tuple[float, int, int]:
    """Two-sided exact McNemar. Returns (p, wins, losses) over discordant pairs."""
    if len(candidate) != len(baseline):
        raise ValueError("paired vectors must be the same length")
    wins = sum(1 for c, b in zip(candidate, baseline) if c and not b)
    losses = sum(1 for c, b in zip(candidate, baseline) if b and not c)
    n = wins + losses
    if n == 0:
        return 1.0, 0, 0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail), wins, losses


def paired_bootstrap_ci(candidate: list[bool], baseline: list[bool], iters: int = 2000,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI on mean(candidate) - mean(baseline), resampling task indices."""
    rng = random.Random(seed)
    n = len(candidate)
    diffs = sorted(
        sum(candidate[j] - baseline[j] for j in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(iters)
    )
    lo, hi = int(iters * alpha / 2), int(iters * (1 - alpha / 2)) - 1
    return diffs[lo], diffs[hi]


def minimum_detectable_effect(n: int, discordant_rate: float = 0.25,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate accuracy-delta detectable by paired McNemar at (alpha, power).

    Normal approximation on discordant pairs: delta ≈ (z_a + z_b) * sqrt(d/n),
    where d is the expected discordant fraction. Published on the dashboard so
    miners know the bar before they submit.
    """
    z = {0.05: 1.96, 0.1: 1.645}[alpha], {0.8: 0.84, 0.9: 1.282}[power]
    return (z[0] + z[1]) * math.sqrt(discordant_rate / n)


@dataclass(frozen=True)
class Comparison:
    p_value: float
    wins: int
    losses: int
    delta: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.delta > 0


def compare(candidate: list[bool], baseline: list[bool]) -> Comparison:
    p, wins, losses = mcnemar_exact(candidate, baseline)
    delta = (sum(candidate) - sum(baseline)) / len(candidate)
    lo, hi = paired_bootstrap_ci(candidate, baseline)
    return Comparison(p, wins, losses, delta, lo, hi)
