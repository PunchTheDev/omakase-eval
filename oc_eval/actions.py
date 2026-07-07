"""Action schema — the fixed contract between a router and the harness.

A router never touches workers directly; it emits actions and the engine
executes them. Keeping this surface minimal is what makes OC-R a policy
competition rather than a systems one.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLES = ("thinker", "worker", "verifier")


@dataclass(frozen=True)
class Call:
    """Ask one pool worker to respond to the task under a role."""

    worker: str
    role: str = "worker"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")


@dataclass(frozen=True)
class Answer:
    """Terminate the task with a final answer."""

    final: str


Action = Call | Answer
