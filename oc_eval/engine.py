"""The reference harness engine: a bounded turn loop that executes router actions.

This file is locked in OC-R (it is the scoring function) and is the seed that
OC-H miners evolve. Budgets are enforced here, not trusted to the router.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import suites, templates
from .actions import Answer, Call
from .workers import Pool


@dataclass(frozen=True)
class Budget:
    max_turns: int = 5
    max_tokens: int = 4000


@dataclass
class Step:
    call: Call
    response: str
    tokens: int


@dataclass
class TaskResult:
    task_id: str
    suite: str
    correct: bool
    tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    steps: list[Step] = field(default_factory=list)
    answer: str = ""


def run_task(router, task: suites.Task, pool: Pool, run_seed: int, split: str, budget: Budget = Budget()) -> TaskResult:
    result = TaskResult(task.id, task.suite, correct=False)
    prompt = suites.render_prompt(task, run_seed)
    metadata = {"split": split, "seed": run_seed, "task_id": task.id}
    draft: str | None = None

    for _ in range(budget.max_turns):
        action = router.decide(task=task, prompt=prompt, steps=result.steps)
        if isinstance(action, Answer):
            result.answer = action.final
            break
        if not isinstance(action, Call):
            break  # malformed router output forfeits the task
        if result.tokens >= budget.max_tokens:
            break  # budget exhausted before the router answered
        completion = pool.chat(
            action.worker,
            system=templates.SYSTEM[action.role],
            user=templates.user_message(action.role, prompt, draft),
            metadata=metadata,
        )
        result.steps.append(Step(action, completion.text, completion.tokens))
        result.tokens += completion.tokens
        result.cost += completion.cost
        result.latency_ms += completion.latency_ms
        if action.role != "verifier":
            draft = completion.text

    result.correct = bool(result.answer) and suites.grade(task, result.answer, run_seed)
    return result


def run_split(router, tasks: list[suites.Task], pool: Pool, run_seed: int, split: str, budget: Budget = Budget()) -> list[TaskResult]:
    return [run_task(router, t, pool, run_seed, split, budget) for t in tasks]
