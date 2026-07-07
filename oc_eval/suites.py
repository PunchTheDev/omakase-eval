"""Procedural task suites.

Every task is generated from a (split, seed) pair, so splits are reproducible
anywhere from two integers — no dataset downloads, nothing to leak. The dev
split is public (miners iterate on it); gate splits use rotation seeds the
maintainer publishes only after scoring.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

SUITES = ("reasoning", "math", "code_qa")
LETTERS = "ABCD"


@dataclass(frozen=True)
class Task:
    id: str
    suite: str
    prompt: str
    options: tuple[str, ...]  # empty for free-form numeric answers
    answer: str  # canonical answer content (option text or number)
    meta: dict = field(default_factory=dict, compare=False)


def _rng(*parts: object) -> random.Random:
    seed = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(seed[:8], "big"))


def _mcq(rng: random.Random, correct: int, distractors: list[int]) -> tuple[tuple[str, ...], str]:
    opts = [str(correct)] + [str(d) for d in distractors]
    rng.shuffle(opts)
    return tuple(opts), str(correct)


def _gen_reasoning(rng: random.Random, tid: str) -> Task:
    """Next term of an arithmetic-geometric sequence."""
    a, d, r = rng.randint(1, 9), rng.randint(2, 9), rng.choice((1, 1, 2))
    terms = [a]
    for _ in range(4):
        terms.append(terms[-1] * r + d)
    nxt = terms[-1] * r + d
    near = {nxt + d, nxt - d, terms[-1] + d, nxt + r} - {nxt}
    options, answer = _mcq(rng, nxt, sorted(near)[:3])
    prompt = f"What is the next term of the sequence {', '.join(map(str, terms))}?"
    return Task(tid, "reasoning", prompt, options, answer)


def _gen_math(rng: random.Random, tid: str) -> Task:
    """Two-step word arithmetic, free-form numeric answer."""
    x, y, k = rng.randint(12, 96), rng.randint(3, 11), rng.randint(2, 9)
    prompt = (
        f"A crate holds {x} parts. {y} crates arrive and {k} parts are removed "
        f"from the total. How many parts remain? Answer with a number only."
    )
    return Task(tid, "math", prompt, (), str(x * y - k))


def _gen_code_qa(rng: random.Random, tid: str) -> Task:
    """Predict the output of a tiny generated snippet (evaluated here, not by the model)."""
    n, m, s = rng.randint(2, 6), rng.randint(2, 5), rng.randint(1, 4)
    code = f"x = {n}\nfor i in range({m}):\n    x = x + i * {s}\nprint(x)"
    out = n + s * (m * (m - 1) // 2)
    options, answer = _mcq(rng, out, [out + s, out - s, out + m])
    prompt = f"What does this Python program print?\n\n{code}"
    return Task(tid, "code_qa", prompt, options, answer)


_GENERATORS = {"reasoning": _gen_reasoning, "math": _gen_math, "code_qa": _gen_code_qa}


def task_by_id(split: str, seed: int, tid: str) -> Task:
    """Reconstruct one task from its id — generation is a pure function of (split, seed, suite, i)."""
    prefix, suite, idx = tid.rsplit("-", 2)
    if prefix != split or suite not in _GENERATORS:
        raise ValueError(f"task id {tid!r} does not belong to split {split!r}")
    return _GENERATORS[suite](_rng(split, seed, suite, int(idx)), tid)


def generate_split(split: str, seed: int, per_suite: int = 40) -> list[Task]:
    tasks = []
    for suite in SUITES:
        for i in range(per_suite):
            tid = f"{split}-{suite}-{i:04d}"
            tasks.append(_GENERATORS[suite](_rng(split, seed, suite, i), tid))
    return tasks


def shuffled_options(task: Task, run_seed: int) -> list[tuple[str, str]]:
    """Per-run option order (letter, text). Defeats fixed-letter strategies."""
    pairs = list(task.options)
    _rng("shuffle", run_seed, task.id).shuffle(pairs)
    return list(zip(LETTERS, pairs))


def render_prompt(task: Task, run_seed: int) -> str:
    if not task.options:
        return task.prompt
    lines = [task.prompt, ""] + [f"{letter}. {text}" for letter, text in shuffled_options(task, run_seed)]
    return "\n".join(lines)


def grade(task: Task, response: str, run_seed: int) -> bool:
    """Content-based grading; a bare option letter is resolved through this run's shuffle."""
    tail = response.strip().splitlines()[-1] if response.strip() else ""
    if task.options and tail.strip().rstrip(".").upper() in LETTERS:
        letter = tail.strip().rstrip(".").upper()
        text = dict(shuffled_options(task, run_seed)).get(letter, "")
        return text == task.answer
    tokens = tail.replace(".", " ").replace(",", " ").split()
    return task.answer in tokens or tail.strip() == task.answer
