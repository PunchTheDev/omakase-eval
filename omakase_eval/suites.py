"""Procedural task suites — the un-gameable backbone of the gate.

Every task is a pure function of (split, seed, suite, index), so a whole split
is reproducible from two integers — nothing to download, nothing to leak, and
fresh instances every round by bumping the seed. The **dev** split is public
(miners self-score); **gate** splits use private rotation seeds Punch reveals
only after scoring — the Kaggle private-holdout pattern.

Generators aim high on difficulty (best single worker ≈ 55–75%, the band that
maximizes routing headroom). Deep-knowledge coverage (grad STEM, expert
domains) comes from the JSONL holdout tier — see datasets.py — since real
knowledge can't be procedurally generated. These procedural families cover
quantitative + symbolic + code reasoning.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from math import gcd

SUITES = ("reasoning", "math", "code_qa", "logic", "number_theory")
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


def _mcq(rng: random.Random, correct, distractors) -> tuple[tuple[str, ...], str]:
    seen, opts = {str(correct)}, [str(correct)]
    for d in distractors:
        if str(d) not in seen:
            seen.add(str(d))
            opts.append(str(d))
    opts = opts[:4]
    rng.shuffle(opts)
    return tuple(opts), str(correct)


def _gen_reasoning(rng: random.Random, tid: str) -> Task:
    """Second-order sequence: aₙ = aₙ₋₁·r + d·n. Hard to eyeball."""
    a, d, r = rng.randint(2, 7), rng.randint(2, 6), rng.choice((2, 2, 3))
    terms = [a]
    for n in range(1, 5):
        terms.append(terms[-1] * r + d * n)
    nxt = terms[-1] * r + d * 5
    near = [nxt + d, nxt - r, terms[-1] * r, nxt + r * d]
    options, answer = _mcq(rng, nxt, near)
    return Task(tid, "reasoning", f"Find the next term: {', '.join(map(str, terms))}, ?", options, answer)


def _gen_math(rng: random.Random, tid: str) -> Task:
    """Four-step quantitative word problem, numeric answer."""
    crates, per, add, rem = rng.randint(6, 19), rng.randint(11, 29), rng.randint(3, 12), rng.randint(5, 40)
    rate = rng.randint(2, 5)
    total = (crates + add) * per - rem * rate
    prompt = (
        f"A depot has {crates} crates of {per} units each. {add} more crates arrive. "
        f"Then {rem} orders ship, each taking {rate} units. How many units remain? "
        f"Answer with a number only."
    )
    return Task(tid, "math", prompt, (), str(total))


def _gen_code_qa(rng: random.Random, tid: str) -> Task:
    """Trace a nested loop with a conditional — no execution, verified here."""
    n, m, k = rng.randint(2, 5), rng.randint(3, 6), rng.randint(2, 4)
    x = n
    for i in range(m):
        for j in range(k):
            x = x + i - j if (i + j) % 2 == 0 else x + 1
    code = (f"x = {n}\nfor i in range({m}):\n    for j in range({k}):\n"
            f"        x = x + i - j if (i + j) % 2 == 0 else x + 1\nprint(x)")
    options, answer = _mcq(rng, x, [x + 1, x - 2, x + k])
    return Task(tid, "code_qa", f"What does this print?\n\n{code}", options, answer)


_NAMES = ("Ava", "Ben", "Cora", "Dan", "Eve")


def _gen_logic(rng: random.Random, tid: str) -> Task:
    """Total-order constraint puzzle with a unique answer."""
    people = _NAMES[: rng.randint(4, 5)]
    order = list(people)
    rng.shuffle(order)  # order[0] tallest … order[-1] shortest
    facts = [f"{order[i]} is taller than {order[i+1]}" for i in range(len(order) - 1)]
    rng.shuffle(facts)
    ask_short = rng.random() < 0.5
    answer = order[-1] if ask_short else order[0]
    who = "shortest" if ask_short else "tallest"
    options, answer = _mcq(rng, answer, [p for p in people if p != answer])
    return Task(tid, "logic", f"{'. '.join(facts)}. Who is {who}?", options, answer)


def _gen_number_theory(rng: random.Random, tid: str) -> Task:
    """gcd / lcm / modular exponentiation — numeric answer."""
    kind = rng.choice(("lcm", "modpow", "gcd"))
    if kind == "gcd":
        a, b = rng.randint(48, 480), rng.randint(48, 480)
        return Task(tid, "number_theory", f"What is gcd({a}, {b})? Answer with a number only.", (), str(gcd(a, b)))
    if kind == "lcm":
        a, b = rng.randint(6, 40), rng.randint(6, 40)
        return Task(tid, "number_theory", f"What is lcm({a}, {b})? Answer with a number only.", (), str(a * b // gcd(a, b)))
    base, exp, mod = rng.randint(2, 9), rng.randint(5, 20), rng.randint(7, 97)
    return Task(tid, "number_theory",
                f"Compute {base}^{exp} mod {mod}. Answer with a number only.", (), str(pow(base, exp, mod)))


_GENERATORS = {
    "reasoning": _gen_reasoning, "math": _gen_math, "code_qa": _gen_code_qa,
    "logic": _gen_logic, "number_theory": _gen_number_theory,
}


def task_by_id(split: str, seed: int, tid: str) -> Task:
    """Reconstruct one task from its id — pure function of (split, seed, suite, i)."""
    prefix, suite, idx = tid.rsplit("-", 2)
    if suite not in _GENERATORS:
        raise ValueError(f"task id {tid!r}: unknown suite {suite!r}")
    return _GENERATORS[suite](_rng(split, seed, suite, int(idx)), tid)


def generate_split(split: str, seed: int, per_suite: int = 40, suites: tuple[str, ...] = SUITES) -> list[Task]:
    tasks = []
    for suite in suites:
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
        return dict(shuffled_options(task, run_seed)).get(letter, "") == task.answer
    tokens = tail.replace(".", " ").replace(",", " ").split()
    return task.answer in tokens or tail.strip() == task.answer
