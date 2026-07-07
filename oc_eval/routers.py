"""Router implementations and the manifest loader.

TinyRouter is the competition's seed architecture: hashed bag-of-words features
into a linear policy over workers (~10K parameters, tinyrouter-scale). OC-R
submissions may use any architecture that satisfies the manifest contract; this
one exists so the competition launches with a real, beatable champion.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

from .actions import Answer, Call

DIM = 1024


def features(text: str) -> dict[int, float]:
    """Sparse hashed bag-of-words with an always-on bias at index 0."""
    idx: dict[int, float] = {0: 1.0}
    for word in text.lower().split():
        h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        i = 1 + h % (DIM - 1)
        idx[i] = idx.get(i, 0.0) + 1.0
    return idx


@dataclass
class SingleWorkerRouter:
    """Baseline: always route to one worker. Solo runs of these define best-single and oracle."""

    worker: str

    def decide(self, task, prompt, steps):
        if steps:
            return Answer(steps[-1].response)
        return Call(self.worker)


class TinyRouter:
    """Linear policy: argmax_w W[w]·φ(prompt) → one CALL, then ANSWER."""

    def __init__(self, workers: list[str], weights: list[list[float]]):
        if len(weights) != len(workers) or any(len(row) != DIM for row in weights):
            raise ValueError("weights shape must be [n_workers][DIM]")
        self.workers = workers
        self.weights = weights

    def decide(self, task, prompt, steps):
        if steps:
            return Answer(steps[-1].response)
        phi = features(prompt)
        scores = [sum(row[i] * v for i, v in phi.items()) for row in self.weights]
        return Call(self.workers[scores.index(max(scores))])

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        blob = {"arch": "tiny-linear", "dim": DIM, "workers": self.workers,
                "weights": [[round(x, 6) for x in row] for row in self.weights]}
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "TinyRouter":
        with open(path) as f:
            blob = json.load(f)
        if blob.get("arch") != "tiny-linear" or blob.get("dim") != DIM:
            raise ValueError("unsupported weights blob")
        return cls(blob["workers"], blob["weights"])


def fit_tiny_router(prompts: list[str], labels: list[int], workers: list[str],
                    epochs: int = 12, lr: float = 0.5, seed: int = 0) -> TinyRouter:
    """Averaged multiclass perceptron over hashed features. Pure stdlib, seconds to train."""
    rng = random.Random(seed)
    w = [[0.0] * DIM for _ in workers]
    acc = [[0.0] * DIM for _ in workers]
    order = list(range(len(prompts)))
    for _ in range(epochs):
        rng.shuffle(order)
        for j in order:
            phi = features(prompts[j])
            scores = [sum(row[i] * v for i, v in phi.items()) for row in w]
            pred = scores.index(max(scores))
            if pred != labels[j]:
                for i, v in phi.items():
                    w[labels[j]][i] += lr * v
                    w[pred][i] -= lr * v
        for k in range(len(workers)):
            for i in range(DIM):
                acc[k][i] += w[k][i]
    norm = 1.0 / (epochs * max(1, len(prompts)) ** 0.5)
    return TinyRouter(workers, [[x * norm for x in row] for row in acc])


# -- manifest loading (the OC-R submission contract) ------------------------

MAX_WEIGHTS_BYTES = 25_000_000  # tiny class: generous for JSON blobs, tiny for models


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_router(manifest_path: str, weights_dir: str) -> TinyRouter:
    """Load a submission: verify declared sha256 and size cap, then construct the router."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["arch"] != "tiny-linear":
        raise ValueError(f"unknown arch {manifest['arch']!r}")
    path = f"{weights_dir}/{manifest['weights_file']}"
    import os

    if os.path.getsize(path) > MAX_WEIGHTS_BYTES:
        raise ValueError("weights exceed the tiny-class size cap")
    digest = sha256_file(path)
    if digest != manifest["weights_sha256"]:
        raise ValueError(f"weights sha256 mismatch: manifest {manifest['weights_sha256'][:12]}…, file {digest[:12]}…")
    return TinyRouter.load(path)


def perplexity_check(router: TinyRouter) -> float:
    """Cheap Gate-3 sanity signal: weight-mass entropy (a lookup table concentrates)."""
    mass = [sum(abs(x) for x in row) for row in router.weights]
    total = sum(mass) or 1.0
    return -sum(m / total * math.log(m / total + 1e-12) for m in mass)
