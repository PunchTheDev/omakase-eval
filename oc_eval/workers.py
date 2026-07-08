"""Worker pool client — OpenAI-compatible chat completions over stdlib HTTP.

The pool config pins every worker (model id, endpoint, cost class). In dev the
endpoints point at the deterministic mock pool; in production they point at the
pinned vLLM cluster and Gate 4's egress allow-list is exactly this set.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    id: str
    model: str
    base_url: str
    cost_per_1k: float  # relative cost units per 1k tokens
    api_key_env: str = ""  # env var holding the Bearer token (OpenRouter/vLLM); empty = no auth (mock)


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int
    latency_ms: float
    cost: float


class Pool:
    def __init__(self, workers: list[Worker], timeout_s: float = 60.0):
        self.workers = {w.id: w for w in workers}
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, path: str) -> "Pool":
        with open(path) as f:
            cfg = json.load(f)
        return cls([Worker(**w) for w in cfg["workers"]], timeout_s=cfg.get("timeout_s", 60.0))

    def chat(self, worker_id: str, system: str, user: str, metadata: dict | None = None) -> Completion:
        w = self.workers[worker_id]
        body = {
            "model": w.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }
        if metadata:
            body["metadata"] = metadata  # dev-only routing hints; real servers ignore extra fields
        headers = {"Content-Type": "application/json"}
        if w.api_key_env:  # real provider (OpenRouter, authed vLLM); key from env, never config
            headers["Authorization"] = f"Bearer {os.environ[w.api_key_env]}"
        req = urllib.request.Request(
            f"{w.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.load(resp)
        latency_ms = (time.monotonic() - t0) * 1000
        text = payload["choices"][0]["message"]["content"]
        tokens = payload.get("usage", {}).get("total_tokens") or max(1, len(text.split()))
        return Completion(text, tokens, latency_ms, tokens / 1000 * w.cost_per_1k)
