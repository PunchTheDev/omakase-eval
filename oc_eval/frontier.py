"""Append-only, hash-chained results ledger (the frontier log).

Each entry commits to its predecessor, so history is tamper-evident: rewriting
any past entry breaks every hash after it. GitHub commits give timestamps; the
chain gives integrity. The dashboard is a projection of this file.
"""
from __future__ import annotations

import hashlib
import json
import time

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "sha"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def append(path: str, kind: str, payload: dict, ts: float | None = None) -> dict:
    entries = read(path)
    entry = {
        "seq": len(entries),
        "prev": entries[-1]["sha"] if entries else GENESIS,
        "ts": round(ts if ts is not None else time.time(), 3),
        "kind": kind,
        "payload": payload,
    }
    entry["sha"] = _digest(entry)
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


def verify(path: str) -> tuple[bool, str]:
    prev = GENESIS
    for i, entry in enumerate(read(path)):
        if entry["seq"] != i or entry["prev"] != prev or entry["sha"] != _digest(entry):
            return False, f"chain broken at seq {i}"
        prev = entry["sha"]
    return True, "ok"
