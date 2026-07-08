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


def _numstr(x: float | int) -> str:
    """Language-neutral number form so a JS reader reproduces the digest exactly.

    Integer-valued numbers drop the decimal; others use fixed 12-decimal
    notation (no sci, no int/float ambiguity). Mirrors `numstr` in the
    dashboard's data.ts — keep the two in lockstep.
    """
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return format(x, ".12f").rstrip("0").rstrip(".")


def _canonical(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _numstr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(_canonical(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{json.dumps(k)}:{_canonical(v[k])}" for k in sorted(v)) + "}"
    raise TypeError(f"uncanonicalizable {type(v)}")


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "sha"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


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
