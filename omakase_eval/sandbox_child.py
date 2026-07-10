"""Untrusted child bootstrap — runs the miner's harness with nothing to steal.

This file is COPIED into a throwaway sandbox root and executed there as
`_child.py` (never imported by the parent). Inside this process:

- `sys.path` is exactly the sandbox root. The real `omakase_eval` (answers,
  generators, grading) is not on it, so `import omakase_eval.suites` raises
  ModuleNotFoundError — there is no object graph to reflect your way into. The
  only `omakase_eval` importable is the stub SDK the parent wrote: `templates`
  and `actions`, which contain no answers.
- The gate seed is never sent here. Even with filesystem access, tasks are not
  reconstructible without it.
- The environment is scrubbed: no signing key, no API tokens, no seed.
- stdout/stdin are re-pointed at /dev/null before the harness is imported, so a
  `print()` in miner code cannot corrupt or spoof the RPC channel; the real
  channel is held on private duplicated fds.

The harness's only capability is asking the parent to do things: `chat` (a pool
call, metered and budget-enforced there) and `route` (the opaque pinned router).
Everything the parent returns is data, never behavior.
"""
from __future__ import annotations

import json
import os
import sys

# The sandbox root (this file's directory) is the only *project* import source.
# `-I -S` already keep PYTHONPATH and site-packages out — which is what makes the
# real `omakase_eval` (pip-installed, answers and all) unreachable. Re-filter here
# so a future launcher flag can't silently widen the path. The stdlib entries must
# stay: strip them and `import dataclasses` dies inside the SDK.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [_ROOT] + [
    p for p in sys.path
    if p and p != _ROOT and "site-packages" not in p and "dist-packages" not in p
]


def _private_channel():
    """Duplicate the real pipes away, then blind fd 0/1 with /dev/null.

    After this the harness can print() and read() freely: it hits /dev/null, so
    ordinary miner code cannot accidentally corrupt the protocol. This is a
    hygiene measure, NOT a security boundary: miner code runs in this same
    process, so it can still reach the dup'd fds (via `__main__` globals or
    `/proc/self/fd`) and inject frames deliberately. That is tolerable because
    the parent treats every frame as hostile — a spoofed `{"done": …}` still
    can't guess the centrally-graded answer, and any malformed frame forfeits
    one task instead of crashing the run (see `_Channel.recv`). Real containment
    is the process/OS boundary, not fd hiding.
    """
    rfd, wfd = os.dup(0), os.dup(1)
    devnull_r = os.open(os.devnull, os.O_RDONLY)
    devnull_w = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_r, 0)
    os.dup2(devnull_w, 1)
    os.close(devnull_r)
    os.close(devnull_w)
    rx = os.fdopen(rfd, "r", buffering=1)
    tx = os.fdopen(wfd, "w", buffering=1)
    sys.stdin = open(os.devnull)  # noqa: SIM115 — lives for the process
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    return rx, tx


_RX, _TX = _private_channel()


def _send(obj: dict) -> None:
    _TX.write(json.dumps(obj, separators=(",", ":")) + "\n")
    _TX.flush()


def _recv() -> dict | None:
    line = _RX.readline()
    return json.loads(line) if line else None


class BudgetExceeded(Exception):
    """Raised when the parent refuses a call: turns or tokens are spent."""


def _rpc(request: dict) -> dict:
    _send(request)
    reply = _recv()
    if reply is None:  # parent hung up (timeout kill) — unwind quietly
        raise SystemExit(0)
    if reply.get("error") == "budget":
        raise BudgetExceeded
    if "error" in reply:
        raise RuntimeError(str(reply["error"]))
    return reply


# -- the surface the harness contract promises -------------------------------
# Shapes mirror the in-process objects exactly, so miner code is unchanged.

class TaskView:
    """What the harness is allowed to know about a task: no answer, no seed."""

    __slots__ = ("id", "suite", "prompt")

    def __init__(self, id: str, suite: str, prompt: str) -> None:  # noqa: A002
        self.id, self.suite, self.prompt = id, suite, prompt


class Budget:
    __slots__ = ("max_turns", "max_tokens")

    def __init__(self, max_turns: int, max_tokens: int) -> None:
        self.max_turns, self.max_tokens = max_turns, max_tokens


class Completion:
    __slots__ = ("text", "tokens")

    def __init__(self, text: str, tokens: int) -> None:
        self.text, self.tokens = text, tokens


class PoolProxy:
    """`pool.chat(...)` is an RPC. Metering and budgets are the parent's job."""

    def __init__(self, workers: dict) -> None:
        self.workers = workers

    def chat(self, worker: str, system: str, user: str) -> Completion:
        reply = _rpc({"call": "chat", "worker": worker, "system": system, "user": user})
        return Completion(reply["text"], reply["tokens"])


class RouterProxy:
    """The pinned router stays in the parent — here it is an opaque oracle."""

    def decide(self, task=None, prompt: str = "", steps=()):  # noqa: ANN001
        from omakase_eval.actions import Answer, Call

        reply = _rpc({
            "call": "route",
            "prompt": str(prompt),
            # only `.response` of each step is load-bearing for a router policy
            "steps": [{"response": getattr(s, "response", "")} for s in steps],
        })
        if reply.get("action") == "call":
            return Call(reply["worker"], reply.get("role", "worker"))
        return Answer(reply.get("final", ""))


def main() -> int:
    try:
        import harness  # the mutable miner artifact
        run_task = harness.run_task
    except Exception as exc:  # noqa: BLE001 — a broken contract must report, not crash silently
        _send({"fatal": f"{type(exc).__name__}: {exc}"})
        return 1
    _send({"ready": True})

    router = RouterProxy()
    while True:
        cmd = _recv()
        if cmd is None or cmd.get("cmd") == "stop":
            return 0
        if cmd.get("cmd") != "task":
            _send({"done": ""})
            continue
        view = TaskView(cmd["id"], cmd["suite"], cmd["prompt"])
        budget = Budget(cmd["budget"]["max_turns"], cmd["budget"]["max_tokens"])
        pool = PoolProxy(cmd.get("workers", {}))
        try:
            answer = run_task(router, view, pool, budget)
        except BudgetExceeded:
            answer = ""  # budget blown = forfeited task
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 — a crashing harness forfeits one task, never the eval
            _send({"done": "", "crash": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        _send({"done": "" if answer is None else str(answer)})


if __name__ == "__main__":
    raise SystemExit(main())
