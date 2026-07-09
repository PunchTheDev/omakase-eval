"""The wall between miner code and the answers.

A Harness submission is arbitrary code. It must never be able to read the answer
key, the gate seed, the signing key, or its own score. Running it in-process
(the old adapter) gave it all four: `sys.modules["omakase_eval.suites"]` grades,
the seed rode along in call metadata, and the maintainer's key sat on disk in a
readable path. A regex ban on `os`/`socket` is not containment — reflection and
string-splitting walk around it.

So the harness runs in a child process that simply does not have those things:

    TRUSTED PARENT (this module)          UNTRUSTED CHILD (sandbox_child.py)
    ├─ tasks + answers, grading           ├─ sys.path = [sandbox root] only
    ├─ the gate seed                      ├─ no omakase_eval.suites → no answers
    ├─ signing key, API tokens            ├─ scrubbed env → no seed, no secrets
    ├─ real Pool, cost/token meters       ├─ rlimits: memory, cpu, no file writes
    └─ budget + wall-clock enforcement    └─ capabilities = 2 RPCs, nothing more

The child's only powers are `chat` (one pool call, which the parent meters and
budget-checks) and `route` (ask the opaque pinned router). Both return data.
A crash, a hang, a flood, or a budget blow-out forfeits exactly one task; the
child is replaced and the split continues.

`mode="docker"` adds an OS-level layer under the same protocol — no network
namespace, read-only mount, pid cap — for the untrusted-PR case where the
process boundary alone is not a security boundary you want to bet a key on.
Note the composition that makes this sound: even a child that escaped to the
filesystem cannot regenerate gate tasks, because the seed lives only here.
"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace

from . import suites
from .actions import Answer, Call
from .engine import Budget, Step, TaskResult
from .workers import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHILD_SRC = os.path.join(_HERE, "sandbox_child.py")
# The only omakase_eval modules a harness may import. Both are pure constants /
# dataclasses with no path back to answers, generators, or grading.
_SDK_MODULES = ("templates.py", "actions.py")
_SDK_INIT = '"""Sandbox SDK: the only omakase_eval surface a harness may import."""\n'


class SandboxError(RuntimeError):
    """The sandbox itself failed (spawn, protocol, contract) — not a task forfeit."""


class _Forfeit(Exception):
    """This task is lost (timeout, crash, flood). The child is replaced."""


@dataclass(frozen=True)
class SandboxConfig:
    mode: str = "process"  # "process" | "docker"
    per_task_timeout_s: float = 60.0
    startup_timeout_s: float = 30.0
    max_rpc_per_task: int = 64
    mem_bytes: int = 1 << 30
    cpu_seconds: int = 600
    max_line_bytes: int = 8 << 20
    image: str = "python:3.12-slim"


def _redacted(task: suites.Task) -> suites.Task:
    return replace(task, answer="")


class _Channel:
    """Line-framed JSON over the child's pipes, with a hard deadline on reads."""

    def __init__(self, proc: subprocess.Popen, max_line_bytes: int) -> None:
        self.proc = proc
        self.buf = b""
        self.max_line_bytes = max_line_bytes
        self.fd = proc.stdout.fileno()
        os.set_blocking(self.fd, False)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.fd, selectors.EVENT_READ)

    def send(self, obj: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise _Forfeit(f"child gone: {exc}") from exc

    def recv(self, deadline: float) -> dict:
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.sel.select(remaining):
                raise _Forfeit("timeout")
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise _Forfeit("child exited")
            self.buf += chunk
            if len(self.buf) > self.max_line_bytes:
                raise _Forfeit("child flooded the channel")
        line, _, self.buf = self.buf.partition(b"\n")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise _Forfeit(f"malformed frame: {exc}") from exc

    def close(self) -> None:
        self.sel.close()


class _Step:
    """The slice of a Step a router policy may read, rebuilt from the child's claim."""

    __slots__ = ("response",)

    def __init__(self, response: str) -> None:
        self.response = response


def _rlimits(cfg: SandboxConfig):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (cfg.mem_bytes, cfg.mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # cannot write any regular file
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


class HarnessSandbox:
    """Owns the sandbox root and the current child process."""

    def __init__(self, harness_dir: str, cfg: SandboxConfig | None = None) -> None:
        self.harness_dir = os.path.abspath(harness_dir)
        self.cfg = cfg or SandboxConfig()
        if self.cfg.mode not in ("process", "docker"):
            raise SandboxError(f"unknown sandbox mode {self.cfg.mode!r}")
        self.root: str | None = None
        self.proc: subprocess.Popen | None = None
        self.chan: _Channel | None = None
        self.forfeits: list[str] = []

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "HarnessSandbox":
        self.root = tempfile.mkdtemp(prefix="omakase-sandbox-")
        self._build_root()
        self._spawn()
        return self

    def __exit__(self, *exc) -> None:
        self._kill()
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)

    def _build_root(self) -> None:
        assert self.root
        shutil.copytree(self.harness_dir, os.path.join(self.root, "harness"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sdk = os.path.join(self.root, "omakase_eval")
        os.makedirs(sdk)
        with open(os.path.join(sdk, "__init__.py"), "w") as f:
            f.write(_SDK_INIT)
        for mod in _SDK_MODULES:
            shutil.copy(os.path.join(_HERE, mod), os.path.join(sdk, mod))
        shutil.copy(_CHILD_SRC, os.path.join(self.root, "_child.py"))
        # docker mode runs the child as `nobody`, which cannot traverse mkdtemp's
        # 0700 root. Everything in here is the miner's own code plus the public SDK
        # — there is nothing secret to protect with these bits.
        if self.cfg.mode == "docker":
            self._make_world_readable(self.root)

    @staticmethod
    def _make_world_readable(root: str) -> None:
        os.chmod(root, 0o755)
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames:
                os.chmod(os.path.join(dirpath, name), 0o755)
            for name in filenames:
                os.chmod(os.path.join(dirpath, name), 0o644)

    def _env(self) -> dict:
        """Scrubbed: no seed, no signing key, no API tokens — nothing to exfiltrate."""
        return {
            "PATH": "/usr/bin:/bin",
            "HOME": self.root or "/tmp",  # noqa: S108 — sandbox root, not a real home
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

    def _docker_argv(self) -> list[str]:
        return [
            "docker", "run", "--rm", "-i",
            "--network=none",                     # no exfiltration path
            "--read-only",                        # no writes anywhere
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=64",                    # no fork bombs
            f"--memory={self.cfg.mem_bytes}", "--memory-swap=-1",
            "--user=65534:65534",                 # nobody
            "-v", f"{self.root}:/sandbox:ro", "-w", "/sandbox",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0",
            self.cfg.image, "python", "-I", "-S", "-B", "_child.py",
        ]

    def _spawn(self) -> None:
        assert self.root
        if self.cfg.mode == "docker":
            argv, kwargs = self._docker_argv(), {}
        else:
            # -I: isolated (ignores PYTHONPATH/user site, sys.path[0] = sandbox root)
            # -S: no site-packages, so the *real* omakase_eval (pip-installed) is
            #     unreachable — the stub SDK is the only one importable.
            argv = [sys.executable, "-I", "-S", "-B", "_child.py"]
            kwargs = {"preexec_fn": _rlimits(self.cfg), "start_new_session": True}
        self.proc = subprocess.Popen(  # noqa: S603
            argv, cwd=self.root, env=self._env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            close_fds=True, **kwargs,
        )
        self.chan = _Channel(self.proc, self.cfg.max_line_bytes)
        try:
            hello = self.chan.recv(time.monotonic() + self.cfg.startup_timeout_s)
        except _Forfeit as exc:
            raise SandboxError(f"harness failed to start: {exc}") from exc
        if "fatal" in hello:  # `import harness` blew up — a broken contract, not a forfeit
            raise SandboxError(f"contract-broken: {hello['fatal']}")
        if not hello.get("ready"):
            raise SandboxError(f"unexpected handshake: {hello}")

    def _kill(self) -> None:
        if self.chan:
            self.chan.close()
            self.chan = None
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def _restart(self) -> None:
        self._kill()
        self._spawn()

    # -- the eval loop -------------------------------------------------------
    def run_task(self, router, task: suites.Task, pool: Pool, seed: int, split: str,
                 budget: Budget) -> TaskResult:
        """Score one task. The child never sees `seed`, the answer, or its verdict."""
        result = TaskResult(task.id, task.suite, correct=False)
        prompt = suites.render_prompt(task, seed)
        metadata = {"split": split, "seed": seed, "task_id": task.id}  # parent-side only
        deadline = time.monotonic() + self.cfg.per_task_timeout_s
        assert self.chan

        try:
            self.chan.send({
                "cmd": "task", "id": task.id, "suite": task.suite, "prompt": prompt,
                "budget": {"max_turns": budget.max_turns, "max_tokens": budget.max_tokens},
                "workers": {w: {"id": w, "cost_per_1k": pool.workers[w].cost_per_1k} for w in pool.workers},
            })
            calls = 0
            while True:
                msg = self.chan.recv(deadline)
                if "done" in msg:
                    result.answer = str(msg["done"] or "")
                    break
                calls += 1
                if calls > self.cfg.max_rpc_per_task:
                    raise _Forfeit("rpc flood")
                self._serve(msg, router, task, pool, prompt, metadata, budget, result)
        except _Forfeit as exc:
            self.forfeits.append(f"{task.id}: {exc}")
            result.answer = ""
            self._restart()

        # Grading is central and happens only here. The child is never told.
        result.correct = bool(result.answer) and suites.grade(task, result.answer, seed)
        return result

    def _serve(self, msg: dict, router, task: suites.Task, pool: Pool, prompt: str,
               metadata: dict, budget: Budget, result: TaskResult) -> None:
        assert self.chan
        op = msg.get("call")
        if op == "chat":
            # Budget is enforced here, on measured truth — never on the child's word.
            if len(result.steps) >= budget.max_turns or result.tokens >= budget.max_tokens:
                self.chan.send({"error": "budget"})
                return
            worker = msg.get("worker")
            if worker not in pool.workers:
                self.chan.send({"error": f"unknown worker {worker!r}"})
                return
            completion = pool.chat(worker, str(msg.get("system", "")), str(msg.get("user", "")),
                                   metadata=metadata)
            result.steps.append(Step(Call(worker), completion.text, completion.tokens))
            result.tokens += completion.tokens
            result.cost += completion.cost
            result.latency_ms += completion.latency_ms
            self.chan.send({"text": completion.text, "tokens": completion.tokens})
        elif op == "route":
            steps = [_Step(str(s.get("response", ""))) for s in msg.get("steps", [])]
            action = router.decide(task=_redacted(task), prompt=str(msg.get("prompt", prompt)), steps=steps)
            if isinstance(action, Call):
                self.chan.send({"action": "call", "worker": action.worker, "role": action.role})
            elif isinstance(action, Answer):
                self.chan.send({"action": "answer", "final": action.final})
            else:
                self.chan.send({"error": "router returned an unknown action"})
        else:
            self.chan.send({"error": f"unknown op {op!r}"})


def run_harness_split(harness_dir: str, router, tasks: list[suites.Task], pool: Pool,
                      seed: int, split: str, budget: Budget | None = None,
                      cfg: SandboxConfig | None = None) -> tuple[list[TaskResult], list[str]]:
    """Run every task through the sandboxed harness. Returns (results, forfeits)."""
    budget = budget or Budget()
    with HarnessSandbox(harness_dir, cfg) as sbx:
        results = [sbx.run_task(router, t, pool, seed, split, budget) for t in tasks]
        return results, list(sbx.forfeits)
