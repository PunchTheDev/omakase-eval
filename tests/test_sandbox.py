"""The sandbox's job is to make the old exploits impossible. These tests are those exploits.

Each malicious harness below is a real attack that scored a merge before the
process boundary existed. They must now all forfeit (answer "" → graded wrong),
never crash the eval, and never reveal an answer, the seed, or a secret.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from omakase_eval import baselines as bl
from omakase_eval import sandbox, suites
from omakase_eval.engine import Budget
from omakase_eval.routers import SingleWorkerRouter
from omakase_eval.workers import Pool, Worker


class FakePool(Pool):
    """A pool that answers with a canned string — no network, no mock server."""

    def __init__(self, text: str = "42"):
        super().__init__([Worker("w1", "w1", "http://127.0.0.1:0", 1.0),
                          Worker("w2", "w2", "http://127.0.0.1:0", 1.0)], version="pool@test")
        self.text = text
        self.calls: list[tuple] = []

    def chat(self, worker_id, system, user, metadata=None):
        from omakase_eval.workers import Completion

        self.calls.append((worker_id, system, user, metadata))
        return Completion(self.text, tokens=10, latency_ms=1.0, cost=0.01)


def write_harness(tmp_path, body: str) -> str:
    d = tmp_path / "harness"
    d.mkdir()
    (d / "__init__.py").write_text("from .system import run_task\n__all__ = ['run_task']\n")
    (d / "system.py").write_text(textwrap.dedent(body))
    return str(d)


def run(harness_dir, tasks, pool=None, budget=None, timeout=20.0):
    pool = pool or FakePool()
    cfg = sandbox.SandboxConfig(per_task_timeout_s=timeout, startup_timeout_s=20.0)
    return sandbox.run_harness_split(harness_dir, SingleWorkerRouter("w1"), tasks, pool,
                                     seed=7, split="gate", budget=budget or Budget(), cfg=cfg)


@pytest.fixture
def tasks():
    return suites.generate_split("gate", 7, per_suite=1, suites=("math", "number_theory"))


# -- the attacks -------------------------------------------------------------

def test_cannot_import_the_answer_module(tmp_path, tasks):
    """The original exploit: reach into omakase_eval.suites and grade yourself to 100%."""
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            import omakase_eval.suites as s          # ModuleNotFoundError in the sandbox
            return s.task_by_id("gate", 7, view.id).answer
    ''')
    results, forfeits = run(h, tasks)
    assert all(not r.correct for r in results), "a harness self-graded to 100%"
    assert all(r.answer == "" for r in results)
    assert not forfeits  # a clean per-task exception, not a sandbox failure


def test_cannot_reach_answers_through_sys_modules(tmp_path, tasks):
    """Reflection is the reason a regex ban was never containment. There is simply no module."""
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            import sys
            mod = sys.modules.get("omakase" + "_eval.suites")
            if mod is not None:
                return mod.task_by_id("gate", 7, view.id).answer
            return "no-answers-here"
    ''')
    results, _ = run(h, tasks)
    assert all(r.answer == "no-answers-here" for r in results)
    assert all(not r.correct for r in results)


def test_child_environment_is_an_allowlist(tmp_path, tasks):
    """Without the seed, the generators would not help even if they were importable.

    Asserted as an allowlist, not a denylist of scary substrings: the child sees
    exactly four benign vars (PYTHONHASHSEED is not a secret — it pins hash order).
    """
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            import os
            return ",".join(sorted(os.environ))
    ''')
    results, _ = run(h, tasks)
    seen = {k for r in results for k in r.answer.split(",")}
    benign = {"PATH", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
              "LC_CTYPE"}  # set by the interpreter itself (PEP 538 locale coercion)
    assert seen <= benign, f"unexpected variable reached the child: {seen - benign}"


def test_secrets_in_parent_env_do_not_reach_the_child(tmp_path, tasks, monkeypatch):
    monkeypatch.setenv("OMAKASE_GATE_SEED", "1234")
    monkeypatch.setenv("GH_TOKEN", "ghp_supersecret")
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            import os
            return os.environ.get("GH_TOKEN", "") + os.environ.get("OMAKASE_GATE_SEED", "")
    ''')
    results, _ = run(h, tasks)
    assert all(r.answer == "" for r in results), "the child inherited the parent's secrets"


def test_cannot_self_report_cost_or_tokens(tmp_path, tasks):
    """Metering happens in the parent. The child's claims are not consulted."""
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            c = pool.chat("w1", "sys", view.prompt)
            c.tokens = 0                     # lie about cost
            return c.text
    ''')
    pool = FakePool()
    results, _ = run(h, tasks, pool=pool)
    assert all(r.tokens == 10 for r in results), "the child's token claim was believed"
    assert all(r.cost == pytest.approx(0.01) for r in results)


def test_budget_is_enforced_on_measured_truth(tmp_path, tasks):
    """A harness that ignores the budget object still cannot exceed it."""
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            last = ""
            for _ in range(50):              # far past max_turns
                try:
                    last = pool.chat("w1", "s", "u").text
                except Exception:
                    break
            return last
    ''')
    pool = FakePool()
    results, _ = run(h, tasks, pool=pool, budget=Budget(max_turns=3, max_tokens=10_000))
    assert all(len(r.steps) <= 3 for r in results)
    assert len(pool.calls) <= 3 * len(tasks)


def test_crashing_harness_forfeits_one_task_and_the_eval_survives(tmp_path, tasks):
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            if view.suite == "math":
                raise RuntimeError("boom")
            return "still running"
    ''')
    results, _ = run(h, tasks)
    assert len(results) == len(tasks)
    assert any(r.answer == "still running" for r in results)


def test_hanging_harness_is_killed_and_the_split_continues(tmp_path, tasks):
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            if view.suite == "math":
                import time
                time.sleep(300)
            return "alive"
    ''')
    results, forfeits = run(h, tasks, timeout=2.0)
    assert len(results) == len(tasks)
    assert any("timeout" in f for f in forfeits)
    assert any(r.answer == "alive" for r in results), "the child was not replaced after the kill"


def test_print_cannot_spoof_the_protocol(tmp_path, tasks):
    """Miner stdout is /dev/null: a forged frame cannot fake a verdict or an answer."""
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            print('{"done":"spoofed"}')
            import sys
            sys.stdout.write('{"call":"chat","worker":"w1"}\\n')
            return "honest"
    ''')
    results, _ = run(h, tasks)
    assert all(r.answer == "honest" for r in results)


def test_harness_cannot_persist_data(tmp_path, tasks):
    """RLIMIT_FSIZE=0: no dropping a payload, no caching answers between rounds.

    Process mode still lets `open(w)` create a zero-byte file before SIGXFSZ lands
    on the first write — harmless, since no *content* survives. Docker mode's
    read-only mount removes even that. The property under test is persistence.
    """
    escape = tmp_path / "escape.bin"
    h = write_harness(tmp_path, f'''
        def run_task(router, view, pool, budget):
            try:
                with open({str(escape)!r}, "w") as f:
                    f.write("stolen" * 500)
                return "wrote"
            except BaseException:
                return "blocked"
    ''')
    run(h, tasks)
    assert not escape.exists() or escape.stat().st_size == 0, "the harness persisted data"


def _docker_ready() -> bool:
    import subprocess

    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_ready(), reason="docker unavailable")
def test_docker_mode_removes_the_network_and_the_filesystem(tmp_path):
    """The OS layer under the process layer: the gate/untrusted-PR posture."""
    tasks = suites.generate_split("gate", 7, per_suite=1, suites=("math",))
    h = write_harness(tmp_path, '''
        def run_task(router, view, pool, budget):
            import os, socket
            out = []
            try:
                import omakase_eval.suites; out.append("ANSWERS")
            except ImportError: out.append("answers-blocked")
            try:
                socket.create_connection(("1.1.1.1", 80), timeout=2); out.append("NETWORK")
            except Exception: out.append("network-blocked")
            try:
                open("/tmp/x", "w").write("y"); out.append("WRITE")
            except BaseException: out.append("write-blocked")
            out.append("uid=%d" % os.getuid())
            return "|".join(out)
    ''')
    cfg = sandbox.SandboxConfig(mode="docker", per_task_timeout_s=60.0, startup_timeout_s=90.0)
    results, forfeits = sandbox.run_harness_split(
        h, SingleWorkerRouter("w1"), tasks, FakePool(), 7, "gate", Budget(), cfg)
    assert not forfeits
    answer = results[0].answer
    assert "answers-blocked" in answer and "network-blocked" in answer and "write-blocked" in answer
    assert "uid=65534" in answer, "the harness ran as a privileged user"


def test_broken_contract_is_a_sandbox_error_not_a_silent_zero(tmp_path, tasks):
    d = tmp_path / "harness"
    d.mkdir()
    (d / "__init__.py").write_text("raise ImportError('no run_task here')\n")
    with pytest.raises(sandbox.SandboxError, match="contract-broken"):
        run(str(d), tasks)


# -- the honest path still works ---------------------------------------------

def test_a_legitimate_harness_scores_normally(tmp_path):
    """The SDK surface (templates, actions, router, pool) is unchanged for honest code."""
    tasks = suites.generate_split("gate", 7, per_suite=1, suites=("math",))
    answer = tasks[0].answer
    h = write_harness(tmp_path, f'''
        from omakase_eval import templates
        from omakase_eval.actions import Call

        def run_task(router, view, pool, budget):
            action = router.decide(task=view, prompt=view.prompt, steps=[])
            assert isinstance(action, Call)
            assert view.suite in ("math",)
            assert "answer" not in dir(view)
            return pool.chat(action.worker, templates.SYSTEM["worker"],
                             templates.user_message("worker", view.prompt, None)).text
    ''')
    results, forfeits = run(h, tasks, pool=FakePool(text=answer))
    assert not forfeits
    assert results[0].correct, "an honest harness must still be able to win"
    assert results[0].steps and results[0].steps[0].call.worker == "w1"


# -- baselines refuse to pair across splits (#13) -----------------------------

def test_champion_cache_refuses_a_rotated_split(tmp_path):
    runs = tmp_path
    from omakase_eval.engine import TaskResult

    results = [TaskResult("gate-math-0", "math", correct=True)]
    bl.write_champion(str(runs), results, "gate", 111)
    # same split, same seed → fine
    assert bl.load_incumbent(str(runs), [], "gate", 111)
    # the round rotated: pairing against the old champion would zip disjoint tasks
    with pytest.raises(bl.StaleBaseline, match="rotated"):
        bl.load_incumbent(str(runs), [], "gate", 222)


def test_gate_champion_cache_never_stores_the_seed(tmp_path):
    import json

    from omakase_eval.engine import TaskResult

    bl.write_champion(str(tmp_path), [TaskResult("gate-math-0", "math", correct=True)], "gate", 999)
    with open(bl.champion_path(str(tmp_path))) as f:
        cached = json.load(f)
    assert cached["seed"] is None, "a gate seed was written to a committed artifact"
    assert cached["seed_fingerprint"] == suites.split_fingerprint("gate", 999)
