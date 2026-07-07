from oc_eval import engine, suites
from oc_eval.actions import Answer, Call
from oc_eval.routers import SingleWorkerRouter

TASKS = suites.generate_split("dev", 1)[:12]


def test_single_worker_router_round_trip(pool):
    results = engine.run_split(SingleWorkerRouter("generalist-mock"), TASKS, pool, 1, "dev")
    assert len(results) == len(TASKS)
    assert all(r.tokens > 0 and r.cost > 0 for r in results)
    assert any(r.correct for r in results)


def test_runs_are_deterministic(pool):
    a = engine.run_split(SingleWorkerRouter("math-mock"), TASKS, pool, 1, "dev")
    b = engine.run_split(SingleWorkerRouter("math-mock"), TASKS, pool, 1, "dev")
    assert [r.correct for r in a] == [r.correct for r in b]


def test_turn_budget_enforced(pool):
    class Chatterbox:  # never answers
        def decide(self, task, prompt, steps):
            return Call("small-mock")

    r = engine.run_task(Chatterbox(), TASKS[0], pool, 1, "dev", engine.Budget(max_turns=3))
    assert len(r.steps) == 3 and not r.correct


def test_malformed_action_forfeits(pool):
    class Broken:
        def decide(self, task, prompt, steps):
            return "gibberish"

    r = engine.run_task(Broken(), TASKS[0], pool, 1, "dev")
    assert not r.correct and not r.steps


def test_verifier_role_flow(pool):
    class DraftThenVerify:
        def decide(self, task, prompt, steps):
            if not steps:
                return Call("generalist-mock")
            if len(steps) == 1:
                return Call("reasoner-mock", role="verifier")
            return Answer(steps[0].response)

    r = engine.run_task(DraftThenVerify(), TASKS[0], pool, 1, "dev")
    assert len(r.steps) == 2
    assert r.steps[1].response in ("CORRECT", "REVISE")
