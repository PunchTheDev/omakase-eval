from omakase_eval import baselines as bl
from omakase_eval import engine, suites, transcripts
from omakase_eval.routers import SingleWorkerRouter

TASKS = suites.generate_split("dev", 1)[:10]


def test_transcript_captures_steps_and_is_content_addressed(pool, tmp_path):
    results = engine.run_split(SingleWorkerRouter("generalist-mock"), TASKS, pool, 1, "dev")
    tx = transcripts.build(TASKS, results, 1, header={"competition": "omakase-router"})

    assert len(tx["tasks"]) == len(TASKS)
    rec = tx["tasks"][0]
    assert rec["steps"] and rec["steps"][0]["worker"] == "generalist-mock"
    assert "prompt" in rec and "answer" in rec

    digest = transcripts.write(tx, str(tmp_path))
    assert transcripts.write(tx, str(tmp_path)) == digest  # idempotent
    assert transcripts.read(str(tmp_path), digest) == tx
    assert transcripts.read(str(tmp_path), "0" * 64) is None


def test_summary_matches_tasks(pool):
    results = engine.run_split(SingleWorkerRouter("math-mock"), TASKS, pool, 1, "dev")
    tx = transcripts.build(TASKS, results, 1, header={})
    summary = transcripts.summarize(tx)
    assert len(summary) == len(TASKS)
    assert sum(r["correct"] for r in summary) == sum(r.correct for r in results)
    assert all(r["n_steps"] >= 1 for r in summary)
