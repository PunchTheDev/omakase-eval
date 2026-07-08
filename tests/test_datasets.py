from omakase_eval import datasets, suites


def test_jsonl_roundtrips_to_tasks(tmp_path):
    src = tmp_path / "gate.jsonl"
    src.write_text(
        '{"id":"g-reasoning-0","suite":"reasoning","prompt":"2+2?","options":["3","4"],"answer":"4"}\n'
        '{"id":"g-math-0","suite":"math","prompt":"1+1","options":[],"answer":"2"}\n'
    )
    tasks = datasets.load_jsonl(str(src))
    assert len(tasks) == 2
    assert tasks[0].options == ("3", "4") and tasks[0].answer == "4"
    assert suites.grade(tasks[1], "the answer is 2", run_seed=1)


def test_load_split_procedural_matches_generator():
    a = datasets.load_split({"kind": "procedural"}, "dev", 1)
    assert a == suites.generate_split("dev", 1)


def test_unknown_suite_rejected(tmp_path):
    src = tmp_path / "bad.jsonl"
    src.write_text('{"id":"x","suite":"chemistry","prompt":"?","options":[],"answer":"1"}\n')
    try:
        datasets.load_split({"kind": "jsonl", "path": str(src)}, "gate", 1)
        raise AssertionError("should reject unknown suite")
    except ValueError as e:
        assert "chemistry" in str(e)
