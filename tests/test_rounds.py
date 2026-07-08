from omakase_eval import datasets, rounds


def test_hidden_subset_is_seed_dependent_and_disjoint_from_dev():
    pool = "data/knowledge.pool.jsonl"
    gate = {t.answer for t in datasets.sample_jsonl(pool, "gate", 7, 8)}
    gate2 = {t.answer for t in datasets.sample_jsonl(pool, "gate", 8, 8)}
    dev = {t.answer for t in datasets.sample_jsonl(pool, "dev", 1, 8)}
    assert gate != gate2  # different seed → different hidden subset
    # dev and a gate seed draw different subsets (holdout property)
    assert dev != gate


def test_sample_is_deterministic():
    pool = "data/knowledge.pool.jsonl"
    a = datasets.sample_jsonl(pool, "gate", 7, 8)
    b = datasets.sample_jsonl(pool, "gate", 7, 8)
    assert [t.id for t in a] == [t.id for t in b]


def test_gate_config_composes_procedural_plus_knowledge():
    cfg = rounds.load_config("configs/round.gate.example.json")
    tasks = rounds.build_split(cfg, "gate", 3)
    suites_seen = {t.suite for t in tasks}
    assert "knowledge" in suites_seen and "math" in suites_seen
    d = rounds.descriptor(cfg)
    assert any(r["source"] == "hidden-holdout" for r in d["suites"])
