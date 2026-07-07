"""End-to-end: baselines → trained tiny router → verdict. The whole competition loop."""
from oc_eval import baselines as bl
from oc_eval import engine, routers, score, suites


def _train_router(pool, base: bl.Baselines, split: str, seed: int) -> routers.TinyRouter:
    tasks = sorted(suites.generate_split(split, seed), key=lambda t: t.id)
    workers = list(pool.workers)
    prompts, labels = [], []
    for t in tasks:
        best = base.best_worker_per_task.get(t.id)
        if best:
            prompts.append(t.prompt)
            labels.append(workers.index(best))
    return routers.fit_tiny_router(prompts, labels, workers, seed=0)


def test_trained_router_beats_best_single(pool):
    split, seed = "dev", 1
    base = bl.compute(pool, split, seed)
    router = _train_router(pool, base, split, seed)

    tasks = suites.generate_split(split, seed)
    results = engine.run_split(router, tasks, pool, seed, split)
    verdict = score.judge(results, bl.deserialize_results(base.best_single_results), base.oracle_accuracy)

    assert verdict.passed, verdict.reason
    assert verdict.candidate.accuracy > verdict.baseline.accuracy
    assert verdict.oracle_capture and verdict.oracle_capture > 0.3


def test_router_persistence_and_manifest_guard(pool, tmp_path):
    base = bl.compute(pool, "dev", 1)
    router = _train_router(pool, base, "dev", 1)
    weights = tmp_path / "weights.json"
    router.save(str(weights))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"arch": "tiny-linear", "weights_file": "weights.json", "weights_sha256": "%s"}'
        % routers.sha256_file(str(weights))
    )
    loaded = routers.load_router(str(manifest), str(tmp_path))
    assert loaded.workers == router.workers

    weights.write_text(weights.read_text().replace("tiny-linear", "tiny-linear "))  # tamper
    try:
        routers.load_router(str(manifest), str(tmp_path))
        raise AssertionError("tampered weights must not load")
    except ValueError as e:
        assert "mismatch" in str(e)
