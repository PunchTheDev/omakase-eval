from oc_eval import stats


def test_mcnemar_null():
    p, w, l = stats.mcnemar_exact([True] * 50 + [False] * 50, [True] * 50 + [False] * 50)
    assert p == 1.0 and w == l == 0


def test_mcnemar_detects_improvement():
    baseline = [False] * 30 + [True] * 70
    candidate = [True] * 20 + [False] * 10 + [True] * 70  # 20 wins, 0 losses
    p, wins, losses = stats.mcnemar_exact(candidate, baseline)
    assert wins == 20 and losses == 0 and p < 1e-4


def test_mcnemar_symmetric_discordance_not_significant():
    baseline = [True, False] * 20
    candidate = [False, True] * 20
    p, _, _ = stats.mcnemar_exact(candidate, baseline)
    assert p > 0.5


def test_bootstrap_ci_brackets_delta():
    baseline = [False] * 40 + [True] * 60
    candidate = [True] * 55 + [False] * 25 + [True] * 20
    lo, hi = stats.paired_bootstrap_ci(candidate, baseline, seed=1)
    delta = (sum(candidate) - sum(baseline)) / 100
    assert lo <= delta <= hi


def test_mde_shrinks_with_n():
    assert stats.minimum_detectable_effect(1000) < stats.minimum_detectable_effect(100)


def test_compare_significant():
    baseline = [False] * 25 + [True] * 95
    candidate = [True] * 120
    c = stats.compare(candidate, baseline)
    assert c.significant and c.delta > 0.2
