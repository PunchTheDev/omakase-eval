from omakase_eval import suites


def test_split_is_deterministic():
    a = suites.generate_split("dev", 1)
    b = suites.generate_split("dev", 1)
    assert a == b
    n = len(suites.SUITES) * 40
    assert len(a) == n and len({t.id for t in a}) == n


def test_different_seeds_differ():
    assert suites.generate_split("gate", 1) != suites.generate_split("gate", 2)


def test_option_shuffle_varies_by_run_seed():
    task = next(t for t in suites.generate_split("dev", 1) if t.options)
    orders = {tuple(text for _, text in suites.shuffled_options(task, s)) for s in range(20)}
    assert len(orders) > 1


def test_grade_by_content_and_letter():
    task = next(t for t in suites.generate_split("dev", 1) if t.options)
    assert suites.grade(task, f"reasoning...\n{task.answer}", run_seed=7)
    letter = next(l for l, text in suites.shuffled_options(task, 7) if text == task.answer)
    assert suites.grade(task, letter, run_seed=7)
    wrong = next(l for l, text in suites.shuffled_options(task, 7) if text != task.answer)
    assert not suites.grade(task, wrong, run_seed=7)


def test_grade_numeric():
    task = next(t for t in suites.generate_split("dev", 1) if not t.options)
    assert suites.grade(task, f"The answer is {task.answer}", run_seed=1)
    assert not suites.grade(task, "no idea", run_seed=1)


def test_grade_rejects_answer_spray():
    """The old `answer in tokens` rule let a harness win every task by returning
    every candidate at once. Grading must reject candidate sprays."""
    tasks = suites.generate_split("gate", 0xBEEF, per_suite=25)
    numeric_spray = " ".join(str(n) for n in range(-3000, 3001))
    won = 0
    for t in tasks:
        spray = " ".join(t.options) if t.options else numeric_spray  # all options / all ints on one line
        won += suites.grade(t, spray, run_seed=0xBEEF)
    assert won == 0, f"{won}/{len(tasks)} tasks won by spraying candidates — grading is exploitable"

    # honest bare answers on the final line still pass
    assert all(suites.grade(t, t.answer, run_seed=0xBEEF) for t in tasks)


def test_grade_rejects_multitoken_numeric_membership():
    """A numeric answer must be the final token, not merely present among many."""
    task = next(t for t in suites.generate_split("dev", 1) if not t.options)
    assert not suites.grade(task, f"maybe {task.answer} or 999999", run_seed=1)
    assert suites.grade(task, f"...therefore {task.answer}", run_seed=1)
