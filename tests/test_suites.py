from omakase_eval import suites


def test_split_is_deterministic():
    a = suites.generate_split("dev", 1)
    b = suites.generate_split("dev", 1)
    assert a == b
    assert len(a) == 120 and len({t.id for t in a}) == 120


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
