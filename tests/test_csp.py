import random

import conftest  # noqa: F401
from csp import WordleCSP
from game import compute_pattern
from sample_words import ALLOWED, ANSWERS, WORDS


def brute_force(words, boards):
    """Words that reproduce every observed feedback exactly (the ground truth)."""
    return [w for w in words if all(compute_pattern(g, w) == p for g, p in boards)]


def test_filter_equals_brute_force_single_feedback():
    rng = random.Random(3)
    for _ in range(40):
        guess = rng.choice(WORDS)
        secret = rng.choice(ANSWERS)
        pattern = compute_pattern(guess, secret)
        csp = WordleCSP()
        csp.add_feedback(guess, pattern)
        assert csp.filter(ANSWERS) == brute_force(ANSWERS, [(guess, pattern)])
        assert secret in csp.filter(ANSWERS)


def test_filter_equals_brute_force_multiple_feedback():
    rng = random.Random(4)
    for _ in range(40):
        secret = rng.choice(ANSWERS)
        csp, boards = WordleCSP(), []
        for guess in rng.sample(WORDS, 3):
            pattern = compute_pattern(guess, secret)
            csp.add_feedback(guess, pattern)
            boards.append((guess, pattern))
        assert csp.filter(ANSWERS) == brute_force(ANSWERS, boards)


def test_repeated_letter_constraints():
    csp = WordleCSP()
    csp.add_feedback("speed", compute_pattern("speed", "abide"))   # one e, d present
    assert csp.max_count["e"] == 1 and csp.min_count["e"] == 1
    assert "abide" in csp.filter(ANSWERS)
    assert "geese" not in csp.filter(WORDS)


def test_solve_modes_match_filter():
    rng = random.Random(6)
    for _ in range(15):
        secret = rng.choice(ANSWERS)
        csp = WordleCSP()
        for guess in rng.sample(WORDS, rng.choice([1, 2])):
            csp.add_feedback(guess, compute_pattern(guess, secret))
        expected = sorted(csp.filter(ANSWERS))
        assert csp.solve(ANSWERS, "plain") == expected
        assert csp.solve(ANSWERS, "mrv_fc") == expected


def test_solve_without_feedback_returns_whole_vocabulary():
    csp = WordleCSP()
    assert csp.solve(ANSWERS, "plain") == sorted(ANSWERS)
    assert csp.solve(ANSWERS, "mrv_fc") == sorted(ANSWERS)


def test_counters_increase():
    csp = WordleCSP()
    csp.add_feedback("crane", compute_pattern("crane", "trace") if "trace" in WORDS else (0,) * 5)
    csp.reset_counters()
    csp.solve(ANSWERS, "mrv_fc")
    assert csp.checks > 0 and csp.nodes > 0
