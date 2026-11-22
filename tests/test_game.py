import random

import numpy as np

import conftest  # noqa: F401  (sets sys.path)
from game import (WIN_PATTERN, WordleGame, build_pattern_matrix, compute_pattern,
                  decode_pattern, encode_pattern, parse_pattern, pattern_to_string)
from sample_words import ALLOWED, ANSWERS, WORDS


def test_known_patterns():
    assert compute_pattern("crane", "crane") == WIN_PATTERN
    assert compute_pattern("speed", "abide") == (0, 0, 1, 0, 1)   # one e: first e yellow, second grey
    assert compute_pattern("geese", "those") == (0, 0, 0, 2, 2)   # greens use up the answer's only e
    assert compute_pattern("llama", "hello") == (1, 1, 0, 0, 0)   # l, l both in word, elsewhere
    assert compute_pattern("sissy", "glass") == (1, 0, 0, 2, 0)   # only one spare s -> first s yellow
    assert compute_pattern("abcde", "vwxyz") == (0, 0, 0, 0, 0)


def test_repeated_letter_cases():
    # answer has two e: both guess e's can be marked
    assert compute_pattern("eerie", "geese") == (1, 2, 0, 0, 2)
    # answer has one s and it is matched green by the guess' 4th letter: other s's are grey
    assert compute_pattern("sassy", "those") == (0, 0, 0, 2, 0)


def test_encode_decode_roundtrip():
    for code in range(243):
        assert encode_pattern(decode_pattern(code)) == code
    assert pattern_to_string((2, 1, 0, 0, 2)) == "gyxxg"
    assert parse_pattern("gyxxg") == (2, 1, 0, 0, 2)
    assert parse_pattern("21002") == (2, 1, 0, 0, 2)


def test_parse_pattern_errors():
    for bad in ("gyxx", "gyxxgg", "gyxxq"):
        try:
            parse_pattern(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_pattern_matrix_matches_scalar_function():
    guesses, answers = ALLOWED[:120], ANSWERS[:120]
    matrix = build_pattern_matrix(guesses, answers)
    assert matrix.shape == (120, 120)
    for gi, g in enumerate(guesses):
        for ai, a in enumerate(answers):
            assert matrix[gi, ai] == encode_pattern(compute_pattern(g, a)), (g, a)


def test_pattern_matrix_full_sample_repeated_letters():
    rng = random.Random(1)
    guesses = rng.sample(WORDS, 60) + ["geese", "speed", "llama", "eerie", "sissy"]
    answers = rng.sample(WORDS, 80) + ["those", "abide", "sweep", "gloss"]
    matrix = build_pattern_matrix(guesses, answers)
    for gi, g in enumerate(guesses):
        for ai, a in enumerate(answers):
            assert matrix[gi, ai] == encode_pattern(compute_pattern(g, a)), (g, a)


def test_game_flow_and_validation():
    game = WordleGame(ANSWERS, ALLOWED, seed=5)
    game.reset(secret="crane" if "crane" in ALLOWED else ANSWERS[0])
    secret = game.secret
    assert not game.is_over()
    try:
        game.guess("zzzzz")
        raise AssertionError("invalid word accepted")
    except ValueError:
        pass
    try:
        game.guess("abc")
        raise AssertionError("short word accepted")
    except ValueError:
        pass
    wrong = next(w for w in ALLOWED if w != secret)
    game.guess(wrong)
    assert game.guesses_used == 1 and not game.is_won()
    assert game.guess(secret) == WIN_PATTERN
    assert game.is_won() and game.is_over()
    try:
        game.guess(wrong)
        raise AssertionError("guess after game over accepted")
    except RuntimeError:
        pass


def test_game_loses_after_six_guesses_and_seed_is_reproducible():
    a = WordleGame(ANSWERS, ALLOWED, seed=11)
    b = WordleGame(ANSWERS, ALLOWED, seed=11)
    assert a.secret == b.secret
    wrong = [w for w in ALLOWED if w != a.secret][:6]
    for w in wrong:
        a.guess(w)
    assert a.is_over() and not a.is_won()
