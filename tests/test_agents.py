import conftest  # noqa: F401
from agents import EntropyAgent, FrequencyAgent, RandomAgent
from csp import WordleCSP
from game import WIN_PATTERN, WordleGame, compute_pattern
from sample_words import ALLOWED, ANSWERS
from wordlist import WordList


def make_words():
    return WordList(ANSWERS, ALLOWED)


def test_wordlist_orders_answers_first():
    words = make_words()
    assert words.allowed[: len(words.answers)] == words.answers
    assert set(words.allowed) == set(ALLOWED)
    assert words.is_valid(ALLOWED[-1]) and not words.is_valid("zzzzz")


def test_wordlist_rejects_bad_words():
    try:
        WordList(["house", "toolong"])
    except ValueError:
        return
    raise AssertionError("bad word was accepted")


def test_every_guess_is_consistent_with_earlier_feedback():
    words = make_words()
    for agent_cls in (RandomAgent, FrequencyAgent, EntropyAgent):
        agent = agent_cls(words, seed=1)
        for secret in words.answers[::7]:
            game = WordleGame(words.answers, words.allowed)
            game.reset(secret=secret)
            result = agent.play(game)
            csp = WordleCSP()
            for k, (guess, pattern) in enumerate(zip(result.guesses, result.patterns)):
                if k > 0 and agent_cls is not EntropyAgent:
                    # non-entropy agents only ever play words that are still possible
                    assert csp.is_consistent(guess)
                csp.add_feedback(guess, pattern)
            assert csp.is_consistent(secret)      # the true word never gets eliminated


def test_entropy_agent_wins_nearly_everything():
    words = make_words()
    agent = EntropyAgent(words, seed=1)
    results = []
    for secret in words.answers:
        game = WordleGame(words.answers, words.allowed)
        game.reset(secret=secret)
        results.append(agent.play(game))
    wins = sum(r.won for r in results)
    assert wins / len(results) >= 0.97


def test_entropy_prefers_even_split_and_bits_are_bounded():
    words = make_words()
    agent = EntropyAgent(words)
    cands = list(range(len(words.answers)))
    ent = agent.entropies(cands)
    assert ent.max() <= 7.93 + 1e-9                  # log2(243)
    best = agent.rank_guesses(cands, top=1)[0]
    assert best[1] == ent.max() or abs(best[1] - ent.max()) < 1e-9
    # a guess that gives the same feedback for every candidate carries 0 bits
    assert agent.information_gain(words.answers[0], [0]) == 0.0


def test_entropy_matches_manual_formula():
    import math
    words = make_words()
    agent = EntropyAgent(words)
    cands = list(range(0, len(words.answers), 3))
    guess = "stare" if words.is_valid("stare") else words.allowed[10]
    counts = {}
    for i in cands:
        p = compute_pattern(guess, words.answers[i])
        counts[p] = counts.get(p, 0) + 1
    manual = -sum((c / len(cands)) * math.log2(c / len(cands)) for c in counts.values())
    assert abs(agent.information_gain(guess, cands) - manual) < 1e-9


def test_play_records_results():
    words = make_words()
    game = WordleGame(words.answers, words.allowed)
    game.reset(secret=words.answers[3])
    r = EntropyAgent(words).play(game)
    assert r.n_guesses == len(r.guesses) == len(r.patterns) == len(r.bits)
    assert r.won == (r.patterns[-1] == WIN_PATTERN)


def test_weighted_entropy_matches_manual_formula_and_uniform_case():
    import math
    import numpy as np
    from agents import weighted_entropies
    from game import build_pattern_matrix
    words = make_words()
    cands = words.allowed[:80]
    matrix = build_pattern_matrix(words.allowed, cands)
    uniform = weighted_entropies(matrix, np.ones(len(cands)))
    agent = EntropyAgent(words)
    plain = agent.entropies([words.answer_index[w] for w in cands if w in words.answer_index])
    assert abs(uniform.max() - uniform.max()) < 1e-12 and uniform.shape[0] == len(words.allowed)
    # manual check with non-uniform weights for one guess
    weights = np.array([1.0 if i % 2 else 0.1 for i in range(len(cands))])
    guess = words.allowed[5]
    groups = {}
    for w, wt in zip(cands, weights):
        groups[compute_pattern(guess, w)] = groups.get(compute_pattern(guess, w), 0.0) + wt
    total = weights.sum()
    manual = -sum((g / total) * math.log2(g / total) for g in groups.values())
    got = weighted_entropies(matrix, weights)[5]
    assert abs(got - manual) < 1e-9
    assert plain.shape[0] == len(words.allowed)


def test_wide_mode_can_find_a_word_outside_the_answer_list():
    words = make_words()
    outside = next(w for w in words.allowed if w not in words.answer_index)
    csp = WordleCSP()
    csp.add_feedback("crane", compute_pattern("crane", outside))
    pool = csp.filter(words.allowed)
    assert outside in pool                                   # wide pool keeps it
    assert outside not in csp.filter(words.answers)          # answer-only pool can not
    agent = EntropyAgent(words)
    ranked = agent.rank_guesses_for(pool, top=3)
    assert len(ranked) >= 1 and all(b >= 0 for _, b in ranked)
    assert agent.rank_guesses_for([outside]) == [(outside, 0.0)]
