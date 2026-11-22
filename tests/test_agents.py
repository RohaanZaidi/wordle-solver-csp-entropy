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
