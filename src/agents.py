"""Agent base class and the Random, Frequency and Entropy agents.

All agents share the same loop (Agent.play): guess -> receive feedback -> add the
feedback to the CSP -> keep only the consistent candidates. They differ only in
choose_guess(), i.e. in how they pick the next word.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from csp import WordleCSP
from game import NUM_PATTERNS, WIN_PATTERN, WordleGame, build_pattern_matrix
from wordlist import WordList

NAN = float("nan")


@dataclass
class GameResult:
    """Everything recorded about one played game."""
    agent: str
    secret: str
    won: bool
    n_guesses: int
    guesses: list[str] = field(default_factory=list)
    patterns: list[tuple[int, ...]] = field(default_factory=list)
    candidates_after: list[int] = field(default_factory=list)   # candidates left after each guess
    bits: list[float] = field(default_factory=list)             # actual bits gained per guess
    expected_bits: list[float] = field(default_factory=list)    # entropy predicted before each guess
    csp_checks: int = 0
    seconds: float = 0.0


class Agent:
    """Abstract Wordle player. Candidates are indices into WordList.answers."""

    name = "Agent"

    def __init__(self, words: WordList, seed=None):
        self.words = words
        self.rng = random.Random(seed)
        self.last_expected_bits = NAN

    def choose_guess(self, candidates: list[int], history: list) -> str:
        """Return the next guess. Subclasses override this."""
        raise NotImplementedError

    def play(self, game: WordleGame) -> GameResult:
        """Play `game` to the end using a WordleCSP to track what is still possible."""
        answers = self.words.answers
        csp = WordleCSP()
        candidates = list(range(len(answers)))
        history: list[tuple[str, tuple[int, ...]]] = []
        result = GameResult(self.name, game.secret, False, 0)
        start = time.perf_counter()
        while not game.is_over():
            self.last_expected_bits = NAN
            guess = self.choose_guess(candidates, history)
            result.expected_bits.append(self.last_expected_bits)
            pattern = game.guess(guess)
            history.append((guess, pattern))
            result.guesses.append(guess)
            result.patterns.append(pattern)
            csp.add_feedback(guess, pattern)
            before = len(candidates)
            candidates = [i for i in candidates if csp.is_consistent(answers[i])]
            after = len(candidates)
            result.candidates_after.append(after)
            result.bits.append(math.log2(before / after) if after > 0 else NAN)
            if pattern == WIN_PATTERN:
                break
        result.seconds = time.perf_counter() - start
        result.won = game.is_won()
        result.n_guesses = game.guesses_used
        result.csp_checks = csp.checks
        return result



def weighted_entropies(pattern_matrix: np.ndarray, weights: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Entropy (bits) of every row of a [guess, candidate] pattern matrix when the
    candidates are not equally likely: candidate j has probability weights[j] / sum(weights)."""
    G, K = pattern_matrix.shape
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    out = np.empty(G, dtype=np.float64)
    for start in range(0, G, chunk):
        block = pattern_matrix[start:start + chunk].astype(np.int64)
        rows = block.shape[0]
        flat = (block + np.arange(rows)[:, None] * NUM_PATTERNS).ravel()
        counts = np.bincount(flat, weights=np.tile(weights, rows), minlength=rows * NUM_PATTERNS)
        p = counts.reshape(rows, NUM_PATTERNS) / total
        safe = np.where(p > 0, p, 1.0)
        out[start:start + rows] = -(p * np.log2(safe)).sum(axis=1)
    return out


class RandomAgent(Agent):
    """Baseline: a random word among those still consistent with all feedback."""

    name = "Random (consistent)"

    def choose_guess(self, candidates, history):
        return self.words.answers[self.rng.choice(candidates)]


class FrequencyAgent(Agent):
    """Greedy heuristic: the consistent word whose letters are most common among
    the remaining candidates (each letter counted once per word)."""

    name = "Letter frequency"

    def choose_guess(self, candidates, history):
        answers = self.words.answers
        if len(candidates) <= 2:
            return answers[candidates[0]]
        freq: dict[str, int] = {}
        for i in candidates:
            for letter in set(answers[i]):
                freq[letter] = freq.get(letter, 0) + 1
        best_index, best_score = candidates[0], -1
        for i in candidates:
            score = sum(freq[letter] for letter in set(answers[i]))
            if score > best_score:
                best_index, best_score = i, score
        return answers[best_index]


class EntropyAgent(Agent):
    """Information-theoretic player.

    For every legal guess g the remaining candidates split into groups by the
    feedback pattern k that g would produce. If pattern k covers a fraction p(k)
    of the candidates, the expected information of the guess is
        H(g) = - sum_k p(k) * log2 p(k)      (bits).
    The agent plays the guess with the largest H; ties favour guesses that are
    themselves possible answers (so a lucky guess can win immediately).
    """

    name = "CSP + Entropy"
    CHUNK = 512

    def __init__(self, words: WordList, seed=None, cache_dir=None):
        super().__init__(words, seed)
        self.matrix = words.pattern_matrix(cache_dir)    # [guess, answer] -> pattern code
        self._opening: tuple[str, float] | None = None

    def entropies(self, candidates: list[int]) -> np.ndarray:
        """Expected information (bits) of every allowed guess over `candidates`."""
        cand = np.asarray(candidates)
        K = len(cand)
        sub = self.matrix if K == self.matrix.shape[1] else self.matrix[:, cand]
        M = sub.shape[0]
        out = np.empty(M, dtype=np.float64)
        for start in range(0, M, self.CHUNK):
            block = sub[start:start + self.CHUNK].astype(np.int64)
            offsets = np.arange(block.shape[0])[:, None] * NUM_PATTERNS
            counts = np.bincount((block + offsets).ravel(),
                                 minlength=block.shape[0] * NUM_PATTERNS)
            counts = counts.reshape(block.shape[0], NUM_PATTERNS)
            p = counts / K
            safe = np.where(counts > 0, p, 1.0)
            out[start:start + block.shape[0]] = -(p * np.log2(safe)).sum(axis=1)
        return out

    def information_gain(self, guess: str, candidates: list[int]) -> float:
        """Expected information (bits) of one specific guess."""
        return float(self.entropies(candidates)[self.words.allowed_index[guess]])

    def rank_guesses(self, candidates: list[int], top: int = 10) -> list[tuple[str, float]]:
        """The `top` best guesses with their expected bits (best first)."""
        ent = self.entropies(candidates)
        order = np.lexsort((np.arange(len(ent)), -np.round(ent, 9)))[:top]
        return [(self.words.allowed[i], float(ent[i])) for i in order]

    def rank_guesses_for(self, candidate_words: list[str], weights=None, top: int = 5) -> list[tuple[str, float]]:
        """Best guesses when the possible answers are ANY list of words (not just the
        official answers), optionally weighted by how likely each one is.

        Used by the helper's "all valid words" mode. The pattern matrix is built on the
        fly for these candidates, so it takes a few seconds when many words are left.
        """
        allowed = self.words.allowed
        if len(candidate_words) == 1:
            return [(candidate_words[0], 0.0)]
        matrix = build_pattern_matrix(allowed, candidate_words)
        w = np.ones(len(candidate_words)) if weights is None else np.asarray(weights, dtype=np.float64)
        ent = weighted_entropies(matrix, w)
        is_candidate = np.isin(np.array(allowed), np.array(candidate_words))
        order = np.lexsort((np.arange(len(allowed)), ~is_candidate, -np.round(ent, 9)))[:top]
        return [(allowed[i], float(ent[i])) for i in order]

    def choose_guess(self, candidates, history):
        answers = self.words.answers
        if len(candidates) <= 2:                      # guessing a candidate is optimal
            self.last_expected_bits = math.log2(len(candidates)) if len(candidates) > 1 else 0.0
            return answers[candidates[0]]
        if not history and self._opening is not None:
            guess, bits = self._opening
            self.last_expected_bits = bits
            return guess
        ent = self.entropies(candidates)
        best = ent.max()
        tied = np.flatnonzero(ent >= best - 1e-9)
        cand_set = set(candidates)
        in_candidates = [i for i in tied if i in cand_set]
        choice = int(in_candidates[0]) if in_candidates else int(tied[0])
        guess = self.words.allowed[choice]
        self.last_expected_bits = float(ent[choice])
        if not history:
            self._opening = (guess, self.last_expected_bits)
        return guess
