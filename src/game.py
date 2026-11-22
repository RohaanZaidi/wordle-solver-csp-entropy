"""Wordle environment: feedback computation and the WordleGame class.

Feedback patterns are tuples of five integers:
    0 = grey (letter not in the word, or all copies already accounted for)
    1 = yellow (letter is in the word but in a different position)
    2 = green (right letter, right position)

A pattern can also be stored as a single integer 0..242 (base 3, position 0 is
the least significant digit); this is what the NumPy pattern matrix uses.
"""
from __future__ import annotations

import random

import numpy as np

WORD_LENGTH = 5
MAX_GUESSES = 6
GREY, YELLOW, GREEN = 0, 1, 2
WIN_PATTERN = (GREEN,) * WORD_LENGTH
NUM_PATTERNS = 3 ** WORD_LENGTH          # 243
WIN_CODE = NUM_PATTERNS - 1              # 242 = all green


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def compute_pattern(guess: str, answer: str) -> tuple[int, ...]:
    """Return the Wordle feedback for `guess` against the hidden `answer`.

    Repeated letters follow the real game: greens are assigned first, then each
    remaining guess letter is yellow only while the answer still has an unmatched
    copy of that letter (scanning left to right); otherwise it is grey.
    """
    if len(guess) != len(answer):
        raise ValueError("guess and answer must have the same length")
    n = len(guess)
    pattern = [GREY] * n
    unmatched: dict[str, int] = {}        # answer letters not used by a green
    for i in range(n):
        if guess[i] == answer[i]:
            pattern[i] = GREEN
        else:
            unmatched[answer[i]] = unmatched.get(answer[i], 0) + 1
    for i in range(n):
        if pattern[i] == GREEN:
            continue
        letter = guess[i]
        if unmatched.get(letter, 0) > 0:
            pattern[i] = YELLOW
            unmatched[letter] -= 1
    return tuple(pattern)


def encode_pattern(pattern) -> int:
    """Tuple of 0/1/2 -> integer 0..242."""
    return sum(int(p) * 3 ** i for i, p in enumerate(pattern))


def decode_pattern(code: int, length: int = WORD_LENGTH) -> tuple[int, ...]:
    """Integer 0..242 -> tuple of 0/1/2."""
    out = []
    for _ in range(length):
        out.append(code % 3)
        code //= 3
    return tuple(out)


def pattern_to_string(pattern) -> str:
    """(2, 0, 1, 0, 0) -> 'gxyxx'  (g = green, y = yellow, x = grey)."""
    return "".join("xyg"[p] for p in pattern)


_CHAR_TO_VALUE = {
    "g": GREEN, "G": GREEN, "2": GREEN,
    "y": YELLOW, "Y": YELLOW, "1": YELLOW,
    "x": GREY, "X": GREY, "b": GREY, "B": GREY, "0": GREY, "-": GREY, ".": GREY,
}


def parse_pattern(text: str, length: int = WORD_LENGTH) -> tuple[int, ...]:
    """Parse user input such as 'gyxxg' or '21002' into a pattern tuple."""
    text = text.strip().replace(" ", "")
    if len(text) != length:
        raise ValueError(f"feedback must be exactly {length} characters (g/y/x or 2/1/0)")
    try:
        return tuple(_CHAR_TO_VALUE[c] for c in text)
    except KeyError as bad:
        raise ValueError(f"unknown feedback character {bad}; use g/y/x or 2/1/0") from None


def words_to_array(words) -> np.ndarray:
    """List of lowercase words -> int8 array of shape (n, 5) with a=0 ... z=25."""
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)


def build_pattern_matrix(guesses, answers) -> np.ndarray:
    """Feedback code of every guess against every answer, as a uint8 matrix.

    result[g, a] == encode_pattern(compute_pattern(guesses[g], answers[a])).
    Vectorised over the answers so that ~13,000 x 2,315 pairs take seconds.
    """
    A = words_to_array(answers)                      # (N, 5)
    G = words_to_array(guesses)                      # (M, 5)
    N = A.shape[0]
    rows = np.arange(N)
    letter_counts = np.zeros((N, 26), dtype=np.int8)
    for i in range(WORD_LENGTH):
        letter_counts[rows, A[:, i]] += 1
    powers = 3 ** np.arange(WORD_LENGTH)
    out = np.empty((G.shape[0], N), dtype=np.uint8)
    for gi in range(G.shape[0]):
        g = G[gi]
        green = A == g[None, :]                      # (N, 5)
        remaining = letter_counts.copy()             # copies of each letter not used by a green
        for i in range(WORD_LENGTH):
            remaining[green[:, i], g[i]] -= 1
        code = np.zeros(N, dtype=np.int32)
        for i in range(WORD_LENGTH):
            letter = g[i]
            yellow = (~green[:, i]) & (remaining[:, letter] > 0)
            remaining[yellow, letter] -= 1
            code += np.where(green[:, i], GREEN, np.where(yellow, YELLOW, GREY)) * powers[i]
        out[gi] = code
    return out


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class WordleGame:
    """One Wordle puzzle: a hidden word, a guess limit and a feedback function."""

    def __init__(self, answers, allowed=None, max_guesses: int = MAX_GUESSES, seed=None):
        self.answers = list(answers)
        self.allowed = set(allowed) if allowed is not None else set()
        self.allowed.update(self.answers)
        self.max_guesses = max_guesses
        self._rng = random.Random(seed)
        self.secret: str = ""
        self.guesses_used = 0
        self.history: list[tuple[str, tuple[int, ...]]] = []
        self.reset()

    def reset(self, secret: str | None = None, seed=None) -> None:
        """Start a new puzzle. Picks the hidden word at random (seeded) unless given."""
        if seed is not None:
            self._rng = random.Random(seed)
        if secret is None:
            secret = self._rng.choice(self.answers)
        elif secret not in self.allowed:
            raise ValueError(f"'{secret}' is not a valid word")
        self.secret = secret
        self.guesses_used = 0
        self.history = []

    def guess(self, word: str) -> tuple[int, ...]:
        """Submit a guess and return its feedback pattern."""
        if self.is_over():
            raise RuntimeError("the game is already over")
        if len(word) != WORD_LENGTH or not word.isalpha() or not word.islower():
            raise ValueError("a guess must be five lowercase letters")
        if word not in self.allowed:
            raise ValueError(f"'{word}' is not in the valid-guess list")
        pattern = compute_pattern(word, self.secret)
        self.guesses_used += 1
        self.history.append((word, pattern))
        return pattern

    def is_won(self) -> bool:
        return bool(self.history) and self.history[-1][1] == WIN_PATTERN

    def is_over(self) -> bool:
        return self.is_won() or self.guesses_used >= self.max_guesses
