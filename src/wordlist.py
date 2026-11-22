"""Loading and validating the Wordle answer and allowed-guess word lists."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from game import WORD_LENGTH, build_pattern_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _clean(words, source: str) -> list[str]:
    """Lowercase, strip, check every entry is five letters; return sorted unique words."""
    out = set()
    for line_no, raw in enumerate(words, start=1):
        w = raw.strip().lower()
        if not w:
            continue
        if len(w) != WORD_LENGTH or not w.isascii() or not w.isalpha():
            raise ValueError(f"{source}, line {line_no}: '{raw.strip()}' is not a "
                             f"{WORD_LENGTH}-letter word")
        out.add(w)
    return sorted(out)


class WordList:
    """The hidden-answer list and the (larger) list of words accepted as guesses.

    `allowed` always starts with the answers (in the same order), followed by the
    extra guess-only words, so answer index i is also guess index i. This lets the
    pattern matrix be addressed with the same indices for both.
    """

    def __init__(self, answers, allowed=None):
        self.answers: list[str] = _clean(answers, "answers")
        if not self.answers:
            raise ValueError("the answer list is empty")
        extra = set(_clean(allowed, "allowed")) - set(self.answers) if allowed is not None else set()
        self.allowed: list[str] = self.answers + sorted(extra)
        self.answer_index = {w: i for i, w in enumerate(self.answers)}
        self.allowed_index = {w: i for i, w in enumerate(self.allowed)}
        self._matrix: np.ndarray | None = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, data_dir=None, answers_file="answers.txt", allowed_file="allowed.txt") -> "WordList":
        """Read answers.txt and allowed.txt (one word per line) from data/raw/."""
        base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        raw_dir = base / "raw" if (base / "raw").is_dir() else base
        paths = {}
        for label, name in (("answers", answers_file), ("allowed", allowed_file)):
            path = raw_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Word list '{path}' not found. Run 'python scripts/download_words.py' "
                    f"to download the Wordle lists into data/raw/ (see README).")
            paths[label] = path
        answers = paths["answers"].read_text(encoding="utf-8").splitlines()
        allowed = paths["allowed"].read_text(encoding="utf-8").splitlines()
        return cls(answers, allowed)

    # -- queries ------------------------------------------------------------ #
    def is_valid(self, word: str) -> bool:
        """True if `word` may be used as a guess."""
        return word in self.allowed_index

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update("\n".join(self.allowed).encode())
        h.update(b"|")
        h.update("\n".join(self.answers).encode())
        return h.hexdigest()[:12]

    def pattern_matrix(self, cache_dir=None) -> np.ndarray:
        """uint8 matrix [guess index, answer index] of feedback codes (cached on disk).

        The matrix is a pure speed-up: it is built with the same rules as
        game.compute_pattern (checked in tests/test_game.py).
        """
        if self._matrix is not None:
            return self._matrix
        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_path = cache_dir / f"pattern_matrix_{self.fingerprint()}.npy"
            if cache_path.is_file():
                try:
                    cached = np.load(cache_path)
                    if cached.shape == (len(self.allowed), len(self.answers)):
                        self._matrix = cached
                        return cached
                except (OSError, ValueError):
                    pass                      # corrupt cache: rebuild below
        matrix = build_pattern_matrix(self.allowed, self.answers)
        if cache_path is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, matrix)
        self._matrix = matrix
        return matrix
