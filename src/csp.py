"""WordleCSP: Wordle feedback as a constraint satisfaction problem.

Variables   the five letter positions of the hidden word.
Domains     the letters still allowed at each position.
Constraints (derived from every (guess, feedback) pair)
            * green at position i      -> position i must be that letter
            * yellow / grey at i       -> position i can not be that letter
            * green or yellow letters  -> the word contains at least that many copies
            * a grey copy of a letter  -> the word contains no more copies than the
                                          number of green/yellow copies of it
            * global constraint        -> the assembled word must be in the vocabulary

Two ways to use it:
    filter(words) / is_consistent(word)   direct constraint test, used by the agents
                                          during play because it is fast;
    solve(words, mode)                    explicit backtracking search that enumerates
                                          the consistent words, either "plain"
                                          (fixed order, prefix check) or "mrv_fc"
                                          (minimum-remaining-values variable ordering
                                          plus forward checking). Both return exactly
                                          the same words as filter(); they differ in
                                          how much work (checks, nodes, backtracks)
                                          they need, which is what the experiments compare.

Counters: every time a word, letter or count is tested against a constraint,
`checks` is increased by one. `nodes` counts search nodes; `backtracks` counts
nodes that led to no solution.
"""
from __future__ import annotations

import string

from game import GREEN, GREY, WORD_LENGTH


class WordleCSP:
    def __init__(self, length: int = WORD_LENGTH):
        self.length = length
        self.domains: list[set[str]] = [set(string.ascii_lowercase) for _ in range(length)]
        self.min_count: dict[str, int] = {}
        self.max_count: dict[str, int] = {}
        self.history: list[tuple[str, tuple[int, ...]]] = []
        self.checks = 0
        self.nodes = 0
        self.backtracks = 0

    # ------------------------------------------------------------------ #
    def reset_counters(self) -> None:
        self.checks = self.nodes = self.backtracks = 0

    def add_feedback(self, guess: str, pattern) -> None:
        """Turn one (guess, feedback) pair into constraints."""
        pattern = tuple(pattern)
        if len(guess) != self.length or len(pattern) != self.length:
            raise ValueError("guess and pattern must match the word length")
        self.history.append((guess, pattern))
        marked: dict[str, int] = {}        # copies of a letter shown as green/yellow
        has_grey: set[str] = set()
        for i, (letter, p) in enumerate(zip(guess, pattern)):
            if p == GREEN:
                self.domains[i] &= {letter}
            else:
                self.domains[i].discard(letter)
            if p == GREY:
                has_grey.add(letter)
            else:
                marked[letter] = marked.get(letter, 0) + 1
        for letter, n in marked.items():
            self.min_count[letter] = max(self.min_count.get(letter, 0), n)
        for letter in has_grey:
            n = marked.get(letter, 0)
            self.max_count[letter] = min(self.max_count.get(letter, self.length), n)

    # ------------------------------------------------------------------ #
    def _counts_ok(self, word: str) -> bool:
        for letter, n in self.min_count.items():
            if word.count(letter) < n:
                return False
        for letter, n in self.max_count.items():
            if word.count(letter) > n:
                return False
        return True

    def is_consistent(self, word: str) -> bool:
        """True if `word` satisfies every constraint collected so far."""
        self.checks += 1
        for i, letter in enumerate(word):
            if letter not in self.domains[i]:
                return False
        return self._counts_ok(word)

    def filter(self, words) -> list[str]:
        """Keep only the words that satisfy all constraints."""
        return [w for w in words if self.is_consistent(w)]

    # ------------------------------------------------------------------ #
    def solve(self, words, mode: str = "mrv_fc") -> list[str]:
        """Enumerate the consistent words by backtracking search over the positions.

        mode="plain"  : positions assigned left to right; a partial word is
                        rejected as soon as it is not the start of any vocabulary
                        word or breaks a letter-count limit.
        mode="mrv_fc" : the next position is the one with the fewest remaining
                        letters (MRV); after each assignment forward checking
                        shrinks the other positions' domains to letters that still
                        occur in some matching vocabulary word, and fails early if
                        a domain empties or a required letter can no longer fit.
        """
        words = list(words)
        if mode == "plain":
            found = self._solve_plain(words)
        elif mode == "mrv_fc":
            found = self._solve_mrv_fc(words)
        else:
            raise ValueError("mode must be 'plain' or 'mrv_fc'")
        return sorted(found)

    # -- plain backtracking -------------------------------------------- #
    def _solve_plain(self, words: list[str]) -> list[str]:
        L = self.length
        prefixes = {w[:k] for w in words for k in range(1, L + 1)}
        solutions: list[str] = []

        def rec(prefix: str) -> int:
            self.nodes += 1
            pos = len(prefix)
            if pos == L:
                self.checks += 1
                if self._counts_ok(prefix):
                    solutions.append(prefix)
                    return 1
                return 0
            found = 0
            for letter in sorted(self.domains[pos]):
                self.checks += 1                       # prefix + count test
                nxt = prefix + letter
                if nxt not in prefixes:
                    continue
                limit = self.max_count.get(letter)
                if limit is not None and nxt.count(letter) > limit:
                    continue
                found += rec(nxt)
            if found == 0:
                self.backtracks += 1
            return found

        rec("")
        return solutions

    # -- MRV + forward checking ---------------------------------------- #
    def _solve_mrv_fc(self, words: list[str]) -> list[str]:
        L = self.length
        solutions: list[str] = []

        def feasible(assigned: dict[int, str], doms: dict[int, set[str]]) -> bool:
            """Each required letter must still have enough free positions."""
            for letter, need in self.min_count.items():
                self.checks += 1
                have = sum(1 for x in assigned.values() if x == letter)
                free = sum(1 for p in doms if letter in doms[p])
                if have + free < need:
                    return False
            return True

        def rec(assigned: dict[int, str], support: list[str]) -> int:
            self.nodes += 1
            if len(assigned) == L:
                word = "".join(assigned[p] for p in range(L))
                self.checks += 1
                if self._counts_ok(word):
                    solutions.append(word)
                    return 1
                return 0
            # forward checking: letters that still have support in the vocabulary
            doms: dict[int, set[str]] = {}
            for p in range(L):
                if p in assigned:
                    continue
                self.checks += len(support)
                doms[p] = {w[p] for w in support} & self.domains[p]
                if not doms[p]:
                    self.backtracks += 1
                    return 0
            if not feasible(assigned, doms):
                self.backtracks += 1
                return 0
            # MRV: unassigned position with the smallest domain (ties: leftmost)
            pos = min(doms, key=lambda q: (len(doms[q]), q))
            found = 0
            for letter in sorted(doms[pos]):
                self.checks += 1
                used = sum(1 for x in assigned.values() if x == letter) + 1
                limit = self.max_count.get(letter)
                if limit is not None and used > limit:
                    continue
                self.checks += len(support)
                narrowed = [w for w in support if w[pos] == letter]
                found += rec({**assigned, pos: letter}, narrowed)
            if found == 0:
                self.backtracks += 1
            return found

        rec({}, words)
        return solutions
