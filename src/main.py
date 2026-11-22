"""Console menu for the Wordle solver.

    python src/main.py

  1  Watch the solver play a hidden word (pick the agent and the word)
  2  Helper: type in your real game's guesses and colours, get the next best guess
  3  Show the best opening guesses by entropy
  4  Quit
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from agents import Agent, EntropyAgent, FrequencyAgent, RandomAgent
from csp import WordleCSP
from game import WIN_PATTERN, WORD_LENGTH, WordleGame, parse_pattern, pattern_to_string
from wordlist import DEFAULT_DATA_DIR, WordList

if os.name == "nt":                       # enable ANSI colours in the Windows console
    os.system("")

GREEN_BG, YELLOW_BG, GREY_BG, RESET = "\033[42;30m", "\033[43;30m", "\033[100;97m", "\033[0m"


def coloured(word: str, pattern) -> str:
    """The guess as coloured tiles (green / yellow / grey)."""
    tiles = {2: GREEN_BG, 1: YELLOW_BG, 0: GREY_BG}
    return "".join(f"{tiles[p]} {c.upper()} {RESET}" for c, p in zip(word, pattern))


def show_candidates(words: WordList, candidates: list[int], limit: int = 12) -> str:
    shown = ", ".join(words.answers[i] for i in candidates[:limit])
    more = f", ... (+{len(candidates) - limit} more)" if len(candidates) > limit else ""
    return shown + more


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return "q"


# --------------------------------------------------------------------------- #
def choose_agent(words: WordList, cache_dir) -> Agent:
    print("\nWhich agent?\n  1  CSP + Entropy (default)\n  2  Letter frequency\n  3  Random (consistent)")
    choice = ask("Choice [1]: ") or "1"
    if choice == "2":
        return FrequencyAgent(words)
    if choice == "3":
        return RandomAgent(words)
    print("Preparing entropy tables (the first run takes a few seconds) ...")
    return EntropyAgent(words, cache_dir=cache_dir)


def watch_solver(words: WordList, cache_dir) -> None:
    agent = choose_agent(words, cache_dir)
    secret = ask(f"Hidden word (Enter = random, must be one of the {len(words.answers)} answers): ").lower()
    game = WordleGame(words.answers, words.allowed)
    if secret and secret in words.answer_index:
        game.reset(secret=secret)
    else:
        if secret:
            print("  not in the answer list, picking a random word instead")
        game.reset()
    print(f"\n{agent.name} is solving a hidden word ... (it has {len(words.answers)} candidates to start)\n")

    # play one guess at a time so every step can be shown
    csp = WordleCSP()
    candidates = list(range(len(words.answers)))
    history: list = []
    while not game.is_over():
        agent.last_expected_bits = float("nan")
        guess = agent.choose_guess(candidates, history)
        expected = agent.last_expected_bits
        pattern = game.guess(guess)
        history.append((guess, pattern))
        csp.add_feedback(guess, pattern)
        before = len(candidates)
        candidates = [i for i in candidates if csp.is_consistent(words.answers[i])]
        after = len(candidates)
        gained = math.log2(before / after) if after else float("nan")
        exp_txt = "" if expected != expected else f"expected {expected:.2f} bits, "
        print(f"Guess {game.guesses_used}: {coloured(guess, pattern)}  "
              f"{exp_txt}gained {gained:.2f} bits, {after} candidate{'s' if after != 1 else ''} left")
        if 1 < after <= 8:
            print(f"         left: {show_candidates(words, candidates)}")
    if game.is_won():
        print(f"\nSolved in {game.guesses_used} guess{'es' if game.guesses_used != 1 else ''}!")
    else:
        print(f"\nNot solved in {game.max_guesses} guesses. The word was '{game.secret}'.")


OTHER_WORD_WEIGHT = 0.1   # how likely a word outside the original answer list is, relative to one inside it


def helper_mode(words: WordList, cache_dir) -> None:
    print("\nHelper mode: play your real Wordle and tell me what the colours were.")
    print("Enter feedback as five characters: g = green, y = yellow, x = grey (or 2 / 1 / 0).")
    print("Type 'q' at any prompt to go back to the menu.\n")
    print("Which words may be the answer?")
    print("  1  All valid words (default) - also finds answers outside the original Wordle list")
    print("  2  Only the original 2,315 answers (matches the original game)")
    wide = ask("Choice [1]: ") != "2"
    print("Preparing entropy tables (the first run takes a few seconds) ...")
    agent = EntropyAgent(words, cache_dir=cache_dir)
    csp = WordleCSP()
    in_answers = set(words.answers)
    if wide:
        pool = list(words.allowed)                      # candidate words (strings)
        weights = [1.0 if w in in_answers else OTHER_WORD_WEIGHT for w in pool]
        print(f"  {len(pool)} possible words; words outside the original answer list count "
              f"{OTHER_WORD_WEIGHT:g}x as likely.\n")
    else:
        candidates = list(range(len(words.answers)))
        pool = [words.answers[i] for i in candidates]
        weights = [1.0] * len(pool)
    history: list = []
    while True:
        if not history:
            ranked = agent.rank_guesses(list(range(len(words.answers))), top=5)   # best opening
        elif wide:
            print("  working out the best guesses ...")
            ranked = agent.rank_guesses_for(pool, weights, top=5)
        else:
            idx = [words.answer_index[w] for w in pool]
            ranked = agent.rank_guesses(idx, top=5) if len(idx) > 1 else [(pool[0], 0.0)]
        suggestion = ranked[0][0]
        print("Best guesses: " + ", ".join(f"{w} ({b:.2f} bits)" for w, b in ranked))
        guess = ask(f"Word you played [{suggestion}]: ").lower() or suggestion
        if guess == "q":
            return
        if not words.is_valid(guess):
            print(f"  '{guess}' is not a valid {WORD_LENGTH}-letter guess word, try again.")
            continue
        try:
            pattern = parse_pattern(ask("Feedback (g/y/x): "))
        except ValueError as err:
            print(f"  {err}")
            continue
        print("  " + coloured(guess, pattern))
        if pattern == WIN_PATTERN:
            print("\nSolved!")
            return
        csp.add_feedback(guess, pattern)
        before_weight = sum(weights)
        kept = [(w, wt) for w, wt in zip(pool, weights) if csp.is_consistent(w)]
        history.append((guess, pattern))
        if not kept:
            print("\nNo valid word fits all that feedback. Check the colours you entered.")
            return
        pool, weights = [w for w, _ in kept], [wt for _, wt in kept]
        gained = math.log2(before_weight / sum(weights))
        ordered = sorted(pool, key=lambda w: (w not in in_answers, w))        # original answers first
        shown = ", ".join(w if w in in_answers else w + "*" for w in ordered[:12])
        more = f", ... (+{len(pool) - 12} more)" if len(pool) > 12 else ""
        note = "   (* = outside the original answer list)" if any(w not in in_answers for w in ordered[:12]) else ""
        print(f"  {len(pool)} word{'s' if len(pool) != 1 else ''} left (gained {gained:.2f} bits): "
              f"{shown}{more}{note}\n")


def show_openers(words: WordList, cache_dir) -> None:
    print("Ranking all guesses by entropy over every possible answer ...")
    agent = EntropyAgent(words, cache_dir=cache_dir)
    ranked = agent.rank_guesses(list(range(len(words.answers))), top=10)
    print("\nBest opening guesses (expected bits of information):")
    for i, (word, bits) in enumerate(ranked, start=1):
        print(f"  {i:2d}. {word}  {bits:.3f} bits")
    print()


class ConsoleMenu:
    """The text interface: a menu that calls the solver modes above."""

    def __init__(self, words: WordList, cache_dir):
        self.words = words
        self.cache_dir = cache_dir

    def run(self) -> int:
        print(f"Wordle Solver with Constraints and Entropy "
              f"({len(self.words.answers)} answers, {len(self.words.allowed)} valid guesses)")
        while True:
            print("\n  1  Watch the solver play a hidden word\n  2  Helper for my real Wordle game"
                  "\n  3  Show the best opening guesses\n  4  Quit")
            choice = ask("Choice: ")
            if choice == "1":
                watch_solver(self.words, self.cache_dir)
            elif choice == "2":
                helper_mode(self.words, self.cache_dir)
            elif choice == "3":
                show_openers(self.words, self.cache_dir)
            elif choice in ("4", "q", "quit", "exit"):
                return 0
            else:
                print("Please enter 1, 2, 3 or 4.")


def main() -> int:
    try:
        words = WordList.load()
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1
    except ValueError as err:
        print(f"Error in the word lists: {err}")
        return 1
    return ConsoleMenu(words, Path(DEFAULT_DATA_DIR) / "processed").run()


if __name__ == "__main__":
    sys.exit(main())
