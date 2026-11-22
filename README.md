# Wordle Solver with Constraints and Entropy

AL-2002 Artificial Intelligence (Laboratory), Fall 2026 — FAST NUCES, Karachi Campus.

A Python program that plays Wordle. Every green / yellow / grey clue is treated as a **constraint**
on the hidden word (constraint satisfaction with backtracking, MRV and forward checking), and each
guess is chosen by **entropy**: the program plays the word whose feedback is expected to tell it the
most about the answer. It is compared with a random-consistent baseline and a letter-frequency agent.

## Group

| Member | Roll number | What they built |
|---|---|---|
| Leader: _name_ | _roll no._ | `game.py`, `wordlist.py`, `csp.py`, `RandomAgent`, README, repository |
| Member 2: _name_ | _roll no._ | `EntropyAgent`, `FrequencyAgent`, `experiments.py`, `main.py` |

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_words.py # downloads the official word lists into data/raw/
```

## Run

```bash
python src/main.py               # console menu: watch the solver, or get help with a real game
python src/experiments.py        # reproduces every table and chart (seed 42, 100 games per agent)
python tests/run_all.py          # unit tests (no extra packages needed)
```

`experiments.py` options: `--games N` (number of hidden words), `--seed S`.
The first run builds a 30 MB pattern table (about 5-10 seconds) and caches it in `data/processed/`
(ignored by Git); later runs reuse it.

### Console menu

1. **Watch the solver** — pick an agent and a hidden word and see each guess in colour with the
   bits of information gained and the candidates left.
2. **Helper** — type the word you played in the real game and its colours (`g` green, `y` yellow,
   `x` grey), and get the best next guesses. By default every one of the 12,972 valid words may be
   the answer (words outside the original 2,315-word answer list count 0.1x as likely, a tunable
   assumption in `src/main.py`), so it still works when the real game's answer is not in the
   original list; those words are marked `*`. Choose option 2 to use only the original answers.
3. **Best openers** — the ten opening guesses with the highest entropy.

## How it works

| Module | Role |
|---|---|
| `src/game.py` | `compute_pattern` (exact Wordle feedback, repeated letters included), `WordleGame` environment, vectorised pattern matrix |
| `src/wordlist.py` | `WordList`: loads and validates the answer list and the guess list |
| `src/csp.py` | `WordleCSP`: clues as constraints; `filter()` for fast pruning, `solve()` with plain backtracking or MRV + forward checking |
| `src/agents.py` | `Agent` base class; `RandomAgent`, `FrequencyAgent`, `EntropyAgent` override `choose_guess()` |
| `src/experiments.py` | `Experiment`: seeded games, CSV tables, four charts |
| `src/main.py` | `ConsoleMenu`: the text interface |
| `tests/` | unit tests: feedback rules, CSP equals brute force, entropy formula, agents |

**Entropy.** For a guess *g*, the remaining candidates split into groups by the feedback pattern *k*
*g* would produce. If pattern *k* covers a fraction *p(k)* of the candidates, the guess is worth
`H(g) = − Σ p(k) · log2 p(k)` bits. The agent plays the guess with the largest `H`; ties favour words
that are themselves possible answers.

**CSP.** The five letter positions are the variables and their letters are the domains. Green fixes
a letter; yellow and grey remove it from that position; the counts of green/yellow letters give a
minimum number of copies, and a grey copy caps the number of copies. `solve()` enumerates the words
that satisfy all of this, and `tests/test_csp.py` checks that both search modes return exactly the
same words as brute-force feedback comparison. During play the agents use the direct `filter()` check
because it is faster; the search modes are compared in the experiments.

## Outputs

After `python src/experiments.py`:

- `outputs/results/summary.csv` — win rate, mean / median / worst-case guesses, bits per guess, time
- `outputs/results/games.csv` — every game: hidden word, guesses, feedback, candidates left
- `outputs/results/csp_ablation.csv` — plain backtracking vs MRV + forward checking
- `outputs/results/experiment_log.txt` — the run log
- `outputs/figures/` — `fig1_mean_guesses.png`, `fig2_guess_distribution.png`,
  `fig3_candidates_remaining.png`, `fig4_csp_effort.png`

Notes on reading the results: mean guesses count a failed game as 7 (a "won only" column is also
given). The random agent is repeated with seeds 42-46 because it is the only stochastic agent. In the
CSP comparison, forward checking visits far fewer search nodes but does more work per node, so it can
show a *higher* number of constraint checks; both numbers are reported.

## Data

The official Wordle lists: 2,315 possible answers and 12,972 accepted guesses. `scripts/download_words.py`
fetches them from the `data/` folder of [github.com/deedy/wordle-solver](https://github.com/deedy/wordle-solver)
(`official_wordle_common.txt` and `official_wordle_all.txt`), which copied them from the original game.
They are saved as `data/raw/answers.txt` and `data/raw/allowed.txt` and committed, so the program
never needs the network. If the download fails, save the two lists manually under those names
(one lowercase word per line).

## Known limitations

- English five-letter words only; no connection to the live Wordle website.
- The experiments assume the hidden word comes from the original 2,315-word answer list. Helper mode
  can look beyond it (default), but a word outside all 12,972 valid words would still not be found.
- The 0.1 weight for words outside the original answer list is an assumption, not a measured value.
- The word lists are from the original Wordle; today's live game may differ slightly.
- Greedy one-step entropy is not the provably optimal strategy (that needs a full game-tree search).

## Repository layout

```
README.md          what it is, how to run it, who wrote what
requirements.txt   pinned dependencies
.gitignore
docs/              proposal and report, each as PDF and MD
src/               all source code, in modules
scripts/           download_words.py
tests/             unit tests
data/              raw/ and processed/
outputs/           figures/ and results/
```
