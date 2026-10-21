# Wordle Solver with Constraints and Entropy

AL-2002 Artificial Intelligence (Laboratory), Fall 2026 — FAST NUCES, Karachi Campus.

A Python program that plays Wordle by treating each green / yellow / grey clue as a constraint
(constraint satisfaction with backtracking, MRV and forward checking) and choosing each guess by
entropy (expected information gain). It is compared against a random-consistent baseline and a
letter-frequency agent.

## Group

| Member | Roll number | What they built |
|---|---|---|
| Leader: _name_ | _roll no._ | Simulator, word lists, CSP engine, random agent |
| Member 2: _name_ | _roll no._ | Entropy and frequency agents, experiments, charts, menu |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py               # console menu
python src/experiments.py        # reproduces the experiments (seed 42)
```

> Status: proposal stage — the source modules in `src/` are being written (see `docs/proposal.md`).
> These commands will work once the milestones in the proposal are complete.

## Planned features

- Wordle simulator with correct repeated-letter feedback
- CSP candidate filtering with backtracking, MRV and forward checking
- Entropy-based guess selection reporting bits gained per guess
- Comparison of three agents on 100 fixed target words, with charts

## Known limitations

- English five-letter words only; no connection to the live Wordle website.

## Data

Wordle answer list (2,315 words) and valid-guess list (12,972 words) from a public GitHub
repository; add the exact source link here and place the files in `data/raw/`.

## Repository layout

```
README.md          what it is, how to run it, who wrote what
requirements.txt   pinned dependencies
.gitignore
docs/              proposal and report, each as PDF and MD
src/               all source code, in modules
data/              raw/ and processed/
outputs/           figures/ and results/
```
