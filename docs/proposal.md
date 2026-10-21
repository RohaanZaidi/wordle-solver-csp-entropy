<table>
<colgroup>
<col style="width: 11%" />
<col style="width: 88%" />
</colgroup>
<tbody>
<tr class="odd">
<td><img src="/home/claude/wordle-solver-csp-entropy/docs/media/image1.png" title="FAST logo" style="width:0.64583in;height:0.64583in" alt="FAST NUCES logo" /></td>
<td><p><strong>FAST — NATIONAL UNIVERSITY OF COMPUTER AND EMERGING SCIENCES</strong></p>
<p>FAST School of Computing · Karachi Campus · Department of Artificial Intelligence &amp; Data Science</p>
<p><em>AL-2002 Artificial Intelligence (Laboratory) · Fall 2026</em></p></td>
</tr>
</tbody>
</table>

**PROJECT PROPOSAL**

*Semester Lab Project · Weightage 1 mark*

## Group Information

|            | **Name** | **Roll Number** | **Section** | **Role in the project**           |
|------------|----------|-----------------|-------------|-----------------------------------|
| **Leader** |          |                 |             | Simulator, CSP engine, repository |
| Member 2   |          |                 |             | Agents, experiments, interface    |

|                           |                                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Project Title**         | Wordle Solver with Constraints and Entropy                                      |
| **Project Idea**          | Idea No. \_\_\_\_\_\_ from the Project Ideas list ☐ Own idea (needs approval) ☐ |
| **Category**              | Part A — Game ☒ Part B — Applied / Data ☐                                       |
| **GitHub Repository URL** |                                                                                 |
| **Date of Submission**    |                                                                                 |

# 1. Project Title and Summary

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Wordle Solver with Constraints and Entropy is a Python program that plays Wordle, the game in which a hidden five-letter word must be found in six guesses using green, yellow and grey feedback. Every piece of feedback is treated as a set of constraints on the hidden word, and a constraint satisfaction solver keeps only the words that could still be correct. The program then chooses each guess by entropy: it plays the word whose feedback is expected to reveal the most information about the answer. Players can watch it solve a puzzle or enter their own feedback to get a hint. A plain program can filter words but cannot tell which of thousands of legal guesses is smartest; entropy gives it that ability, and the project measures how much better it plays than random guessing. |

# 2. Problem Statement and Motivation

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| In Wordle the hidden word comes from a list of 2,315 answers, but about 13,000 five-letter words are accepted as guesses. Each guess returns one of 3^5 = 243 colour patterns, so information is partial, and clues interact: a repeated letter can be marked grey if the answer contains it only once, and a yellow letter limits where it may appear. Choosing the best first guess by brute force means scoring 12,972 guesses against 2,315 answers, about 30 million feedback computations. Players rely on intuition and favourite openers, and often waste guesses on words that barely separate the candidates. Solving it properly shows how constraints and information theory combine to decide under uncertainty. |

# 3. Problem Formulation

## 3.1 PEAS Description

| **Component**           | **Description for your project**                                                                                                                                              |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Performance measure** | Games solved within six guesses (%), average and worst-case guesses, bits gained per guess, time per move.                                                                    |
| **Environment**         | A simulated Wordle game: a hidden word drawn (seeded) from 2,315 answers, 12,972 valid guesses, a limit of six guesses, and green / yellow / grey feedback after every guess. |
| **Actuators**           | Submitting one five-letter guess word per turn from the valid-guess list.                                                                                                     |
| **Sensors**             | The green / yellow / grey feedback pattern for each guess, and the number of guesses used.                                                                                    |

## 3.2 Environment and Agent Type

| **Property**      | **Classification — tick one**                                         |
|-------------------|-----------------------------------------------------------------------|
| Observability     | Fully observable ☐ Partially observable ☒                             |
| Determinism       | Deterministic ☒ Stochastic ☐                                          |
| Change over time  | Static ☒ Dynamic ☐                                                    |
| State and actions | Discrete ☒ Continuous ☐                                               |
| Agents            | Single-agent ☒ Multi-agent / adversarial ☐                            |
| Agent type        | Simple reflex ☐ Model-based ☐ Goal-based ☐ Utility-based ☒ Learning ☐ |

## 3.3 Problem Representation

| **Element**                              | **Your representation**                                                                                                                                                                                 |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **State representation**                 | Candidate answers still consistent with all feedback, the history of (guess, pattern) pairs and the guesses used; the CSP view holds a letter domain for each of the five positions plus letter counts. |
| **Actions / moves**                      | Guess any word from the valid-guess list (about 13,000 words); it need not be a possible answer.                                                                                                        |
| **Initial state**                        | All 2,315 answers are candidates, no guesses made, hidden word drawn with a seeded generator.                                                                                                           |
| **Goal or terminal test**                | Win when the feedback is all green (five greens); loss when six guesses have been used without a win.                                                                                                   |
| **Cost, utility, fitness or evaluation** | Utility of a guess = expected information gain, the entropy H = −Σ p(k) log2 p(k) over its possible feedback patterns k on the remaining candidates (bits). Each guess costs 1.                         |

# 4. Proposed AI Approach

| **\#** | **Technique**                                                             | **Role in the project**                                                                                                   | **Own code or library**    |
|--------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------|
| 1      | Constraint satisfaction: backtracking with MRV and forward checking       | Turns each clue into constraints on the five positions and removes words that violate them; counts checks and backtracks. | Own code                   |
| 2      | Information theory: entropy-based guess selection                         | Scores each guess by the entropy of the feedback it would give on the remaining candidates and plays the best.            | Own code (NumPy for speed) |
| 3      | Greedy letter-frequency heuristic                                         | Second comparison: plays the consistent word with the most common letters among the candidates.                           | Own code                   |
| B      | Baseline: random guess among the words still consistent with all feedback | Shows what the CSP filter achieves with no smart choice.                                                                  | Own code                   |

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Wordle feedback is exactly a set of constraints, so a CSP is the natural way to keep only words that could still be correct. Hundreds of words may remain, so one must be chosen; entropy answers in plain terms: prefer the guess whose colour patterns split the remaining words most evenly, because that removes the most uncertainty. If pattern k appears for a fraction p(k) of the remaining words, the guess is worth the sum of −p(k) log2 p(k) bits; ties favour guesses that are possible answers. |

## 4.1 Planned Class Design

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p>WordList — answers, allowed; load(), is_valid(word); specific errors for bad files.</p>
<p>WordleGame (environment) — secret, guesses_used; reset(seed), guess(word) giving the pattern (handles repeated letters), is_won().</p>
<p>WordleCSP — position domains, letter counts, check counters; add_feedback(), is_consistent(), filter() using backtracking, MRV and forward checking.</p>
<p>Agent (abstract base) — choose_guess(), play(game). Subclasses override choose_guess: RandomAgent, FrequencyAgent, EntropyAgent (adds entropy()).</p>
<p>Experiment — runs N seeded games per agent, saves CSV and charts; ConsoleMenu is the interface.</p></td>
</tr>
</tbody>
</table>

# 5. Scope

**In scope — the features you commit to delivering.** List the features that must work for the project to be considered complete.

| **\#** | **Feature**                                                                                    | **Why it is essential**                              |
|--------|------------------------------------------------------------------------------------------------|------------------------------------------------------|
| 1      | Wordle simulator with exact feedback rules, including repeated letters                         | Same reproducible environment for all agents.        |
| 2      | CSP model with backtracking, MRV, forward checking and counters                                | Required CSP technique; keeps only valid candidates. |
| 3      | Entropy-based agent reporting bits gained per guess                                            | The core AI technique and source of the main result. |
| 4      | Random-consistent baseline and letter-frequency agent on the same 100 target words, fixed seed | The required comparison against alternatives.        |
| 5      | Results table, three or more charts, and a console menu                                        | Shows how well the AI works; easy to run.            |

**Out of scope — deliberately excluded.** Naming what you will not build is as important as naming what you will; it shows that your scope is realistic for a group of two.

|                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A graphical Wordle clone or web deployment; the live Wordle website; deep or reinforcement learning; an exhaustive optimal-tree solver; other languages or word lengths. |

# 6. Data or Environment Source

|                            |                                                                                                                                |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Source name and URL**    | Wordle answer list (2,315 words) and valid-guess list (12,972 words) from a public GitHub repository; URL cited in the README. |
| **Type**                   | Public dataset ☒ Web API ☐ Self-collected ☐ Generated / simulated ☒ (seeded games)                                             |
| **Access requirements**    | No key needed ☒ Free key required ☐ (Paid sources are not permitted)                                                           |
| **Approximate size**       | About 15,000 words in two plain-text files, under 100 KB in total.                                                             |
| **Licence / terms of use** | Public word lists, no personal data; cited in the README.                                                                      |
| **Offline fallback**       | Files committed to data/raw/, so no network is needed; missing file raises FileNotFoundError.                                  |

> ***Required:*** your program must still run when the network is unavailable, by reading a locally cached copy of any data it downloads. Describe your caching plan in the last row above.

# 7. Evaluation Plan

|                                     |                                                                       |
|-------------------------------------|-----------------------------------------------------------------------|
| **What will be compared**           | Random, frequency and entropy agents; CSP with and without MRV.       |
| **Metric(s)**                       | Win rate; mean, median, worst-case guesses; bits per guess; time.     |
| **Baseline**                        | Random guess among the words still consistent with all feedback.      |
| **Test instances / test set**       | 100 fixed hidden words from the 2,315 answers, same for all agents.   |
| **Number of runs and seed**         | 100 games per agent, seed 42 (random agent: seeds 42–46).             |
| **Planned charts (at least three)** | Mean guesses; guess histogram; candidates left per guess; CSP checks. |

# 8. Division of Work

| **Member** | **Modules / components owned**                          | **Approximate share** |
|------------|---------------------------------------------------------|-----------------------|
| Leader     | Simulator, WordList, WordleCSP, RandomAgent, README     | 50%                   |
| Member 2   | EntropyAgent, FrequencyAgent, experiments, charts, menu | 50%                   |

# 9. Milestones and Timeline

| **\#** | **Milestone — what will be demonstrably working**                             | **Target date** | **Owner** |
|--------|-------------------------------------------------------------------------------|-----------------|-----------|
| 1      | Simulator passes repeated-letter tests; RandomAgent plays 100 games.          | 1 Nov 2026      | Leader    |
| 2      | CSP filter matches brute-force filtering on all 100 words.                    | 8 Nov 2026      | Leader    |
| 3      | EntropyAgent solves all 100 test words.                                       | 15 Nov 2026     | Member 2  |
| 4      | All agents run from one command; table and two charts produced.               | 22 Nov 2026     | Member 2  |
| 5      | Experiments complete, final integration, documentation and repository cleanup | 29 Nov 2026     | Both      |

# 10. Expected Outcome and Risks

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p>Running the main program opens a console menu: watch the solver play a hidden word, or type in the colours from a real game and get the next best guess with candidates left and bits gained. One command reproduces the results and four charts (seed 42). We expect entropy to solve nearly every word in six guesses, with clearly fewer guesses on average than the other agents.</p>
<p>Risk 1: scoring 13,000 guesses is slow; we will cache first-move scores and use NumPy. Risk 2: repeated-letter bugs; we will unit-test and compare the CSP with brute-force filtering.</p></td>
</tr>
</tbody>
</table>

| **For office use — Lab Instructor** |                                                        |
|-------------------------------------|--------------------------------------------------------|
| Proposal status                     | Approved ☐ Approved with changes ☐ Revision required ☐ |
| Comments                            |                                                        |
| Signature and date                  |                                                        |

*End of Project Proposal*
