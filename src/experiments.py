"""Seeded experiments: results tables and charts for all agents.

Usage (from the repository root):
    python src/experiments.py                 # 100 games per agent, seed 42
    python src/experiments.py --games 200     # more games
    python src/experiments.py --seed 7        # different target sample

What it does
    1. Draws a fixed sample of hidden words (the SAME words for every agent).
    2. Plays them with: CSP + Entropy, Letter frequency, and Random (consistent).
       The random agent is repeated with seeds 42-46 because it is the only
       stochastic agent; the other two are deterministic.
    3. Compares the CSP in "plain" and "mrv_fc" (MRV + forward checking) modes on
       the boards reached after the entropy agent's 1st and 2nd guesses.
    4. Writes tables to outputs/results/ and charts to outputs/figures/.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from agents import EntropyAgent, FrequencyAgent, RandomAgent
from csp import WordleCSP
from game import MAX_GUESSES, WordleGame
from wordlist import DEFAULT_DATA_DIR, REPO_ROOT, WordList

RANDOM_SEEDS = (42, 43, 44, 45, 46)
AGENT_ORDER = ["CSP + Entropy", "Letter frequency", "Random (consistent)"]


# --------------------------------------------------------------------------- #
def sample_targets(words: WordList, n: int, seed: int) -> list[str]:
    """The fixed sample of hidden words used for every agent."""
    n = min(n, len(words.answers))
    return random.Random(seed).sample(words.answers, n)


def run_games(agent, words: WordList, targets: list[str]):
    """Play one game per target with `agent`; return the list of GameResult."""
    game = WordleGame(words.answers, words.allowed)
    results = []
    for secret in targets:
        game.reset(secret=secret)
        results.append(agent.play(game))
    return results


def run_all_agents(words, targets, cache_dir, log):
    """-> {agent name: [list of GameResult per run]} (random agent has several runs)."""
    runs: dict[str, list] = {}
    t0 = time.perf_counter()
    entropy = EntropyAgent(words, cache_dir=cache_dir)
    runs[entropy.name] = [run_games(entropy, words, targets)]
    log(f"  {entropy.name}: done in {time.perf_counter() - t0:.1f}s")
    freq = FrequencyAgent(words)
    runs[freq.name] = [run_games(freq, words, targets)]
    log(f"  {freq.name}: done")
    runs[RandomAgent.name] = [run_games(RandomAgent(words, seed=s), words, targets)
                              for s in RANDOM_SEEDS]
    log(f"  {RandomAgent.name}: {len(RANDOM_SEEDS)} runs done (seeds {RANDOM_SEEDS[0]}-{RANDOM_SEEDS[-1]})")
    return runs


# --------------------------------------------------------------------------- #
def summarise(name: str, run_list) -> dict:
    """One summary row for an agent, pooling all of its runs."""
    games = [g for run in run_list for g in run]
    won = [g for g in games if g.won]
    penalised = [g.n_guesses if g.won else MAX_GUESSES + 1 for g in games]
    all_bits = [b for g in games for b in g.bits if not math.isnan(b)]
    per_run_means = [statistics.mean(g.n_guesses if g.won else MAX_GUESSES + 1 for g in run)
                     for run in run_list]
    return {
        "agent": name,
        "runs": len(run_list),
        "games": len(games),
        "win_rate_pct": round(100 * len(won) / len(games), 2),
        "failures": len(games) - len(won),
        "mean_guesses_won_only": round(statistics.mean(g.n_guesses for g in won), 3) if won else "",
        "mean_guesses_fail_as_7": round(statistics.mean(penalised), 3),
        "std_across_runs": round(statistics.pstdev(per_run_means), 3) if len(per_run_means) > 1 else 0.0,
        "median_guesses": statistics.median(penalised),
        "worst_case_guesses": max(penalised),
        "mean_bits_per_guess": round(statistics.mean(all_bits), 3) if all_bits else "",
        "mean_seconds_per_game": round(statistics.mean(g.seconds for g in games), 5),
        "mean_csp_checks_per_game": round(statistics.mean(g.csp_checks for g in games), 1),
    }


def csp_ablation(words: WordList, entropy_results, log) -> list[dict]:
    """Compare plain backtracking vs MRV + forward checking on real boards.

    A board is the set of constraints after the entropy agent's 1st (or 2nd) guess
    of each game. Both modes must return exactly the words CSP.filter() keeps.
    """
    boards: dict[int, list[list]] = {1: [], 2: []}
    for g in entropy_results:
        for depth in (1, 2):
            if len(g.guesses) >= depth and not (depth == len(g.guesses) and g.won):
                boards[depth].append(list(zip(g.guesses[:depth], g.patterns[:depth])))
    rows = []
    for depth, board_list in boards.items():
        if not board_list:
            continue
        for mode in ("plain", "mrv_fc"):
            stats = {"checks": [], "nodes": [], "backtracks": [], "ms": []}
            for feedback in board_list:
                csp = WordleCSP()
                for guess, pattern in feedback:
                    csp.add_feedback(guess, pattern)
                expected = sorted(csp.filter(words.answers))
                csp.reset_counters()
                t0 = time.perf_counter()
                found = csp.solve(words.answers, mode)
                elapsed = time.perf_counter() - t0
                if found != expected:
                    raise AssertionError(f"CSP mode '{mode}' disagrees with filter() on {feedback}")
                stats["checks"].append(csp.checks)
                stats["nodes"].append(csp.nodes)
                stats["backtracks"].append(csp.backtracks)
                stats["ms"].append(1000 * elapsed)
            rows.append({
                "board_after_guess": depth,
                "mode": mode,
                "boards": len(board_list),
                "mean_checks": round(statistics.mean(stats["checks"]), 1),
                "mean_nodes": round(statistics.mean(stats["nodes"]), 1),
                "mean_backtracks": round(statistics.mean(stats["backtracks"]), 1),
                "mean_ms": round(statistics.mean(stats["ms"]), 2),
            })
            log(f"  CSP {mode:7s} after guess {depth}: {rows[-1]['mean_checks']:.0f} checks, "
                f"{rows[-1]['mean_nodes']:.0f} nodes, {rows[-1]['mean_ms']:.1f} ms (avg of {len(board_list)} boards)")
    return rows


# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_games_csv(path: Path, runs: dict) -> None:
    rows = []
    for name, run_list in runs.items():
        for run_no, run in enumerate(run_list):
            seed = RANDOM_SEEDS[run_no] if name == RandomAgent.name else ""
            for g in run:
                rows.append({
                    "agent": name, "run_seed": seed, "secret": g.secret, "won": int(g.won),
                    "guesses": g.n_guesses, "guess_words": " ".join(g.guesses),
                    "feedback": " ".join("".join("xyg"[p] for p in pat) for pat in g.patterns),
                    "candidates_after": " ".join(map(str, g.candidates_after)),
                    "bits_per_guess": " ".join(f"{b:.2f}" for b in g.bits),
                    "seconds": round(g.seconds, 5),
                })
    write_csv(path, rows)


# --------------------------------------------------------------------------- #
def make_figures(runs: dict, ablation: list[dict], fig_dir: Path, n_targets: int) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in AGENT_ORDER if n in runs]
    colors = {"CSP + Entropy": "#1f3a6e", "Letter frequency": "#d98c1f", "Random (consistent)": "#8a8a8a"}
    paths = []

    def per_game_guesses(name):
        return [g.n_guesses if g.won else MAX_GUESSES + 1 for run in runs[name] for g in run]

    # 1. mean guesses per agent (95% confidence interval over games)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    means = [statistics.mean(per_game_guesses(n)) for n in names]
    errs = [1.96 * statistics.pstdev(per_game_guesses(n)) / math.sqrt(len(per_game_guesses(n))) for n in names]
    bars = ax.bar(names, means, yerr=errs, capsize=5, color=[colors[n] for n in names])
    for bar, m, e in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, m + e + 0.08, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean number of guesses (failures counted as 7)")
    ax.set_title(f"Mean guesses per agent ({n_targets} target words, error bars = 95% CI)")
    ax.set_ylim(0, max(means) + 1)
    fig.tight_layout()
    p = fig_dir / "fig1_mean_guesses.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 2. distribution of guesses needed
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [str(k) for k in range(1, MAX_GUESSES + 1)] + ["failed"]
    width = 0.8 / len(names)
    for j, n in enumerate(names):
        data = per_game_guesses(n)
        shares = [100 * sum(1 for d in data if d == k) / len(data) for k in range(1, MAX_GUESSES + 2)]
        ax.bar(np.arange(len(labels)) + j * width, shares, width, label=n, color=colors[n])
    ax.set_xticks(np.arange(len(labels)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Guesses needed")
    ax.set_ylabel("Share of games (%)")
    ax.set_title("Distribution of guesses needed")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "fig2_guess_distribution.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 3. candidates remaining after each guess
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in names:
        games = [g for run in runs[n] for g in run]
        columns = []
        for k in range(MAX_GUESSES):
            vals = [g.candidates_after[min(k, len(g.candidates_after) - 1)] for g in games]
            columns.append(statistics.mean(vals))
        ax.plot(range(1, MAX_GUESSES + 1), columns, marker="o", label=n, color=colors[n])
    ax.set_yscale("log")
    ax.set_xlabel("Guess number")
    ax.set_ylabel("Mean candidate words left (log scale)")
    ax.set_title("How quickly each agent narrows the candidates")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig3_candidates_remaining.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 4. CSP effort: plain backtracking vs MRV + forward checking
    if ablation:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        depths = sorted({r["board_after_guess"] for r in ablation})
        x = np.arange(len(depths))
        for ax, key, title in ((axes[0], "mean_checks", "Constraint checks"),
                               (axes[1], "mean_nodes", "Search nodes")):
            for j, (mode, label, col) in enumerate((("plain", "Plain backtracking", "#8a8a8a"),
                                                    ("mrv_fc", "MRV + forward checking", "#1f3a6e"))):
                vals = [next(r[key] for r in ablation if r["mode"] == mode and r["board_after_guess"] == d)
                        for d in depths]
                ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=label, color=col)
            ax.set_xticks(x)
            ax.set_xticklabels([f"after guess {d}" for d in depths])
            ax.set_yscale("log")
            ax.set_title(title + " per solve (log scale)")
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
        fig.tight_layout()
        p = fig_dir / "fig4_csp_effort.png"
        fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
class Experiment:
    """Runs N seeded games per agent, then saves the CSV tables and the charts."""

    def __init__(self, words: WordList, n_games: int = 100, seed: int = 42,
                 data_dir=None, out_dir=None, log=print):
        self.words = words
        self.n_games = n_games
        self.seed = seed
        self.cache_dir = Path(data_dir if data_dir is not None else DEFAULT_DATA_DIR) / "processed"
        self.out_dir = Path(out_dir if out_dir is not None else REPO_ROOT / "outputs")
        self.log = log

    def run(self) -> dict:
        words, log = self.words, self.log
        targets = sample_targets(words, self.n_games, self.seed)
        log(f"Answers: {len(words.answers)}   valid guesses: {len(words.allowed)}   "
            f"target sample: {len(targets)} words, seed {self.seed}")
        log("Building pattern matrix (first run only, then cached) and playing games ...")
        t0 = time.perf_counter()
        runs = run_all_agents(words, targets, self.cache_dir, log)
        summary = [summarise(n, runs[n]) for n in AGENT_ORDER if n in runs]
        log("\nResults")
        for row in summary:
            log(f"  {row['agent']:22s} win {row['win_rate_pct']:6.2f}%   mean guesses "
                f"{row['mean_guesses_fail_as_7']:.3f} (won only {row['mean_guesses_won_only']})   "
                f"bits/guess {row['mean_bits_per_guess']}")
        log("\nCSP: plain backtracking vs MRV + forward checking")
        ablation = csp_ablation(words, runs["CSP + Entropy"][0], log)
        results_dir = self.out_dir / "results"
        write_csv(results_dir / "summary.csv", summary)
        write_games_csv(results_dir / "games.csv", runs)
        if ablation:
            write_csv(results_dir / "csp_ablation.csv", ablation)
        figures = make_figures(runs, ablation, self.out_dir / "figures", len(targets))
        log(f"\nWrote {len(summary)} summary rows, games.csv, csp_ablation.csv and {len(figures)} figures "
            f"to {self.out_dir}  (total {time.perf_counter() - t0:.1f}s)")
        return {"runs": runs, "summary": summary, "ablation": ablation}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Wordle solver experiments.")
    parser.add_argument("--games", type=int, default=100, help="number of hidden words (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the target sample (default 42)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="folder holding raw/ word lists")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"), help="folder for results/ and figures/")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    try:
        words = WordList.load(args.data_dir)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    log(f"Wordle solver experiments  (python {platform.python_version()}, numpy {np.__version__})")
    Experiment(words, args.games, args.seed, args.data_dir, out_dir, log).run()
    results_dir = out_dir / "results"
    (results_dir / "experiment_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
