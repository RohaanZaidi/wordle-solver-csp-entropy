"""Download the official Wordle word lists into data/raw/.

    python scripts/download_words.py

Files written:
    data/raw/answers.txt  the 2,315 words the game can pick as the hidden word
    data/raw/allowed.txt  all 12,972 words the game accepts as guesses

Source: the data/ folder of github.com/deedy/wordle-solver (official_wordle_common.txt
and official_wordle_all.txt), which copied the lists from the original Wordle website.
Only the Python standard library is needed. Re-run with --answers-url / --allowed-url
if the files ever move.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/deedy/wordle-solver/main/data/"
ANSWERS_URL = BASE + "official_wordle_common.txt"
ALLOWED_URL = BASE + "official_wordle_all.txt"
EXPECTED = {"answers": 2315, "allowed": 12972}
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(url: str) -> list[str]:
    request = urllib.request.Request(url, headers={"User-Agent": "wordle-solver-csp-entropy"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    words = [line.strip().lower() for line in text.splitlines() if line.strip()]
    bad = [w for w in words if not re.fullmatch(r"[a-z]{5}", w)]
    if bad:
        raise ValueError(f"{url}: {len(bad)} lines are not 5-letter words (e.g. {bad[:3]})")
    return sorted(set(words))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers-url", default=ANSWERS_URL)
    parser.add_argument("--allowed-url", default=ALLOWED_URL)
    parser.add_argument("--out-dir", default=str(RAW_DIR))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        answers = fetch(args.answers_url)
        allowed = fetch(args.allowed_url)
    except (urllib.error.URLError, OSError, ValueError) as err:
        print(f"Download failed: {err}", file=sys.stderr)
        print("Check your internet connection, or download the two lists manually and save them "
              "as data/raw/answers.txt and data/raw/allowed.txt (one word per line).", file=sys.stderr)
        return 1

    allowed = sorted(set(allowed) | set(answers))
    (out / "answers.txt").write_text("\n".join(answers) + "\n", encoding="utf-8")
    (out / "allowed.txt").write_text("\n".join(allowed) + "\n", encoding="utf-8")
    print(f"answers.txt: {len(answers)} words   allowed.txt: {len(allowed)} words   -> {out}")
    for label, got in (("answers", len(answers)), ("allowed", len(allowed))):
        if got != EXPECTED[label]:
            print(f"Warning: expected {EXPECTED[label]} {label} but got {got}. "
                  f"The numbers in the proposal assume the official lists.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
