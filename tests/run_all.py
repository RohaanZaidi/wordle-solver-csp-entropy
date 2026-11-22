"""Run every test without needing pytest:  python tests/run_all.py"""
import importlib
import sys
import time
import traceback
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent / "src"))

passed = failed = 0
for path in sorted(here.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        start = time.perf_counter()
        try:
            getattr(module, name)()
            passed += 1
            print(f"PASS  {path.stem}.{name}  ({time.perf_counter() - start:.2f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {path.stem}.{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
