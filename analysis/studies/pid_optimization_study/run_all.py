"""
Run All PID Optimization Sub-Studies
=====================================
Orchestrates all three sub-studies in the correct order:

  Step 1 — Proxy-based box scan  (box_scan_proxy.py)
  Step 2 — Fit-based box scan    (fit_based_scan.py)
  Step 3 — MVA with PID variants (mva_pid_study.py)
  Step 4 — Comparison & summary  (compare_results.py)

Usage
-----
  # Run all steps for both track categories:
  uv run python run_all.py

  # Run specific steps:
  uv run python run_all.py --steps proxy fit    # only steps 1 and 2
  uv run python run_all.py --category LL        # only Lambda-LL

  # Skip heavy fit-based scan (use cached results if available):
  uv run python run_all.py --steps proxy mva compare
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
SCRIPTS = STUDY_DIR / "scripts"

STEP_MAP = {
    "proxy": SCRIPTS / "box_scan_proxy.py",
    "fit": SCRIPTS / "fit_based_scan.py",
    "mva": SCRIPTS / "mva_pid_study.py",
    "compare": SCRIPTS / "compare_results.py",
}

ALL_STEPS = ["proxy", "fit", "mva", "compare"]


def run_step(name: str, script: Path, category: str) -> bool:
    print(f"\n{'='*70}")
    print(f"STEP: {name.upper()}  [{category}]")
    print(f"{'='*70}")
    t0 = time.time()
    cmd = [sys.executable, str(script), "--category", category]
    ret = subprocess.run(cmd, cwd=STUDY_DIR)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n[ERROR] Step '{name}' failed (exit code {ret.returncode}).")
        return False
    print(f"\n[OK] Step '{name}' completed in {elapsed:.1f} s.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all PID optimization sub-studies")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--category",
        choices=["LL", "DD", "both"],
        default="both",
        help="Track category (default: both)",
    )
    parser.add_argument(
        "--skip-on-failure",
        action="store_true",
        help="Continue to next step even if a step fails",
    )
    args = parser.parse_args()

    t_start = time.time()
    print(f"PID Optimization Study — running steps: {args.steps}")
    print(f"Category: {args.category}")

    failures = []
    for step in args.steps:
        ok = run_step(step, STEP_MAP[step], args.category)
        if not ok:
            failures.append(step)
            if not args.skip_on_failure:
                print("\nAborting. Use --skip-on-failure to continue past errors.")
                sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    if failures:
        print(f"Completed with failures in steps: {failures}")
    else:
        print("All steps completed successfully.")
    print(f"Total wall time: {total:.1f} s")
    print("\nOutputs written to:")
    print(f"  {STUDY_DIR / 'output' / 'box_proxy'}")
    print(f"  {STUDY_DIR / 'output' / 'fit_based'}")
    print(f"  {STUDY_DIR / 'output' / 'mva'}")
    print(f"  {STUDY_DIR / 'output' / 'comparison'}")


if __name__ == "__main__":
    main()
