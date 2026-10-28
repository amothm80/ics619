"""Run every step in order, from feature selection to the significance rankings, on one data source.

Each step is its own script, run as a subprocess with DATA_SOURCE set; its output streams to the
terminal and to artifacts/<source>/logs/<step>.log. The run stops at the first step that fails.
Steps that cache their work (sweep.py, stacking.py) reuse it, so a stopped run can resume with --start.
Usage: python run_pipeline.py [TICKER ...] [--source S] [--start STEP] [--units N ...] [--folds K]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ["kaggle", "yfinance_recent"]  # keys of config.SOURCES
STEPS = ["data", "select_features", "preprocessing", "metrics", "sweep", "ensembles", "stacking", "baseline", "rankings"]
TAKES_TICKERS = {"sweep", "ensembles", "stacking", "baseline", "rankings"}  # the others always run all ten


def run(step: str, args: list[str], log_dir: Path) -> None:
    """Run one script, tee its output to log_dir/<step>.log, exit if it fails."""
    log = log_dir / f"{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", f"{step}.py", *args]
    print(f"\n=== {' '.join(cmd[2:])}  (log: {log.relative_to(ROOT)})")
    start = time.time()
    with open(log, "w") as f, subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
    if proc.returncode:
        sys.exit(f"=== {step} failed (exit {proc.returncode}); see {log}")
    print(f"=== {step} done in {(time.time() - start) / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tickers", nargs="*", help="default: all ten")
    p.add_argument("--source", choices=SOURCES, default=os.environ.get("DATA_SOURCE", "kaggle"))
    p.add_argument("--start", choices=STEPS, default=STEPS[0], help="skip the steps before this one")
    p.add_argument("--units", type=int, nargs="+", help="sweep.py unit grid (default: its own)")
    p.add_argument("--folds", type=int, help="stacking.py folds (default: its own)")
    args = p.parse_args()

    os.environ["DATA_SOURCE"] = args.source  # read by config.py here and in every step
    import config

    if unknown := set(args.tickers) - set(config.TICKERS):
        p.error(f"unknown tickers: {sorted(unknown)}")
    print(f"source {config.SOURCE}: artifacts in {config.ARTIFACTS_DIR.relative_to(ROOT)}")
    for step in STEPS[STEPS.index(args.start):]:
        extra = list(args.tickers) if step in TAKES_TICKERS else []
        if step == "sweep" and args.units:
            extra += ["--units", *map(str, args.units)]
        if step == "stacking" and args.folds:
            extra += ["--folds", str(args.folds)]
        run(step, extra, config.ARTIFACTS_DIR / "logs")
    print("\n=== pipeline finished")


if __name__ == "__main__":
    main()
