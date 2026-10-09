"""One command that rebuilds the whole PS3 solution from the raw CSVs in clean_data/.

    python run_pipeline.py            # full run (about 3-4 minutes)
    python run_pipeline.py --fast     # skip the slow optional steps (integrity stress test, re-tuning, learning curve, notebooks)
    python run_pipeline.py --retune   # also refit the pin-model sigmas on visit pseudo-labels

Stages (each writes its outputs; later stages read earlier ones):
  1 validate    validate_inputs.py          data audit               -> results/file_quality.csv, validation_checks.csv
  2 score       eval.py                     baseline + fixed folds   -> results/baseline_scores.csv, folds.csv
  3 evidence    extract_evidence.py         Task 2 visit cleaning    -> evidence.csv
  4 integrity   integrity_analysis.py       fake-visit stress test   -> results/integrity_*.csv
  5 parse       address_parser.py           multilingual parser      -> results/address_features.csv
  6 tune        tune_pin_model.py           (optional) refit sigmas  -> results/pin_model_params.json
  7 pins        run_task3.py                Task 3 pin model + CV    -> pins.csv, results/pin_model_*.csv
  8 confidence  confidence_directions.py    Task 4                   -> predictions.csv, ps2_location_confidence.csv, offline_pack/
  9 impact      impact_analysis.py, learning_loop_experiment.py    -> results/impact_*.csv, learning_curve.csv
 10 present     make_figures.py, build_demo.py, make_summary.py    -> demo_map.html, results/fig_*.png, SUMMARY.md
 11 tests       test_task3.py, test_task4.py, verify_package.py
 12 notebooks   create_notebooks.py, create_task3_notebook.py, create_task4_notebook.py
Guards: folds.csv and evidence.csv must not change silently (their hashes are compared before/after).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    """Hash of the table CONTENT (not bytes), so line endings / float formatting cannot trigger a false alarm."""
    if not p.exists():
        return "-"
    import pandas as pd
    df = pd.read_csv(p)
    h = pd.util.hash_pandas_object(df.round(6).astype(str), index=False).values.tobytes()
    return hashlib.sha256(h + ",".join(df.columns).encode()).hexdigest()[:12]


def run(script: str, *args: str) -> None:
    t = time.time()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, script, *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {script} (exit {r.returncode})")
    print(f"--- done in {time.time() - t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--retune", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    folds_before, ev_before = sha(ROOT / "folds.csv"), sha(ROOT / "evidence.csv")

    run("validate_inputs.py")
    run("eval.py")
    run("extract_evidence.py")
    if not a.fast:
        run("integrity_analysis.py")
    run("address_parser.py")
    if a.retune:
        run("tune_pin_model.py")
    run("run_task3.py", "--skip-parse")
    run("confidence_directions.py")
    run("impact_analysis.py")
    run("pilot_power.py")
    if not a.fast:
        run("learning_loop_experiment.py")
    run("make_figures.py")
    run("build_demo.py")
    run("make_summary.py")
    run("test_task3.py")
    run("test_task4.py")
    if not a.fast:
        run("create_notebooks.py")
        run("create_task3_notebook.py")
        run("create_task4_notebook.py")
        run("verify_package.py")

    folds_after, ev_after = sha(ROOT / "folds.csv"), sha(ROOT / "evidence.csv")
    print(f"\nfolds.csv   {folds_before} -> {folds_after}  {'UNCHANGED' if folds_before in ('-', folds_after) else 'CHANGED!'}")
    print(f"evidence.csv {ev_before} -> {ev_after}  {'UNCHANGED' if ev_before in ('-', ev_after) else 'CHANGED (rerun Task 3+ from here)'}")
    print(f"\nPipeline finished in {time.time() - t0:.0f}s. Open demo_map.html; deliverables: pins.csv, predictions.csv, "
          f"ps2_location_confidence.csv, offline_pack/, SUMMARY.md")


if __name__ == "__main__":
    main()
