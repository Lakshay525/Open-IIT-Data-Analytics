"""Rebuild the whole solution from the raw CSVs in data/ with one command.

    python run_pipeline.py            # everything (about 5 minutes)
    python run_pipeline.py --fast     # skip the slow optional steps (stress test, learning curve, notebook)
    python run_pipeline.py --retune   # also refit the pin-model sigmas on visit pseudo-labels

Stages
  validation   data audit                         -> outputs/results/file_quality.csv, validation_checks.csv
  scoring      baseline scores and fixed folds    -> outputs/folds.csv, results/baseline_scores.csv
  evidence     one trusted point per visit        -> outputs/evidence.csv
  integrity    fake-visit stress test (optional)  -> outputs/results/integrity_*.csv
  parser       multilingual address parser        -> outputs/results/address_features.csv
  tuning       fit sigmas (optional)              -> outputs/results/pin_model_params.json
  evaluate     pin model, CV, ablation            -> outputs/pins.csv
  confidence   calibrated radii, tiers, directions-> outputs/predictions.csv, ps2_location_confidence.csv, offline_pack/
  analysis     impact, learning curve, pilot power-> outputs/results/impact_*.csv, learning_curve.*, pilot_power.csv
  present      figures, demo map, summary         -> outputs/demo_map.html, summary.md
  notebook     analysis notebook (optional)       -> notebooks/analysis.ipynb
  tests        tests/*.py
Guard: folds.csv and evidence.csv must not change silently (content hashes compared before/after).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def content_hash(p: Path) -> str:
    """Hash of the table content (not bytes), so line endings or float formatting cannot cause a false alarm."""
    if not p.exists():
        return "-"
    import pandas as pd
    df = pd.read_csv(p)
    h = pd.util.hash_pandas_object(df.round(6).astype(str), index=False).values.tobytes()
    return hashlib.sha256(h + ",".join(df.columns).encode()).hexdigest()[:12]


def run(module: str, *args: str) -> None:
    t = time.time()
    print(f"\n=== {module} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {module} (exit {r.returncode})")
    print(f"--- done in {time.time() - t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--retune", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    before = {n: content_hash(OUT / n) for n in ("folds.csv", "evidence.csv")}

    run("geocoder.validation")
    run("geocoder.scoring")
    run("geocoder.evidence")
    if not a.fast:
        run("geocoder.integrity")
    run("geocoder.parser")
    if a.retune:
        run("geocoder.tuning")
    run("geocoder.evaluate", "--skip-parse")
    run("geocoder.confidence")
    run("geocoder.analysis")
    run("geocoder.present")
    if not a.fast:
        run("geocoder.notebook")
    for t in ("tests.test_data", "tests.test_model", "tests.test_outputs"):
        if a.fast and t == "tests.test_data":
            continue  # needs the stress-test output
        run(t)

    for n, h in before.items():
        now = content_hash(OUT / n)
        print(f"{n:13s} {h} -> {now}  {'UNCHANGED' if h in ('-', now) else 'CHANGED (rerun from the pin model onward)'}")
    print(f"\nFinished in {time.time() - t0:.0f}s. Open outputs/demo_map.html. Deliverables: outputs/pins.csv, "
          "predictions.csv, ps2_location_confidence.csv, offline_pack/, summary.md")


if __name__ == "__main__":
    main()
