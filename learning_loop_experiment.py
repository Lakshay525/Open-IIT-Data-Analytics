"""Learning curve: how accurate are pins for addresses with NO visit as the system accumulates field visits?

This is the evidence for "every confirmed visit improves the next guess".

Design (no surveyed truth is used, no leakage):
  - 30% of visited addresses form a fixed test set T. Their own visit evidence is removed everywhere; their
    cleaned visit pin is the label (agrees with survey truth to ~8 m where both exist).
  - From the other 70%, a random fraction f of visited addresses is "known" to the model (f = 0 ... 1).
  - Predict T from text + neighbours + landmarks + old pin, score the median / P90 error.
  - Repeat over several seeds. f=0 is the cold start (no visits at all); f=1 is today's coverage.

Run: python learning_loop_experiment.py -> results/learning_curve.csv, results/learning_curve.png
"""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pin_model import Params, PinModel, load_all, ROOT, _nz

FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def main():
    feat, base, loc, lm, evid, sv, folds = load_all()
    ref = PinModel(Params.load()).fit(feat, base, evid, loc, lm)
    visited = sorted(ref._own)
    old = base.set_index("address_id")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(visited)
        n_test = int(0.3 * len(perm))
        T, pool = list(perm[:n_test]), list(perm[n_test:])
        tgt = np.array([ref._own[a][:2] for a in T], float)
        oldp = old.loc[T, ["geocoder_x", "geocoder_y"]].to_numpy(float)
        e_old = np.hypot(oldp[:, 0] - tgt[:, 0], oldp[:, 1] - tgt[:, 1])
        for f in FRACTIONS:
            keep = set(pool[: int(round(f * len(pool)))])
            ev = evid[evid.address_id.isin(keep)]
            m = PinModel(Params.load()).fit(feat, base, ev, loc, lm)
            pr = m.predict(T)
            e = np.hypot(pr.pin_x.to_numpy() - tgt[:, 0], pr.pin_y.to_numpy() - tgt[:, 1])
            rows.append({"seed": seed, "fraction_of_visits_known": f, "visited_addresses_known": len(keep),
                         "median_error_m": float(np.median(e)), "p90_error_m": float(np.quantile(e, 0.9)),
                         "within_100m": float((e <= 100).mean()), "old_geocoder_median_m": float(np.median(e_old)),
                         "old_geocoder_p90_m": float(np.quantile(e_old, 0.9))})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "learning_curve_runs.csv", index=False)
    s = d.groupby("fraction_of_visits_known").agg(
        visited_addresses_known=("visited_addresses_known", "mean"), median_error_m=("median_error_m", "mean"),
        median_sd=("median_error_m", "std"), p90_error_m=("p90_error_m", "mean"), within_100m=("within_100m", "mean"),
        old_geocoder_median_m=("old_geocoder_median_m", "mean")).reset_index()
    s.to_csv(ROOT / "results" / "learning_curve.csv", index=False)
    print(s.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].errorbar(s.visited_addresses_known, s.median_error_m, yerr=s.median_sd, color="#3b6ea5", marker="o", capsize=3, label="pin model")
    ax[0].axhline(s.old_geocoder_median_m.iloc[0], color="#999999", ls="--", label="old geocoder")
    ax[0].set_xlabel("visited addresses the system has learned from"); ax[0].set_ylabel("median error, addresses with no visit (m)")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Median error falls as visits accumulate")
    ax[1].plot(s.visited_addresses_known, s.within_100m * 100, color="#2f7d5b", marker="o")
    ax[1].set_xlabel("visited addresses the system has learned from"); ax[1].set_ylabel("share within 100 m (%)"); ax[1].grid(alpha=.3)
    ax[1].set_title("More pins land within 100 m")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(ROOT / "results" / "learning_curve.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
