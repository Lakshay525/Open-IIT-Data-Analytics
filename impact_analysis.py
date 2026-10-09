"""Task 5 support: how much could a better pin plausibly change in the field?

Observation from the supplied visit logs: the farther the OLD geocoder pin was from where the agent
actually met the borrower, the longer the agent took and the more often the visit ended "address not
traceable". We turn that observed relationship into an indicative estimate of what our pins would
change, using the model's error distribution on addresses without a visit (hide-and-predict).

IMPORTANT: this is an observational mapping on synthetic data, not a causal effect and not a pilot
result. The pilot plan (pilot_plan.md) is how it would be tested properly. Everything is bootstrapped
over addresses to show the uncertainty.

Run: python impact_analysis.py   ->  results/impact_by_pin_error.csv, results/impact_estimate.csv, results/impact_by_pin_error.png
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
BINS = [0, 100, 200, 400, 800, 1e9]
LABELS = ["<100 m", "100-200 m", "200-400 m", "400-800 m", ">800 m"]
CONTACT = {"met_borrower", "cash_collected", "met_family"}


def main():
    fv = pd.read_csv(ROOT / "clean_data" / "field_visits.csv")
    base = pd.read_csv(ROOT / "clean_data" / "baseline_geocodes.csv").set_index("address_id")
    pins = pd.read_csv(ROOT / "pins.csv").set_index("address_id")
    hp = pd.read_csv(RES / "hide_and_predict.csv")

    # distance the agent had to bridge: old pin -> confirmed location (own-visit pin)
    v = pins[pins.has_own_visit]
    gap = np.hypot(v.pin_x - base.loc[v.index, "geocoder_x"], v.pin_y - base.loc[v.index, "geocoder_y"])
    fv["gap_m"] = fv.address_id.map(gap)
    fv["minutes"] = (pd.to_datetime(fv.checkin_ts) - pd.to_datetime(fv.start_ts)).dt.total_seconds() / 60
    fv["not_traceable"] = fv.outcome.eq("address_not_traceable")
    fv["productive"] = fv.outcome.isin(CONTACT)
    x = fv.dropna(subset=["gap_m"]).copy()
    x["bin"] = pd.cut(x.gap_m, BINS, labels=LABELS)
    tab = x.groupby("bin", observed=True).agg(visits=("outcome", "size"), not_traceable=("not_traceable", "mean"),
                                              productive=("productive", "mean"), mean_minutes=("minutes", "mean"),
                                              median_minutes=("minutes", "median")).reset_index()
    tab.to_csv(RES / "impact_by_pin_error.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # apply the mapping to the old vs new error distributions on addresses without a visit
    bins = pd.cut(hp.err_old, BINS, labels=LABELS)
    new_bins = pd.cut(hp.err_model, BINS, labels=LABELS)
    m = tab.set_index("bin")
    keys = ["not_traceable", "productive", "mean_minutes"]
    rng = np.random.default_rng(11)

    def expect(bin_series, idx):
        b = bin_series.iloc[idx]
        return {k: float(m.loc[b, k].astype(float).mean()) for k in keys}

    rows = []
    n = len(hp)
    boots = {"old": [], "new": []}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        # also resample the bin table (visit-level) to carry its sampling noise
        xs = x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))
        mt = xs.groupby("bin", observed=True).agg(nt=("not_traceable", "mean"), pr=("productive", "mean"), mi=("minutes", "mean"))
        for tag, series in [("old", bins), ("new", new_bins)]:
            b = series.iloc[idx]
            vals = mt.reindex(b.values)
            boots[tag].append([vals.nt.mean(), vals.pr.mean(), vals.mi.mean()])
    out = []
    for tag in ("old", "new"):
        a = np.array(boots[tag])
        out.append({"pin": tag, "not_traceable_rate": a[:, 0].mean(), "productive_share": a[:, 1].mean(), "mean_minutes_per_visit": a[:, 2].mean()})
    res = pd.DataFrame(out).set_index("pin")
    # per-agent-day: fixed working time; productive visits/day = (W / minutes per visit) * productive share
    o, nw = np.array(boots["old"]), np.array(boots["new"])
    prod_day_old = (1 / o[:, 2]) * o[:, 1]
    prod_day_new = (1 / nw[:, 2]) * nw[:, 1]
    uplift = prod_day_new / prod_day_old - 1
    nt_drop = (o[:, 0] - nw[:, 0]) / o[:, 0]
    min_drop = (o[:, 2] - nw[:, 2]) / o[:, 2]
    summary = pd.DataFrame([
        {"metric": "address-not-traceable rate (old -> new)", "old": res.at["old", "not_traceable_rate"], "new": res.at["new", "not_traceable_rate"],
         "relative_change": -np.median(nt_drop), "ci95_low": -np.quantile(nt_drop, .975), "ci95_high": -np.quantile(nt_drop, .025)},
        {"metric": "minutes per visit (old -> new)", "old": res.at["old", "mean_minutes_per_visit"], "new": res.at["new", "mean_minutes_per_visit"],
         "relative_change": -np.median(min_drop), "ci95_low": -np.quantile(min_drop, .975), "ci95_high": -np.quantile(min_drop, .025)},
        {"metric": "productive visits per agent-day (relative uplift)", "old": 1.0, "new": 1 + np.median(uplift),
         "relative_change": np.median(uplift), "ci95_low": np.quantile(uplift, .025), "ci95_high": np.quantile(uplift, .975)},
    ])
    summary.to_csv(RES / "impact_estimate.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # share of addresses currently >400 m off that our pin brings within 200 m
    far = hp[hp.err_old > 400]
    print(f"\nAmong no-visit addresses whose old pin is >400 m off ({len(far)}): "
          f"{(far.err_model <= 200).mean():.0%} end up within 200 m of the confirmed location")
    pd.DataFrame([{"far_old_pin_addresses": len(far), "share_brought_within_200m": float((far.err_model <= 200).mean())}]
                 ).to_csv(RES / "impact_far_addresses.csv", index=False)

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    xl = np.arange(len(tab))
    ax[0].bar(xl, tab.not_traceable * 100, color="#c26a3d"); ax[0].set_title("'Address not traceable' rate (%)")
    ax[1].bar(xl, tab.mean_minutes, color="#3b6ea5"); ax[1].set_title("Minutes from start to check-in")
    ax[2].bar(xl, tab.visits, color="#7a8b99"); ax[2].set_title("Visits in bucket")
    for a in ax:
        a.set_xticks(xl); a.set_xticklabels(tab["bin"], rotation=25, fontsize=8); a.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("how far the OLD geocoder pin was from where the borrower was actually met", fontsize=9)
    plt.tight_layout(); plt.savefig(RES / "impact_by_pin_error.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
