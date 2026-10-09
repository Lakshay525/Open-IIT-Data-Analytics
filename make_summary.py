"""Collect every headline number into results/summary_metrics.json and SUMMARY.md (single source of truth for the
documentation - nothing is typed by hand). Run last:  python make_summary.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def main():
    ab = pd.read_csv(RES / "pin_model_ablation.csv")
    sc = pd.read_csv(RES / "pin_model_scores.csv")
    base = pd.read_csv(RES / "baseline_scores.csv")
    cov = pd.read_csv(RES / "calibration_coverage.csv")
    tq = pd.read_csv(RES / "confidence_tier_quality.csv").set_index("tier")
    hp = pd.read_csv(RES / "hide_and_predict.csv")
    pred = pd.read_csv(ROOT / "predictions.csv")
    lc = pd.read_csv(RES / "learning_curve.csv")
    imp = pd.read_csv(RES / "impact_estimate.csv")
    far = pd.read_csv(RES / "impact_far_addresses.csv").iloc[0]
    dchk = pd.read_csv(RES / "directions_checks.csv").iloc[0]
    sv = pd.read_csv(ROOT / "clean_data" / "surveyed_addresses.csv").set_index("address_id")
    p = pred.set_index("address_id")

    full = ab[ab.variant.str.startswith("5")].iloc[0]
    old = ab[ab.variant.str.startswith("0")].iloc[0]
    row = lambda df, dim, seg: df[(df.dimension == dim) & (df.segment == seg)].iloc[0]
    o_all = row(sc, "overall", "all")
    oo = row(base, "overall", "all")
    vis, unv = row(sc, "visit_status", "visited"), row(sc, "visit_status", "unvisited")
    ov, uv = row(base, "visit_status", "visited"), row(base, "visit_status", "unvisited")
    c90 = cov[(cov.dimension == "overall") & (cov.nominal == 0.9)].iloc[0]
    c50 = cov[(cov.dimension == "overall") & (cov.nominal == 0.5)].iloc[0]

    eo = np.hypot(p.loc[sv.index, "old_pin_x"] - sv.surveyed_x, p.loc[sv.index, "old_pin_y"] - sv.surveyed_y)
    en = np.hypot(p.loc[sv.index, "pin_x"] - sv.surveyed_x, p.loc[sv.index, "pin_y"] - sv.surveyed_y)
    tiers = pred.confidence_tier.value_counts().to_dict()
    S = {
        "surveyed": {
            "n": 100, "old_median": float(oo.median_error_m), "old_p90": float(oo.p90_error_m),
            "new_median": float(o_all.median_error_m), "new_p90": float(o_all.p90_error_m),
            "new_median_ci": [float(o_all.median_ci95_low_m), float(o_all.median_ci95_high_m)],
            "old_within": [float(oo[f"share_within_{r}m"]) for r in (50, 100, 250)],
            "new_within": [float(o_all[f"share_within_{r}m"]) for r in (50, 100, 250)],
            "visited": {"n": int(vis.n), "old_median": float(ov.median_error_m), "new_median": float(vis.median_error_m)},
            "unvisited": {"n": int(unv.n), "old_median": float(uv.median_error_m), "new_median": float(unv.median_error_m),
                          "old_p90": float(uv.p90_error_m), "new_p90": float(unv.p90_error_m)},
            "dev85": {"old_median": float(ab.iloc[0].dev85_median), "new_median": float(full.dev85_median),
                      "old_p90": float(ab.iloc[0].dev85_p90), "new_p90": float(full.dev85_p90)},
            "test15": {"old_median": float(old.test15_median), "new_median": float(full.test15_median),
                       "old_p90": float(old.test15_p90), "new_p90": float(full.test15_p90)},
            "share_worse_than_old_by_25m": float((en > eo + 25).mean()),
            "share_better_than_old_by_25m": float((en < eo - 25).mean()),
        },
        "ablation": [{"variant": r.variant, "median": float(r.all100_median), "p90": float(r.all100_p90)} for r in ab.itertuples()],
        "hide_and_predict": {
            "n": int(len(hp)), "old_median": float(hp.err_old.median()), "new_median": float(hp.err_model.median()),
            "old_p90": float(hp.err_old.quantile(.9)), "new_p90": float(hp.err_model.quantile(.9)),
            "old_within100": float((hp.err_old <= 100).mean()), "new_within100": float((hp.err_model <= 100).mean()),
            "share_worse_than_old_by_25m": float((hp.err_model > hp.err_old + 25).mean()),
            "by_precision": {k: {"n": int(len(g)), "old": float(g.err_old.median()), "new": float(g.err_model.median())}
                             for k, g in hp.groupby("precision")},
        },
        "calibration": {"cov90": float(c90.coverage), "cov90_ci": [float(c90.ci95_low), float(c90.ci95_high)], "cov50": float(c50.coverage),
                        "cov50_ci": [float(c50.ci95_low), float(c50.ci95_high)]},
        "tiers": {"counts": tiers, "share_low": tiers.get("low", 0) / len(pred),
                  "quality": {k: {"n": int(v.n), "median_err": float(v.median_err), "within_100": float(v.within_100), "within_250": float(v.within_250)} for k, v in tq.iterrows()}},
        "learning_curve": [{"visited_known": int(r.visited_addresses_known), "median": float(r.median_error_m), "within100": float(r.within_100m)} for r in lc.itertuples()],
        "impact": {r.metric: {"old": float(r.old), "new": float(r.new), "relative": float(r.relative_change), "ci": [float(r.ci95_low), float(r.ci95_high)]} for r in imp.itertuples()},
        "far_old_pin": {"n": int(far.far_old_pin_addresses), "share_within_200m": float(far.share_brought_within_200m)},
        "directions": {k: (float(v) if not isinstance(v, str) else v) for k, v in dchk.items()},
        "coverage_of_evidence": {"addresses_with_visit_pin": int(p.has_own_visit.sum()), "total": int(len(p))},
    }
    (RES / "summary_metrics.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    s = S["surveyed"]
    md = f"""# Headline numbers (generated by make_summary.py)

| | old geocoder | ours |
|---|---:|---:|
| 100 surveyed, median / P90 (m) | {s['old_median']:.0f} / {s['old_p90']:.0f} | **{s['new_median']:.0f} / {s['new_p90']:.0f}** |
| within 50 / 100 / 250 m | {s['old_within'][0]:.0%} / {s['old_within'][1]:.0%} / {s['old_within'][2]:.0%} | {s['new_within'][0]:.0%} / {s['new_within'][1]:.0%} / {s['new_within'][2]:.0%} |
| visited ({s['visited']['n']}) median | {s['visited']['old_median']:.0f} | {s['visited']['new_median']:.0f} |
| unvisited ({s['unvisited']['n']}) median / P90 | {s['unvisited']['old_median']:.0f} / {s['unvisited']['old_p90']:.0f} | {s['unvisited']['new_median']:.0f} / {s['unvisited']['new_p90']:.0f} |
| hide-and-predict ({S['hide_and_predict']['n']}) median / P90 | {S['hide_and_predict']['old_median']:.0f} / {S['hide_and_predict']['old_p90']:.0f} | {S['hide_and_predict']['new_median']:.0f} / {S['hide_and_predict']['new_p90']:.0f} |

Calibration: 90% radius covers {S['calibration']['cov90']:.0%} (95% CI {S['calibration']['cov90_ci'][0]:.0%}-{S['calibration']['cov90_ci'][1]:.0%}); 50% radius covers {S['calibration']['cov50']:.0%}.
Tiers: {tiers}. Surveyed pins worse than the old pin by >25 m: {s['share_worse_than_old_by_25m']:.0%}; better by >25 m: {s['share_better_than_old_by_25m']:.0%}.
"""
    (ROOT / "SUMMARY.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
