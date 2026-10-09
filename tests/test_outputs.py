"""Checks for calibration, directions and the final deliverables.  Run: python -m tests.test_outputs"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from geocoder.paths import DATA, OUT, RES
from geocoder import scoring as ev_mod
from geocoder.confidence import bearing_label



def test_schema_and_monotonicity():
    p = pd.read_csv(OUT / "predictions.csv")
    need = {"address_id", "pin_x", "pin_y", "radius_50", "radius_90", "directions", "confidence_tier"}
    assert need.issubset(p.columns)
    assert len(p) == 2880 and p.address_id.is_unique
    assert np.isfinite(p[["pin_x", "pin_y", "radius_50", "radius_90"]].to_numpy()).all()
    assert (p.radius_50 > 0).all() and (p.radius_90 >= p.radius_50).all() and (p.radius_80.between(p.radius_50, p.radius_90)).all()
    assert set(p.confidence_tier) <= {"high", "medium", "low"}
    order = p.groupby("confidence_tier").radius_90.median()
    assert order["high"] < order["medium"] < order["low"]
    assert p.directions.str.len().min() > 10
    print("PASS schema: 2,880 rows, radii ordered, tiers ordered by radius")


def test_calibration_on_heldout():
    c = pd.read_csv(RES / "calibration_coverage.csv")
    o = c[(c.dimension == "overall") & (c.nominal == 0.9)].iloc[0]
    assert o.ci95_low <= 0.90 <= o.ci95_high, f"90% radius coverage {o.coverage:.2f} CI excludes 0.9"
    o5 = c[(c.dimension == "overall") & (c.nominal == 0.5)].iloc[0]
    assert o5.coverage >= 0.45, "50% radius under-covers"
    for g in ("novisit", "visit"):
        r = c[(c.dimension == "group") & (c.segment == g) & (c.nominal == 0.9)].iloc[0]
        assert r.ci95_low <= 0.90 <= r.ci95_high, f"group {g}: coverage {r.coverage:.2f} CI {r.ci95_low:.2f}-{r.ci95_high:.2f}"
    q = pd.read_csv(RES / "confidence_tier_quality.csv").set_index("tier")
    assert q.within_100["high"] > q.within_100["medium"] > q.within_100["low"], "tiers must rank accuracy"
    print(f"PASS calibration: 90% radius covers {o.coverage:.0%} (CI {o.ci95_low:.0%}-{o.ci95_high:.0%}), "
          f"50% radius {o5.coverage:.0%}; tiers rank accuracy")


def test_directions_are_geometrically_correct():
    p = pd.read_csv(OUT / "predictions.csv")
    lm = pd.read_csv(DATA / "landmarks_poi.csv")
    pat = re.compile(r"(\d+) m (N|NE|E|SE|S|SW|W|NW) of ([A-Za-z ]+?)(?: \(([A-Za-z ]+) side\))?(?:;|\.|$)")
    ok = tot = 0
    for r in p.sample(600, random_state=1).itertuples():
        m = pat.search(r.directions)
        if not m:
            continue
        dist, comp, name = int(m.group(1)), m.group(2), m.group(3).strip()
        cand = lm[(lm.town_id == r.town_id) & (lm.name == name)]
        tot += 1
        for c in cand.itertuples():
            d = np.hypot(r.pin_x - c.x, r.pin_y - c.y)
            if abs(d - dist) <= max(30, 0.1 * dist) and bearing_label(r.pin_x - c.x, r.pin_y - c.y) == comp:
                ok += 1
                break
    assert tot > 300 and ok / tot > 0.99, f"direction/geometry mismatch {ok}/{tot}"
    print(f"PASS directions: {ok}/{tot} sampled first clauses match the landmark distance and compass bearing")


def test_ps2_and_offline():
    ps2 = pd.read_csv(OUT / "ps2_location_confidence.csv")
    assert len(ps2) == 2880 and ps2.location_confidence.between(0, 1).all()
    p = pd.read_csv(OUT / "predictions.csv").set_index("address_id")
    flagged = ps2[ps2.hard_to_find_flag == 1]
    assert (p.loc[flagged.address_id, "confidence_tier"] == "low").all() and not p.loc[flagged.address_id, "has_own_visit"].any()
    # confidence should rank accuracy on surveyed addresses
    d = pd.read_csv(RES / "calibration_surveyed_detail.csv").set_index("address_id")
    d["conf"] = ps2.set_index("address_id").loc[d.index, "location_confidence"]
    hi, lo = d[d.conf >= d.conf.median()].error_m.median(), d[d.conf < d.conf.median()].error_m.median()
    assert hi < lo
    towns = sorted((OUT / "offline_pack").glob("T*.json"))
    assert len(towns) == 3
    total = 0
    for t in towns:
        pack = json.loads(t.read_text(encoding="utf-8"))
        assert pack["addresses"] and pack["landmarks"]
        total += t.stat().st_size
    print(f"PASS PS2 export and offline packs ({total/1e3:.0f} kB for 3 towns, plain JSON, no network needed)")


def test_predictions_score_with_shared_eval():
    p = pd.read_csv(OUT / "predictions.csv")
    _, truth, meta = ev_mod.baseline_inputs()
    _, t = ev_mod.score_predictions(p.rename(columns={"pin_x": "pred_x", "pin_y": "pred_y"})[["address_id", "pred_x", "pred_y"]], truth, meta)
    o = t[t.dimension == "overall"].iloc[0]
    assert o.median_error_m < 80
    print(f"PASS shared eval.py on predictions.csv: median {o.median_error_m:.1f} m, P90 {o.p90_error_m:.0f} m")


if __name__ == "__main__":
    test_schema_and_monotonicity()
    test_calibration_on_heldout()
    test_directions_are_geometrically_correct()
    test_ps2_and_offline()
    test_predictions_score_with_shared_eval()
    print("all output checks passed")
