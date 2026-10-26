"""Build everything the live web app needs, from the pipeline outputs.

Writes into web/ (a self-contained Vercel project):
  web/api/_lib/geocoder/   the model, parser and confidence modules used by the API (copied, with a read-only paths stub)
  web/api/_lib/data/       the small tables the API needs to fit and query the model
  web/public/data/         static JSON for the front end (per-town addresses, statistics, chart data)

Run after the pipeline:  python -m geocoder.webassets
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from .paths import DATA, OUT, RES, ROOT

WEB = ROOT / "web"
LIB = WEB / "api" / "_lib"
PUB = WEB / "public" / "data"

TOWN_NAMES = {"T1": "Kaveripura", "T2": "Devgarh Nagar", "T3": "Navanagara East"}

PATHS_STUB = '''"""Read-only paths for the serverless API (the project folder is read-only on Vercel)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA
RES = DATA
'''


def copy_engine() -> None:
    dst = LIB / "geocoder"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.py", "parser.py", "confidence.py"):
        shutil.copy(ROOT / "geocoder" / name, dst / name)
    (dst / "__init__.py").write_text('"""Engine modules used by the API (copied from the pipeline by geocoder.webassets)."""\n', encoding="utf-8")
    (dst / "paths.py").write_text(PATHS_STUB, encoding="utf-8")
    data = LIB / "data"
    data.mkdir(parents=True, exist_ok=True)
    files = {
        RES / "address_features.csv": "address_features.csv",
        OUT / "evidence.csv": "evidence.csv",
        OUT / "pin_model_params.json": "pin_model_params.json",
        RES / "calibration_quantiles.csv": "calibration_quantiles.csv",
        DATA / "baseline_geocodes.csv": "baseline_geocodes.csv",
        DATA / "localities.csv": "localities.csv",
        DATA / "landmarks_poi.csv": "landmarks_poi.csv",
        DATA / "towns.csv": "towns.csv",
        DATA / "surveyed_addresses.csv": "surveyed_addresses.csv",
    }
    for src, name in files.items():
        shutil.copy(src, data / name)
    addr = pd.read_csv(DATA / "addresses.csv")[["address_id", "town_id", "address_text"]]
    addr.to_csv(data / "addresses.csv", index=False)


def _r(x, n=1):
    return None if pd.isna(x) else round(float(x), n)


def public_data() -> None:
    PUB.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(OUT / "predictions.csv")
    addr = pd.read_csv(DATA / "addresses.csv").set_index("address_id")
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")
    lm = pd.read_csv(DATA / "landmarks_poi.csv")
    loc = pd.read_csv(DATA / "localities.csv")
    towns = pd.read_csv(DATA / "towns.csv")

    index = []
    for town, g in pred.groupby("town_id"):
        rows = []
        for r in g.itertuples():
            d = {"id": r.address_id, "x": _r(r.pin_x), "y": _r(r.pin_y), "ox": _r(r.old_pin_x), "oy": _r(r.old_pin_y),
                 "r5": _r(r.radius_50), "r9": _r(r.radius_90), "t": r.confidence_tier[0],
                 "a": r.suggested_action, "d": r.directions, "dl": r.directions_local if isinstance(r.directions_local, str) else "",
                 "w": r.reason_codes, "m": r.method_used, "s": addr.at[r.address_id, "address_text"], "v": int(r.has_own_visit)}
            if r.address_id in sv.index:
                tx, ty = float(sv.at[r.address_id, "surveyed_x"]), float(sv.at[r.address_id, "surveyed_y"])
                d["tx"], d["ty"] = round(tx, 1), round(ty, 1)
                d["eo"] = round(float(np.hypot(r.old_pin_x - tx, r.old_pin_y - ty)), 1)
                d["en"] = round(float(np.hypot(r.pin_x - tx, r.pin_y - ty)), 1)
            rows.append(d)
        payload = {"town_id": town, "name": TOWN_NAMES.get(town, town), "addresses": rows,
                   "landmarks": lm[lm.town_id == town][["name", "landmark_type", "x", "y"]].round(1).to_dict("records"),
                   "localities": loc[loc.town_id == town][["locality_name", "centroid_x", "centroid_y"]].round(1).to_dict("records")}
        (PUB / f"town_{town}.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        index.append({"town_id": town, "name": TOWN_NAMES.get(town, town), "n": len(rows)})

    S = json.loads((RES / "summary_metrics.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(RES / "pin_model_oof_errors.csv")
    base = pd.read_csv(DATA / "baseline_geocodes.csv").set_index("address_id")
    old = np.hypot(base.geocoder_x - sv.surveyed_x, base.geocoder_y - sv.surveyed_y).loc[oof.address_id].to_numpy()
    hp = pd.read_csv(RES / "hide_and_predict.csv")

    def cdf(v, n=70):
        v = np.sort(np.asarray(v, float))
        qs = np.linspace(0.01, 1, n)
        return [[round(float(np.quantile(v, q)), 1), round(float(q), 3)] for q in qs]

    ab = pd.read_csv(RES / "pin_model_ablation.csv")
    lc = pd.read_csv(RES / "learning_curve.csv")
    rel = pd.read_csv(RES / "calibration_reliability.csv")
    imp = pd.read_csv(RES / "impact_by_pin_error.csv")
    cov = pd.read_csv(RES / "calibration_coverage.csv")
    tq = pd.read_csv(RES / "confidence_tier_quality.csv")
    tier_counts = pred.confidence_tier.value_counts().to_dict()
    stats = {
        "surveyed": S["surveyed"], "hide_and_predict": S["hide_and_predict"], "calibration": S["calibration"],
        "tiers": {"counts": tier_counts, "quality": S["tiers"]["quality"]},
        "cdf_surveyed_old": cdf(old), "cdf_surveyed_new": cdf(oof.error_m), "cdf_hp_old": cdf(hp.err_old), "cdf_hp_new": cdf(hp.err_model),
        "ablation": [{"variant": v[2:].strip(), "median": round(float(m), 1), "p90": round(float(p), 1)}
                     for v, m, p in zip(ab.variant, ab.all100_median, ab.all100_p90)],
        "learning": [{"n": int(a), "median": round(float(b), 1), "within100": round(float(c), 3)}
                     for a, b, c in zip(lc.visited_addresses_known, lc.median_error_m, lc.within_100m)],
        "learning_old": round(float(lc.old_geocoder_median_m.iloc[0]), 1),
        "reliability": [{"curve": k, "points": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(g.nominal, g.observed)], "n": int(g.n.iloc[0])}
                        for k, g in rel.groupby("curve")],
        "impact": [{"bin": b, "not_traceable": round(float(n), 3), "minutes": round(float(m), 1), "visits": int(v)}
                   for b, n, m, v in zip(imp["bin"], imp.not_traceable, imp.mean_minutes, imp.visits)],
        "coverage": cov[(cov.dimension.isin(["overall", "group"])) & (cov.nominal.isin([0.5, 0.9]))][["segment", "n", "nominal", "coverage", "ci95_low", "ci95_high"]].round(3).to_dict("records"),
        "tier_quality": tq.round(3).to_dict("records"),
        "n_addresses": int(len(pred)), "n_visits": 5578, "n_surveyed": 100, "n_landmarks": int(len(lm)),
        "towns": index,
    }
    bf = RES / "benchmark_estimators.csv"
    if bf.exists():
        b = pd.read_csv(bf)
        stats["benchmark"] = {ts: {"n": int(g.n.iloc[0]), "rows": [{"name": r.estimator, "median": round(float(r.median_m), 1), "p90": round(float(r.p90_m), 1)}
                                                                    for r in g.itertuples()]}
                              for ts, g in b.groupby("test_set")}
    (PUB / "stats.json").write_text(json.dumps(stats, separators=(",", ":")), encoding="utf-8")


def demo_scenarios() -> None:
    """Pick real held-out addresses for the demo: locate-examples (with true location) and what-if visit replays.

    Only addresses that have no visit of their own are used, so the live API never sees their own location."""
    import sys
    sys.path.insert(0, str(LIB))
    import engine  # the same code the API runs

    c = engine._load()
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")
    feat = c["feat"].set_index("address_id")
    own = set(pd.read_csv(OUT / "evidence.csv").query("eligible_for_pin").address_id)
    cand = [a for a in sv.index if a not in own and a in feat.index and pd.notna(feat.at[a, "loc_id"])]
    rng = np.random.default_rng(11)
    examples, scenarios = [], []
    for town in ("T1", "T2", "T3"):
        ids = [a for a in cand if feat.at[a, "town_id"] == town]
        rng.shuffle(ids)
        for a in ids[:2]:
            r = engine.locate(c["addr"].at[a, "address_text"], town)
            t = sv.loc[a]
            examples.append({"id": a, "town": town, "text": c["addr"].at[a, "address_text"], "truth": [round(float(t.surveyed_x), 1), round(float(t.surveyed_y), 1)],
                             "old_error": round(float(np.hypot(r["old_pin"]["x"] - t.surveyed_x, r["old_pin"]["y"] - t.surveyed_y)), 1),
                             "new_error": round(float(np.hypot(r["pin"]["x"] - t.surveyed_x, r["pin"]["y"] - t.surveyed_y)), 1)})
        best = []
        for a in ids:
            t = sv.loc[a]
            w = engine.what_if_visit(a, float(t.surveyed_x), float(t.surveyed_y))
            if "error" not in w and w["neighbours_moved"] >= 3:
                best.append((w["neighbours_moved"], a, w))
        best.sort(key=lambda b: -b[0])
        for n, a, w in best[:2]:
            t = sv.loc[a]
            scenarios.append({"id": a, "town": town, "text": c["addr"].at[a, "address_text"], "visit": [round(float(t.surveyed_x), 1), round(float(t.surveyed_y), 1)],
                              "neighbours_moved": n, "locality": w.get("locality")})
    (PUB / "demo_cases.json").write_text(json.dumps({"examples": examples, "scenarios": scenarios}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"demo cases: {len(examples)} examples, {len(scenarios)} what-if scenarios")


def main() -> None:
    copy_engine()
    public_data()
    demo_scenarios()
    size = sum(p.stat().st_size for p in (WEB / "api").rglob("*") if p.is_file()) / 1e6
    print(f"web assets written: api bundle {size:.1f} MB, public data {sum(p.stat().st_size for p in PUB.glob('*.json'))/1e6:.1f} MB")


if __name__ == "__main__":
    main()
