"""Fit the pin-model sigmas and gate on leave-one-out *pseudo-labels*.

Pseudo-label = an address's own cleaned field-visit pin (about 8-10 m from survey
truth where both exist). For each visited address we hide its own visit and ask the
model to predict it from neighbours, landmarks and the old pin.

No surveyed truth is used here, so the 100 surveyed addresses stay clean for
evaluation. To report an honest number for the tuning step itself, we also run a
2-fold check: tune on half of the localities, score on the other half.

Run:  python tune_pin_model.py   ->  results/pin_model_params.json, results/tuning_report.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pin_model import Params, PinModel, fuse, load_all, _nz, ROOT

DELTA = 120.0  # Huber scale (m): robust, but still rewards pulling the bulk in


def huber(e: np.ndarray) -> float:
    return float(np.mean(np.where(e < DELTA, 0.5 * e ** 2 / DELTA, e - 0.5 * DELTA)))


def pack(p: Params) -> np.ndarray:
    return np.log(np.array([getattr(p, k) for k in Params.TUNABLE], float))


def unpack(x: np.ndarray, base: Params) -> Params:
    q = Params(**{k: getattr(base, k) for k in base.__dataclass_fields__ if k != "enabled"})
    for k, v in zip(Params.TUNABLE, np.exp(x)):
        setattr(q, k, float(v))
    q.gate_k = float(np.clip(q.gate_k, 1.5, 8.0))
    q.c_n = float(np.clip(q.c_n, 0.0, 5.0))
    q.b_scatter = float(np.clip(q.b_scatter, 0.0, 3.0))
    return q


def tune(C, tgt, init: Params, mask=None):
    sel = np.ones(len(tgt), bool) if mask is None else mask
    sub = type(C)(C.mu[sel], C.s[sel], C.n[sel], C.valid[sel], C.tier[sel], C.gate_scale[sel])
    t = tgt[sel]

    def obj(x):
        q = unpack(x, init)
        F, _, _ = fuse(sub, q)
        # light ridge toward the default so tiny strata can't blow a sigma up
        reg = 0.002 * float(np.sum((x - pack(init)) ** 2))
        return huber(np.linalg.norm(F - t, axis=1)) + reg

    best = None
    x0 = pack(init)
    for start in range(3):
        xs = x0 + (np.random.default_rng(start).normal(0, 0.15, len(x0)) if start else 0)
        r = minimize(obj, xs, method="Powell", options={"maxiter": 4000, "xtol": 1e-2, "ftol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x, init)


def summarize(name, C, tgt, p):
    F, _, _ = fuse(C, p)
    e = np.linalg.norm(F - tgt, axis=1)
    return {"set": name, "n": len(e), "median_m": np.median(e), "p90_m": np.quantile(e, 0.9),
            "within100": (e < 100).mean()}


def main():
    feat, base, loc, lm, ev, sv, folds = load_all()
    model = PinModel(Params()).fit(feat, base, ev, loc, lm)  # visit pins only: no surveyed truth
    ids = [a for a in model.known_ids if _nz(model.feat.at[a, "loc_id"])]
    C = model.build_candidates(ids, hide_own=True)
    tgt = model.known_xy[[model.known_row[a] for a in ids]]
    loc_ids = model.feat.loc[ids, "loc_id"].to_numpy()
    uniq = np.array(sorted(set(loc_ids)))
    rng = np.random.default_rng(7)
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2])
    inA = np.array([l in half for l in loc_ids])

    rows = []
    default = Params()
    rows.append({**summarize("all (old pin only)", C, tgt, Params(enabled=("baseline",))), "params": "baseline"})
    rows.append({**summarize("all (default sigmas)", C, tgt, default), "params": "default"})
    # 2-fold honesty check
    for tag, train_mask in [("A->B", inA), ("B->A", ~inA)]:
        p = tune(C, tgt, default, train_mask)
        test_mask = ~train_mask
        sub = type(C)(C.mu[test_mask], C.s[test_mask], C.n[test_mask], C.valid[test_mask], C.tier[test_mask], C.gate_scale[test_mask])
        rows.append({**summarize(f"held-out localities ({tag})", sub, tgt[test_mask], p), "params": "tuned on other half"})
        sub0 = type(C)(C.mu[test_mask], C.s[test_mask], C.n[test_mask], C.valid[test_mask], C.tier[test_mask], C.gate_scale[test_mask])
        rows.append({**summarize(f"same half, default ({tag})", sub0, tgt[test_mask], default), "params": "default"})
    final = tune(C, tgt, default)
    # calibrate the posterior spread: choose `inflate` so ~90% of errors fall within 2.146 x raw_spread
    # (the 90% radius of an isotropic 2-D Gaussian with per-axis sigma = raw_spread).
    final.inflate = 1.0
    F, sp, _ = fuse(C, final)
    ratio = np.linalg.norm(F - tgt, axis=1) / sp
    final.inflate = float(np.quantile(ratio, 0.9) / 2.146)
    F, sp, _ = fuse(C, final)
    ratio = np.linalg.norm(F - tgt, axis=1) / sp
    print(f"inflate={final.inflate:.2f}; share of errors within 1.18x / 2.15x raw_spread: "
          f"{(ratio <= 1.177).mean():.2f} / {(ratio <= 2.146).mean():.2f} (ideal 0.50 / 0.90); "
          f"Spearman(spread, error)={pd.Series(sp).corr(pd.Series(np.linalg.norm(F - tgt, axis=1)), method='spearman'):.2f}")
    rows.append({**summarize("all (tuned on all)", C, tgt, final), "params": "tuned (in-sample)"})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    out.to_csv(ROOT / "results" / "tuning_report.csv", index=False)
    final.save(ROOT / "results" / "pin_model_params.json")
    print("\nfitted params:")
    for k in Params.TUNABLE:
        print(f"  {k:12s} default {getattr(default, k):8.2f}  fitted {getattr(final, k):8.2f}")


if __name__ == "__main__":
    main()
