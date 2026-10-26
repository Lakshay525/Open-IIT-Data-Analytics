"""Benchmark: is the fusion model actually better than simpler or more standard alternatives?

Every estimator below is fed the *same* clues (the candidate tables built by `PinModel`) and is scored on
the same two test sets, so differences come from the estimator, not from the data.

  surveyed   the 100 surveyed addresses, out-of-fold (same frozen folds as evaluate.py)
  visited    the visited addresses, with their own visit hidden; target = their cleaned visit pin
             (no surveyed truth involved, about 12x more labels)

Estimators
  old pin               the supplied geocoder
  locality centroid     centre of the parsed locality
  street-key median     weighted median of known pins on the same street key (visit first)
  GP street numbers     per-locality Gaussian-process regression of position on street numbers (kriging)
  gradient boosting     histogram gradient-boosted trees on the parsed address features (the "just use ML" baseline)
  equal-weight clues    mean of all available clues, no precision weighting, no gate
  fusion, no gate       inverse-variance fusion without the robust gate
  fusion + gate         the shipped model

Also written: sensitivity of the shipped model to its settings (all sigmas scaled together, gate threshold)
and an error taxonomy (where the model beats or loses to the old pin).

Run:  python -m geocoder.benchmark   ->  outputs/results/benchmark_*.csv, benchmark.png
"""

from __future__ import annotations

from .paths import RES

import os
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .model import Candidates, Params, PinModel, SLOT_IDX, fuse, load_all, _nz

STREET_ORDER = ["lmcm", "lmbr", "cm", "br", "gali"]
SIGMA_KEYS = ("sb_street", "sb_locality", "sb_pincode", "a_cm", "a_br", "a_gali", "a_lmcm", "a_lmbr", "a_lm",
              "a_locgrp", "a_loccent", "a_poi", "f_grid", "a_visit")


def _copy(p: Params, **kw) -> Params:
    q = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__ if k != "enabled"})
    q.enabled = p.enabled
    for k, v in kw.items():
        setattr(q, k, v)
    return q


def _first_valid(C: Candidates, i: int, order: list[str]) -> np.ndarray:
    """Visit first, then the first available slot in `order`, then the old pin."""
    for s in ["visit"] + order + ["baseline"]:
        j = SLOT_IDX[s]
        if C.valid[i, j]:
            return C.mu[i, j]
    return C.mu[i, SLOT_IDX["baseline"]]


def gp_street_numbers(model: PinModel, a: str, ex) -> np.ndarray | None:
    """Per-locality Gaussian-process regression of (x, y) on street numbers, with a linear mean.

    This is the standard geostatistical competitor to the robust local-linear grid clue. The length scale is
    picked by leave-one-out error over a small grid; noise is fixed at 20 m."""
    f = model.feat.loc[a]
    l = f.loc_id
    if not _nz(l):
        return None
    best = None
    for fsname, cols, _h in (("cm", ["cross", "main"], None), ("br", ["blocknum", "road"], None)):
        if any(not _nz(f.get(c)) for c in cols):
            continue
        G = model.grid.get((l, fsname))
        if G is None:
            continue
        rows, X = G
        if ex is not None:
            keep = rows != ex
            rows, X = rows[keep], X[keep]
        if len(rows) < 6:
            continue
        Y = model.known_xy[rows]
        x0 = np.array([float(f[c]) for c in cols])
        A = np.c_[np.ones(len(X)), X]
        coef = np.linalg.lstsq(A, Y, rcond=None)[0]
        R = Y - A @ coef
        sd = X.std(0) + 1e-9
        Z = X / sd
        z0 = x0 / sd
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2)
        d0 = ((Z - z0) ** 2).sum(1)
        amp2 = max(float((R ** 2).sum(1).mean()), 100.0)
        pick = None
        for ell in (0.5, 1.0, 2.0, 4.0):
            K = amp2 * np.exp(-0.5 * D2 / ell ** 2) + 20.0 ** 2 * np.eye(len(X))
            try:
                Ki = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                continue
            loo = (Ki @ R) / np.diag(Ki)[:, None]
            score = float(np.median(np.linalg.norm(loo, axis=1)))
            if pick is None or score < pick[0]:
                pick = (score, ell, Ki)
        if pick is None:
            continue
        _, ell, Ki = pick
        k0 = amp2 * np.exp(-0.5 * d0 / ell ** 2)
        mean = np.r_[1.0, x0] @ coef
        pred = mean + k0 @ Ki @ R
        var = amp2 - float(k0 @ Ki @ k0)
        if best is None or var < best[1]:
            best = (pred, var)
    return None if best is None else best[0]


ML_COLS = ["cross", "main", "blocknum", "road", "gali", "ward"]


def ml_features(model: PinModel, ids: list[str]) -> np.ndarray:
    """Parsed address fields plus the locality centroid and the old pin: what a generic ML model would be given."""
    f = model.feat.loc[ids]
    b = model.base.loc[ids]
    lc = model.loc.loc[f.loc_id.to_numpy()]
    tier = b.precision.map({t: i for i, t in enumerate(["rooftop", "street", "locality", "pincode"])}).to_numpy(float)
    lm = f.lm_type.astype("category").cat.codes.to_numpy(float)
    X = np.c_[f[ML_COLS].to_numpy(float), lc.centroid_x.to_numpy(float), lc.centroid_y.to_numpy(float),
              b.geocoder_x.to_numpy(float), b.geocoder_y.to_numpy(float), tier, lm]
    return X, np.c_[lc.centroid_x.to_numpy(float), lc.centroid_y.to_numpy(float)]


def gb_fit_predict(model: PinModel, train: list[str], test: list[str]) -> np.ndarray:
    """Train on known locations `train`, predict `test`. Target = offset from the locality centroid."""
    Xtr, ctr = ml_features(model, train)
    Xte, cte = ml_features(model, test)
    Ytr = model.known_xy[[model.known_row[a] for a in train]] - ctr
    out = np.zeros((len(test), 2))
    for d in range(2):
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10,
                                          loss="absolute_error", random_state=0).fit(Xtr, Ytr[:, d])
        out[:, d] = g.predict(Xte)
    return out + cte


def estimators(model: PinModel, ids: list[str], hide_own: bool, params: Params, gb: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Predictions (N, 2) of every estimator for addresses `ids` that have a parsed locality."""
    C = model.build_candidates(ids, hide_own=hide_own)
    N = len(ids)
    out = {k: np.zeros((N, 2)) for k in ["old pin", "locality centroid", "street-key median", "GP street numbers",
                                          "gradient boosting", "equal-weight clues", "fusion, no gate", "fusion + gate (shipped)"]}
    out["old pin"] = C.mu[:, SLOT_IDX["baseline"]].copy()
    for i, a in enumerate(ids):
        ex = model.known_row.get(a)
        out["locality centroid"][i] = C.mu[i, SLOT_IDX["loccent"]] if C.valid[i, SLOT_IDX["loccent"]] else out["old pin"][i]
        out["street-key median"][i] = _first_valid(C, i, STREET_ORDER + ["lm", "loccent"])
        has_visit = C.valid[i, SLOT_IDX["visit"]]
        g = None if has_visit else gp_street_numbers(model, a, ex)
        out["GP street numbers"][i] = C.mu[i, SLOT_IDX["visit"]] if has_visit else (g if g is not None else _first_valid(C, i, ["loccent"]))
        if gb is not None:
            out["gradient boosting"][i] = C.mu[i, SLOT_IDX["visit"]] if has_visit else gb[i]
        out["equal-weight clues"][i] = C.mu[i][C.valid[i]].mean(0)
    out["fusion, no gate"] = fuse(C, _copy(params, gate_k=1e9))[0]
    out["fusion + gate (shipped)"] = fuse(C, params)[0]
    return out


def score_table(errs: dict[str, np.ndarray], label: str) -> pd.DataFrame:
    rows = []
    for k, e in errs.items():
        rows.append({"test_set": label, "estimator": k, "n": len(e), "median_m": np.median(e), "p90_m": np.quantile(e, .9),
                     "within_50m": (e <= 50).mean(), "within_100m": (e <= 100).mean(), "within_250m": (e <= 250).mean()})
    return pd.DataFrame(rows)


def run_surveyed(feat, base, loc, lm, evid, sv, folds, params):
    errs: dict[str, list] = {}
    meta = []
    for f in range(0, 6):
        test = folds.loc[folds.fold == f, "address_id"].tolist()
        train = folds.loc[(folds.fold != f) & (folds.fold != 0), "address_id"].tolist()
        m = PinModel(params).fit(feat, base, evid, loc, lm, truth=sv[sv.address_id.isin(train)])
        ids = [a for a in test if _nz(m.feat.at[a, "loc_id"])]
        if not ids:
            continue
        train_known = [a for a in m.known_ids if a not in set(ids) and _nz(m.feat.at[a, "loc_id"])]
        gb = gb_fit_predict(m, train_known, ids)
        est = estimators(m, ids, hide_own=False, params=params, gb=gb)
        T = sv.set_index("address_id").loc[ids, ["surveyed_x", "surveyed_y"]].to_numpy()
        for k, P in est.items():
            errs.setdefault(k, []).append(np.linalg.norm(P - T, axis=1))
    return {k: np.concatenate(v) for k, v in errs.items()}


def run_visited(feat, base, loc, lm, evid, params):
    m = PinModel(params).fit(feat, base, evid, loc, lm)
    ids = [a for a in m.known_ids if a in m._own and _nz(m.feat.at[a, "loc_id"])]
    gb = np.zeros((len(ids), 2))
    pool = [a for a in m.known_ids if _nz(m.feat.at[a, "loc_id"])]
    pos = {a: i for i, a in enumerate(ids)}
    for tr, te in KFold(10, shuffle=True, random_state=0).split(pool):
        te_ids = [pool[i] for i in te if pool[i] in pos]
        if te_ids:
            gb[[pos[a] for a in te_ids]] = gb_fit_predict(m, [pool[i] for i in tr], te_ids)
    est = estimators(m, ids, hide_own=True, params=params, gb=gb)
    T = m.known_xy[[m.known_row[a] for a in ids]]
    errs = {k: np.linalg.norm(P - T, axis=1) for k, P in est.items()}
    return m, ids, errs


def sensitivity(m: PinModel, ids: list[str], params: Params) -> pd.DataFrame:
    C = m.build_candidates(ids, hide_own=True)
    T = m.known_xy[[m.known_row[a] for a in ids]]
    rows = []

    def err(p):
        return np.linalg.norm(fuse(C, p)[0] - T, axis=1)

    base = err(params)
    for s in (0.5, 0.7, 1.0, 1.4, 2.0):
        e = err(_copy(params, **{k: getattr(params, k) * s for k in SIGMA_KEYS}))
        rows.append({"setting": "all sigmas x", "value": s, "median_m": np.median(e), "p90_m": np.quantile(e, .9)})
    for k in (1.5, 2.0, 2.5, params.gate_k, 4.0, 6.0, 1e9):
        e = err(_copy(params, gate_k=k))
        rows.append({"setting": "gate threshold k", "value": k if k < 1e8 else np.inf, "median_m": np.median(e), "p90_m": np.quantile(e, .9)})
    for s in ("visit", "grid", "cm", "br", "gali", "lmcm", "lmbr", "lm", "locgrp", "loccent", "poi"):
        en = tuple(x for x in params.enabled if x != s)
        e = err(_copy(params, enabled=en))
        rows.append({"setting": "drop one clue", "value": s, "median_m": np.median(e), "p90_m": np.quantile(e, .9)})
    out = pd.DataFrame(rows)
    out["median_vs_shipped_m"] = out.median_m - float(np.median(base))
    return out


def taxonomy(m: PinModel, ids: list[str], params: Params) -> pd.DataFrame:
    hp = m.predict(ids, hide_own=True)
    T = m.known_xy[[m.known_row[a] for a in ids]]
    old = m.base.loc[ids, ["geocoder_x", "geocoder_y"]].to_numpy()
    hp["err_model"] = np.hypot(hp.pin_x - T[:, 0], hp.pin_y - T[:, 1])
    hp["err_old"] = np.hypot(old[:, 0] - T[:, 0], old[:, 1] - T[:, 1])
    hp["old_pin_precision"] = m.base.loc[ids, "precision"].to_numpy()
    hp["worse_than_old_25m"] = hp.err_model > hp.err_old + 25
    hp["better_than_old_25m"] = hp.err_model < hp.err_old - 25
    g = hp.groupby(["method_used", "old_pin_precision"]).agg(
        n=("address_id", "size"), median_old_m=("err_old", "median"), median_model_m=("err_model", "median"),
        p90_model_m=("err_model", lambda x: x.quantile(.9)), share_better=("better_than_old_25m", "mean"),
        share_worse=("worse_than_old_25m", "mean")).reset_index()
    return g[g.n >= 8].sort_values("n", ascending=False)


def figure(tab: pd.DataFrame, sens: pd.DataFrame):
    order = ["old pin", "locality centroid", "street-key median", "equal-weight clues", "GP street numbers", "gradient boosting",
             "fusion, no gate", "fusion + gate (shipped)"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for a, ts, ttl in ((ax[0], "visited", "Visited addresses, own visit hidden"), (ax[1], "surveyed", "Surveyed addresses, out-of-fold")):
        t = tab[tab.test_set == ts].set_index("estimator").loc[order]
        y = np.arange(len(order))
        a.barh(y + .2, t.median_m, .4, label="median", color="#1d4486")
        a.barh(y - .2, t.p90_m, .4, label="90th percentile", color="#f5c542")
        a.set_yticks(y)
        a.set_yticklabels(order, fontsize=8)
        a.invert_yaxis()
        a.set_xlabel("error (m)")
        a.set_title(f"{ttl} (n={int(t.n.iloc[0])})", fontsize=9)
        a.legend(fontsize=7)
    s = sens[sens.setting == "gate threshold k"].copy()
    s["lab"] = s.value.map(lambda v: "off" if not np.isfinite(v) else f"{v:.1f}")
    ax[2].plot(range(len(s)), s.median_m, "o-", color="#1d4486", label="median")
    ax[2].plot(range(len(s)), s.p90_m, "s-", color="#c98a00", label="90th percentile")
    ax[2].set_xticks(range(len(s)))
    ax[2].set_xticklabels(s.lab, fontsize=8)
    ax[2].set_xlabel("gate threshold k (off = no gating)")
    ax[2].set_ylabel("error (m)")
    ax[2].set_title("Sensitivity to the gate", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "benchmark.png", dpi=150)
    plt.close(fig)


def main():
    feat, base, loc, lm, evid, sv, folds = load_all()
    params = Params.load()
    m, ids, errs_v = run_visited(feat, base, loc, lm, evid, params)
    errs_s = run_surveyed(feat, base, loc, lm, evid, sv, folds, params)
    tab = pd.concat([score_table(errs_v, "visited"), score_table(errs_s, "surveyed")], ignore_index=True)
    tab.to_csv(RES / "benchmark_estimators.csv", index=False)
    sens = sensitivity(m, ids, params)
    sens.to_csv(RES / "benchmark_sensitivity.csv", index=False)
    tax = taxonomy(m, ids, params)
    tax.to_csv(RES / "benchmark_error_taxonomy.csv", index=False)
    figure(tab, sens)
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string(index=False))
    print()
    print(sens.round(1).to_string(index=False))
    print()
    print(tax.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
