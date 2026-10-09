"""Safety checks for the pin model. Run:  python -m tests.test_model

1. Leakage: changing a fold's surveyed truth must not change that fold's predictions,
   and changing the official-test fold (0) must not change any development prediction.
2. Online learning: one new visit moves predictions of other addresses on the same street.
3. Output contract: pins.csv has the agreed columns, finite numbers, one row per address,
   and scores with the shared eval.py.
4. Own-visit protection: an address with a tight cluster of visit rows is never overruled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocoder.paths import OUT
from geocoder import scoring as ev_mod
from geocoder.model import Params, PinModel, load_all


def predict_fold(feat, base, loc, lm, evid, sv, folds, f, sv_override=None):
    sv_use = sv if sv_override is None else sv_override
    test_ids = folds.loc[folds.fold == f, "address_id"].tolist()
    train_ids = folds.loc[(folds.fold != f) & (folds.fold != 0), "address_id"].tolist()
    m = PinModel(Params.load()).fit(feat, base, evid, loc, lm, truth=sv_use[sv_use.address_id.isin(train_ids)])
    return m.predict(test_ids).set_index("address_id")[["pin_x", "pin_y"]]


def test_leakage():
    feat, base, loc, lm, evid, sv, folds = load_all()
    ref = {f: predict_fold(feat, base, loc, lm, evid, sv, folds, f) for f in (0, 3)}
    # (a) corrupt the truth of the fold being predicted
    for f in (0, 3):
        bad = sv.copy()
        ids = folds.loc[folds.fold == f, "address_id"]
        bad.loc[bad.address_id.isin(ids), ["surveyed_x", "surveyed_y"]] += 5000.0
        got = predict_fold(feat, base, loc, lm, evid, sv, folds, f, sv_override=bad)
        assert np.allclose(got.values, ref[f].values), f"fold {f} predictions depend on its own truth"
    # (b) corrupt the official-test truth: development predictions must not move
    bad = sv.copy()
    ids0 = folds.loc[folds.fold == 0, "address_id"]
    bad.loc[bad.address_id.isin(ids0), ["surveyed_x", "surveyed_y"]] += 5000.0
    got = predict_fold(feat, base, loc, lm, evid, sv, folds, 3, sv_override=bad)
    assert np.allclose(got.values, ref[3].values), "dev predictions depend on official-test truth"
    print("PASS leakage: a fold's predictions ignore its own truth; dev ignores the official-test truth")


def test_online_update():
    feat, base, loc, lm, evid, sv, folds = load_all()
    m = PinModel(Params.load()).fit(feat, base, evid, loc, lm)
    f = m.feat
    unvisited = f.index[~f.index.isin(m._own.keys())]
    sub = f.loc[unvisited]
    sub = sub[sub.cross.notna() & sub.main.notna() & sub.loc_id.notna()]
    grp = sub.groupby(["loc_id", "cross", "main"]).filter(lambda g: len(g) >= 2)
    first = next(iter(grp.groupby(["loc_id", "cross", "main"])))[1]
    a, b = first.index[0], first.index[1]
    assert (f.loc[a, ["loc_id", "cross", "main"]] == f.loc[b, ["loc_id", "cross", "main"]]).all()
    before = m.predict([a]).iloc[0]
    target = np.array([before.pin_x + 300.0, before.pin_y - 300.0])
    new = pd.DataFrame([{"visit_id": "VSNEW1", "address_id": b, "ev_x": target[0], "ev_y": target[1], "weight": 0.5,
                         "eligible_for_pin": True, "evidence_role": "contact_at_address", "uncertainty_m": 100.0}])
    m.update(new)
    after = m.predict([a]).iloc[0]
    d0 = np.hypot(before.pin_x - target[0], before.pin_y - target[1])
    d1 = np.hypot(after.pin_x - target[0], after.pin_y - target[1])
    assert d1 < d0, "a new visit on the same street should pull its neighbour toward it"
    print(f"PASS online update: neighbour {a} moved {d0:.0f} m -> {d1:.0f} m from the new visit at {b}")


def test_visit_protection():
    feat, base, loc, lm, evid, sv, folds = load_all()
    m = PinModel(Params.load()).fit(feat, base, evid, loc, lm)
    strong = [a for a, (x, y, n, s, W) in m._own.items() if n >= 2 and s < 60][:200]
    p = m.predict(strong)
    own = np.array([[m._own[a][0], m._own[a][1]] for a in strong])
    d = np.hypot(p.pin_x - own[:, 0], p.pin_y - own[:, 1]).to_numpy()
    assert np.median(d) < 30 and np.quantile(d, 0.95) < 120, "tight visit clusters should dominate the pin"
    print(f"PASS visit protection: {len(strong)} tight-cluster addresses, median pull {np.median(d):.1f} m")


def test_output_contract():
    pins = pd.read_csv(OUT / "pins.csv")
    need = {"address_id", "pin_x", "pin_y", "raw_spread", "method_used"}
    assert need.issubset(pins.columns), need - set(pins.columns)
    assert len(pins) == 2880 and pins.address_id.is_unique
    assert np.isfinite(pins[["pin_x", "pin_y", "raw_spread"]].to_numpy()).all()
    assert (pins.raw_spread > 0).all()
    _, truth, meta = ev_mod.baseline_inputs()
    scored, table = ev_mod.score_predictions(pins.rename(columns={"pin_x": "pred_x", "pin_y": "pred_y"})[
        ["address_id", "pred_x", "pred_y"]], truth, meta)
    overall = table[table.dimension == "overall"].iloc[0]
    print(f"PASS output contract: 2,880 rows; eval.py overall median {overall.median_error_m:.1f} m, "
          f"p90 {overall.p90_error_m:.1f} m")


if __name__ == "__main__":
    test_output_contract()
    test_visit_protection()
    test_online_update()
    test_leakage()
    print("all model checks passed")
