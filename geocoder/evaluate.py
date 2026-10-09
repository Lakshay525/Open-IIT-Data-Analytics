"""Evaluation driver: parse addresses -> leakage-safe CV on the 100 surveyed addresses
-> ablations -> hide-and-predict check -> final pins.csv.

Leakage rules (also described in README_TASK3.md)
- Surveyed truth is only ever used as *neighbour knowledge* from folds that are not being scored.
  Fold f is scored with truths from the other development folds (1-5); fold 0 (official
  test) is scored with truths from folds 1-5 and is never used as knowledge for anything else.
- Parameters were fitted on leave-one-out visit pseudo-labels (tune_pin_model.py), not on
  surveyed truth.
- pins.csv holds out-of-fold predictions for the 100 surveyed addresses, so eval.py can score
  the file directly.

Run:  python -m geocoder.evaluate
"""

from __future__ import annotations

from .paths import OUT, RES

import argparse

import numpy as np
import pandas as pd

from . import scoring as ev_mod
from .parser import main as run_parser
from .model import Params, PinModel, SLOTS, load_all



def cv_predict(params: Params, feat, base, loc, lm, evid, sv, folds, include_truth_neighbours=True,
               exclude_test_from_dev=True) -> pd.DataFrame:
    """Out-of-fold prediction for the 100 surveyed addresses."""
    out = []
    for f in range(0, 6):
        test_ids = folds.loc[folds.fold == f, "address_id"].tolist()
        train_ids = folds.loc[(folds.fold != f) & (folds.fold != 0), "address_id"].tolist()
        truth = sv[sv.address_id.isin(train_ids)] if include_truth_neighbours else None
        model = PinModel(params).fit(feat, base, evid, loc, lm, truth=truth)
        pr = model.predict(test_ids)
        pr["fold"] = f
        out.append(pr)
    return pd.concat(out, ignore_index=True)


def score(preds, sv, meta, label):
    pr = preds.rename(columns={"pin_x": "pred_x", "pin_y": "pred_y"})
    scored, table = ev_mod.score_predictions(pr[["address_id", "pred_x", "pred_y"]], sv, meta)
    return scored, table


def headline(preds, sv, meta, name):
    scored, table = score(preds, sv, meta, name)
    folds_ = preds.set_index("address_id").fold
    scored["fold"] = scored.address_id.map(folds_)
    dev = scored[scored.fold > 0].error_m
    tst = scored[scored.fold == 0].error_m
    allv = scored.error_m
    return {"variant": name,
            "dev85_median": dev.median(), "dev85_p90": dev.quantile(.9),
            "test15_median": tst.median(), "test15_p90": tst.quantile(.9),
            "all100_median": allv.median(), "all100_p90": allv.quantile(.9),
            "within50": (allv <= 50).mean(), "within100": (allv <= 100).mean(), "within250": (allv <= 250).mean()}, scored, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parse", action="store_true")
    args = ap.parse_args()
    if not args.skip_parse or not (RES / "address_features.csv").exists():
        run_parser()
    feat, base, loc, lm, evid, sv, folds = load_all()
    _, _, meta = ev_mod.baseline_inputs()
    params = Params.load()

    rows = []
    # ablation ladder -----------------------------------------------------
    ladder = [
        ("0 old geocoder only", ("baseline",)),
        ("1 + own visit evidence", ("baseline", "visit")),
        ("2 + locality group/centroid", ("baseline", "visit", "locgrp", "loccent")),
        ("3 + neighbours on same street key", ("baseline", "visit", "locgrp", "loccent", "cm", "br", "gali")),
        ("4 + street-number grid regression", ("baseline", "visit", "locgrp", "loccent", "cm", "br", "gali", "grid")),
        ("5 + landmark groups and POIs (full model)", tuple(SLOTS)),
    ]
    full_pred = None
    for name, enabled in ladder:
        p = Params.load()
        p.enabled = enabled
        pred = cv_predict(p, feat, base, loc, lm, evid, sv, folds)
        r, scored, table = headline(pred, sv, meta, name)
        rows.append(r)
        if name.startswith("5"):
            full_pred, full_scored, full_table = pred, scored, table
    ablation = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(ablation.round(1).to_string(index=False))
    ablation.to_csv(RES / "pin_model_ablation.csv", index=False)
    full_table.to_csv(RES / "pin_model_scores.csv", index=False)
    full_scored.to_csv(RES / "pin_model_oof_errors.csv", index=False)
    print("\nFull model by segment")
    print(full_table[["dimension", "segment", "n", "median_error_m", "p90_error_m", "share_within_50m",
                      "share_within_100m", "share_within_250m"]].round(2).to_string(index=False))

    # hide-and-predict check on visited addresses (no surveyed truth involved) -------------
    model = PinModel(params).fit(feat, base, evid, loc, lm)
    ids = list(model.known_ids)
    hp = model.predict(ids, hide_own=True)
    tgt = model.known_xy[[model.known_row[a] for a in ids]]
    old = model.base.loc[ids, ["geocoder_x", "geocoder_y"]].to_numpy()
    hp["err_model"] = np.hypot(hp.pin_x - tgt[:, 0], hp.pin_y - tgt[:, 1])
    hp["err_old"] = np.hypot(old[:, 0] - tgt[:, 0], old[:, 1] - tgt[:, 1])
    hp["precision"] = model.base.loc[ids, "precision"].to_numpy()
    hp.to_csv(RES / "hide_and_predict.csv", index=False)
    print(f"\nHide-and-predict on {len(hp)} visited addresses (target = their own cleaned visit pin):")
    print(f"  old pin   median {hp.err_old.median():6.1f} m  p90 {hp.err_old.quantile(.9):6.1f} m")
    print(f"  model     median {hp.err_model.median():6.1f} m  p90 {hp.err_model.quantile(.9):6.1f} m")
    print(hp.groupby("precision")[["err_old", "err_model"]].median().round(0).to_string())

    # final pins ------------------------------------------------------------
    final = PinModel(params).fit(feat, base, evid, loc, lm, truth=sv)
    pins = final.predict()
    oof = full_pred.set_index("address_id")
    is_sv = pins.address_id.isin(oof.index)
    for c in ["pin_x", "pin_y", "raw_spread", "method_used", "clues_used", "locality_used", "n_locality_hyp"]:
        pins.loc[is_sv, c] = pins.loc[is_sv, "address_id"].map(oof[c]).values
    pins["pin_source"] = np.where(is_sv, "oof_cv", "model")
    pins["has_own_visit"] = pins.address_id.isin(final._own.keys())
    pins.to_csv(OUT / "pins.csv", index=False)
    print(f"\nWrote {OUT/'pins.csv'} ({len(pins)} rows; {int(is_sv.sum())} surveyed rows are out-of-fold)")
    print(pins.method_used.value_counts().to_string())


if __name__ == "__main__":
    main()
