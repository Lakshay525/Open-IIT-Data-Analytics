"""Checks on the data audit, the shared scorer, the frozen folds and the visit evidence.  Run: python -m tests.test_data"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocoder.paths import DATA, OUT, RES
from geocoder.scoring import baseline_inputs, join_predictions


def test_scorer_rejects_missing_predictions():
    predictions, truth, meta = baseline_inputs(DATA)
    try:
        join_predictions(predictions[predictions.address_id.ne(truth.address_id.iloc[0])], truth, meta)
    except ValueError as exc:
        assert "lack" in str(exc)
    else:
        raise AssertionError("Scorer accepted a missing surveyed prediction")
    print("PASS scorer refuses incomplete predictions")


def test_folds_and_baseline():
    folds = pd.read_csv(OUT / "folds.csv")
    assert len(folds) == 100 and folds.address_id.is_unique
    assert folds.loc[folds.split == "test", "fold"].eq(0).all()
    assert folds.loc[folds.split != "test", "fold"].value_counts().sort_index().to_dict() == {i: 17 for i in range(1, 6)}
    assert folds[folds.fold > 0].groupby("fold_group").fold.nunique().eq(1).all(), "a locality group crosses folds"
    overall = pd.read_csv(RES / "baseline_scores.csv").query("dimension == 'overall'").iloc[0]
    assert overall.n == 100 and np.isclose(overall.median_error_m, 376.39775255)
    print("PASS folds frozen (15 test + 5 x 17 grouped development); baseline median 376.4 m")


def test_evidence_invariants():
    ev = pd.read_csv(OUT / "evidence.csv")
    assert len(ev) == 5578 and ev.visit_id.is_unique and ev.weight.between(0, 1).all()
    assert (ev.eligible_for_pin == ev.weight.ge(.001)).all()
    assert ev.loc[ev.negative_search_signal, "weight"].eq(0).all()
    assert ev.loc[ev.evidence_role.eq("contact_elsewhere"), "weight"].eq(0).all()
    assert ev.loc[ev.eligible_for_pin, "uncertainty_m"].notna().all()
    assert ev.loc[ev.negative_search_signal, "uncertainty_m"].isna().all()
    reused = ev[ev.photo_reused_cross_address & ev.uncertainty_m.notna()]
    assert reused.uncertainty_m.ge(1000).all(), "reused photos must carry a large uncertainty"
    assert pd.read_csv(RES / "integrity_stress_summary.csv").method.nunique() == 4
    print("PASS evidence invariants (roles, weights, photo reuse, stress-test summary)")


if __name__ == "__main__":
    test_scorer_rejects_missing_predictions()
    test_folds_and_baseline()
    test_evidence_invariants()
    print("all data checks passed")
