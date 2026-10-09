"""Execute notebooks and check the main Task 1/2 invariants."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_mpl_dir = TemporaryDirectory(prefix="geocoder_mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir.name)
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from eval import baseline_inputs, join_predictions

ROOT = Path(__file__).resolve().parent


def run_notebook(path: Path) -> None:
    book = json.loads(path.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(book["cells"], 1):
        if cell["cell_type"] == "code":
            try:
                exec(compile("".join(cell["source"]), f"{path.name}:cell{number}", "exec"),
                     namespace)
            except Exception as exc:
                raise AssertionError(f"{path.name} failed at cell {number}") from exc


def main() -> None:
    os.chdir(ROOT)
    predictions, truth, meta = baseline_inputs(ROOT / "clean_data")
    try:
        join_predictions(predictions[predictions.address_id.ne(truth.address_id.iloc[0])],
                         truth, meta)
    except ValueError as exc:
        assert "lack" in str(exc)
    else:
        raise AssertionError("Scorer accepted a missing surveyed prediction")

    for name in ("baseline_report.ipynb", "integrity_stress_test.ipynb"):
        run_notebook(ROOT / "notebooks" / name)
    folds = pd.read_csv(ROOT / "folds.csv")
    assert len(folds) == 100 and folds.address_id.is_unique
    assert folds.loc[folds.split == "test", "fold"].eq(0).all()
    assert folds.loc[folds.split != "test", "fold"].value_counts().sort_index().to_dict() == {
        i: 17 for i in range(1, 6)}
    assert folds[folds.fold > 0].groupby("fold_group").fold.nunique().eq(1).all()

    scored = pd.read_csv(ROOT / "results" / "baseline_scores.csv")
    overall = scored.query("dimension == 'overall'").iloc[0]
    assert overall.n == 100 and np.isclose(overall.median_error_m, 376.39775255)
    assert all(0 <= overall[f"share_within_{r}m"] <= 1 for r in (50, 100, 250))
    evidence = pd.read_csv(ROOT / "evidence.csv")
    assert len(evidence) == 5578 and evidence.visit_id.is_unique
    assert evidence.weight.between(0, 1).all()
    assert (evidence.eligible_for_pin == evidence.weight.ge(.001)).all()
    assert evidence.loc[evidence.negative_search_signal, "weight"].eq(0).all()
    assert evidence.loc[evidence.evidence_role.eq("contact_elsewhere"), "weight"].eq(0).all()
    assert evidence.loc[evidence.eligible_for_pin, "uncertainty_m"].notna().all()
    assert evidence.loc[evidence.negative_search_signal, "uncertainty_m"].isna().all()
    reused = evidence[evidence.photo_reused_cross_address & evidence.uncertainty_m.notna()]
    assert reused.uncertainty_m.ge(1000).all()
    assert pd.read_csv(ROOT / "results" / "integrity_stress_summary.csv").method.nunique() == 4
    print("PASS: notebooks executed, evaluation and evidence invariants hold")


if __name__ == "__main__":
    main()
