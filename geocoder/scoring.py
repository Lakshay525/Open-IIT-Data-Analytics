"""Shared scoring and leakage-aware development folds for geocoder Tasks 1-2.

Coordinates are town-local Cartesian values. Metres are inferred from the
task's units and town-radius fields; the source has no explicit CRS.
"""

from __future__ import annotations

from .paths import DATA, OUT, RES

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_OUTCOMES = {"met_borrower", "cash_collected", "met_family"}
THRESHOLDS = (50, 100, 250)


def _unique_key(frame: pd.DataFrame, key: str, label: str) -> None:
    if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"{label}.{key} must exist, be non-null and unique")


def join_predictions(predictions: pd.DataFrame, truth: pd.DataFrame,
                     metadata: pd.DataFrame, *, pred_x: str = "pred_x",
                     pred_y: str = "pred_y") -> pd.DataFrame:
    """Require one prediction per surveyed address, then calculate metre error."""
    for frame, label in ((predictions, "predictions"), (truth, "truth"), (metadata, "metadata")):
        _unique_key(frame, "address_id", label)
    if not {pred_x, pred_y}.issubset(predictions):
        raise ValueError("Predictions lack coordinate columns")
    if not {"surveyed_x", "surveyed_y"}.issubset(truth):
        raise ValueError("Truth lacks surveyed coordinates")
    if not {"town_id", "precision"}.issubset(metadata):
        raise ValueError("Metadata lacks town_id or precision")
    missing = set(truth.address_id) - set(predictions.address_id)
    if missing:
        raise ValueError(f"Predictions lack {len(missing)} surveyed addresses")
    segments = [c for c in ("town_id", "precision", "split", "visit_status", "contact_status")
                if c in metadata]
    joined = predictions[["address_id", pred_x, pred_y]].merge(
        truth[["address_id", "surveyed_x", "surveyed_y"]],
        on="address_id", validate="one_to_one",
    ).merge(metadata[["address_id", *segments]], on="address_id", validate="one_to_one")
    if len(joined) != len(truth) or joined[segments].isna().any().any():
        raise ValueError("Missing surveyed row or segment metadata")
    xy = joined[[pred_x, pred_y, "surveyed_x", "surveyed_y"]].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("Prediction/truth coordinates must be finite")
    joined["error_m"] = np.hypot(joined[pred_x] - joined.surveyed_x,
                                  joined[pred_y] - joined.surveyed_y)
    return joined


def _bootstrap_ci(values: np.ndarray, statistic: str, rng: np.random.Generator,
                  n_boot: int) -> tuple[float | None, float | None]:
    if len(values) < 10:
        return None, None
    samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
    if statistic == "median":
        estimates = np.median(samples, axis=1)
    elif statistic == "p90":
        estimates = np.quantile(samples, 0.9, axis=1)
    elif statistic.startswith("within_"):
        estimates = np.mean(samples <= int(statistic.split("_")[1]), axis=1)
    else:
        raise ValueError(statistic)
    return tuple(map(float, np.quantile(estimates, [0.025, 0.975])))


def score_predictions(predictions: pd.DataFrame, truth: pd.DataFrame,
                      metadata: pd.DataFrame, *, pred_x: str = "pred_x",
                      pred_y: str = "pred_y", n_boot: int = 3000,
                      seed: int = 20261006) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all surveyed addresses plus available town/tier/split/visit cuts."""
    scored = join_predictions(predictions, truth, metadata, pred_x=pred_x, pred_y=pred_y)
    rng = np.random.default_rng(seed)
    groups = [("overall", "all", scored)]
    for col in ("precision", "town_id", "split", "visit_status", "contact_status"):
        if col in scored:
            groups.extend((col, str(name), group) for name, group in
                          scored.groupby(col, sort=True))
    rows = []
    for dimension, segment, group in groups:
        errors = group.error_m.to_numpy(float)
        row = {"dimension": dimension, "segment": segment, "n": len(errors),
               "median_error_m": float(np.median(errors)),
               "p90_error_m": float(np.quantile(errors, 0.9)),
               "ci_note": "suppressed: n < 10" if len(errors) < 10 else ""}
        for label in ("median", "p90"):
            low, high = _bootstrap_ci(errors, label, rng, n_boot)
            row[f"{label}_ci95_low_m"] = low
            row[f"{label}_ci95_high_m"] = high
        for radius in THRESHOLDS:
            label = f"share_within_{radius}m"
            row[label] = float(np.mean(errors <= radius))
            low, high = _bootstrap_ci(errors, f"within_{radius}", rng, n_boot)
            row[f"{label}_ci95_low"] = low
            row[f"{label}_ci95_high"] = high
        rows.append(row)
    return scored, pd.DataFrame(rows)


def baseline_inputs(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(data_dir / "baseline_geocodes.csv")
    truth = pd.read_csv(data_dir / "surveyed_addresses.csv")
    addresses = pd.read_csv(data_dir / "addresses.csv")
    splits = pd.read_csv(data_dir / "splits.csv")
    visits = pd.read_csv(data_dir / "field_visits.csv")
    _unique_key(splits, "account_id", "splits")
    visited = set(visits.address_id)
    contacted = set(visits.loc[visits.outcome.isin(POSITIVE_OUTCOMES), "address_id"])
    meta = baseline[["address_id", "precision", "geocoder_x", "geocoder_y"]].merge(
        addresses[["address_id", "account_id", "town_id", "address_text"]],
        on="address_id", validate="one_to_one",
    ).merge(splits, on="account_id", validate="many_to_one")
    meta["visit_status"] = np.where(meta.address_id.isin(visited), "visited", "unvisited")
    meta["contact_status"] = np.where(meta.address_id.isin(contacted), "contact", "no_contact")
    preds = baseline.rename(columns={"geocoder_x": "pred_x", "geocoder_y": "pred_y"})[
        ["address_id", "pred_x", "pred_y"]]
    return preds, truth, meta


def make_fixed_folds(surveyed: pd.DataFrame, metadata: pd.DataFrame,
                     localities: pd.DataFrame, *, n_folds: int = 5,
                     seed: int = 20261006) -> pd.DataFrame:
    """Group development addresses by closest baseline-locality proxy.

    Official test addresses get fold 0. Truth coordinates do not enter group
    assignment. This mitigates, but does not eliminate, spatial leakage.
    """
    _unique_key(surveyed, "address_id", "surveyed")
    _unique_key(metadata, "address_id", "metadata")
    need = ["address_id", "town_id", "precision", "split", "geocoder_x", "geocoder_y"]
    if not set(need).issubset(metadata):
        raise ValueError(f"Fold metadata needs {need}")
    frame = surveyed[["address_id"]].merge(metadata[need], on="address_id", validate="one_to_one")
    if frame[need].isna().any().any() or not set(frame.split).issubset({"train", "validation", "test"}):
        raise ValueError("Missing or unexpected fold metadata")
    nearest = {}
    for town, group in frame.groupby("town_id"):
        loc = localities[localities.town_id == town]
        if loc.empty:
            raise ValueError(f"No locality centroids for {town}")
        xy = group[["geocoder_x", "geocoder_y"]].to_numpy(float)
        centres = loc[["centroid_x", "centroid_y"]].to_numpy(float)
        closest = np.argmin(((xy[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
        nearest.update(zip(group.address_id, loc.iloc[closest].locality_id))
    frame["fold_group"] = frame.address_id.map(nearest)
    dev = frame[frame.split != "test"]
    if len(dev) < n_folds:
        raise ValueError("Too few development addresses")
    rng = np.random.default_rng(seed)
    groups = [(name, g) for name, g in dev.groupby("fold_group", sort=True)]
    rng.shuffle(groups)
    groups.sort(key=lambda x: -len(x[1]))
    sizes = np.zeros(n_folds, dtype=int)
    town_counts = {t: np.zeros(n_folds, dtype=int) for t in dev.town_id.unique()}
    tier_counts = {t: np.zeros(n_folds, dtype=int) for t in dev.precision.unique()}
    assigned = {}
    target = len(dev) / n_folds
    for name, group in groups:
        options = []
        for fold in range(n_folds):
            prospective = sizes.copy(); prospective[fold] += len(group)
            objective = float(np.square(prospective - target).sum())
            for value, counts in town_counts.items():
                c = counts.copy(); c[fold] += int((group.town_id == value).sum())
                objective += .20 * float(np.square(c - (dev.town_id == value).sum() / n_folds).sum())
            for value, counts in tier_counts.items():
                c = counts.copy(); c[fold] += int((group.precision == value).sum())
                objective += .08 * float(np.square(c - (dev.precision == value).sum() / n_folds).sum())
            options.append(objective)
        best = int(np.argmin(options))
        sizes[best] += len(group)
        for value, counts in town_counts.items():
            counts[best] += int((group.town_id == value).sum())
        for value, counts in tier_counts.items():
            counts[best] += int((group.precision == value).sum())
        assigned[name] = best + 1
    frame["fold"] = np.where(frame.split == "test", 0, frame.fold_group.map(assigned)).astype(int)
    frame["role"] = np.where(frame.split == "test", "official_test_holdout", "development_cv")
    if frame.address_id.nunique() != len(surveyed) or set(frame.loc[dev.index, "fold"]) != set(range(1, n_folds + 1)):
        raise AssertionError("Missing fold or address")
    if frame.loc[dev.index].groupby("fold_group").fold.nunique().max() != 1:
        raise AssertionError("Locality group split between folds")
    return frame.drop(columns=["geocoder_x", "geocoder_y"]).sort_values(
        ["fold", "fold_group", "address_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RES / "baseline_scores.csv")
    args = parser.parse_args()
    predictions, truth, metadata = baseline_inputs(args.data_dir)
    scored, summary = score_predictions(predictions, truth, metadata)
    localities = pd.read_csv(args.data_dir / "localities.csv")
    folds = make_fixed_folds(truth, metadata, localities)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    scored.to_csv(args.out.with_name("baseline_address_errors.csv"), index=False)
    folds.to_csv(OUT / "folds.csv", index=False)
    print(summary.to_string(index=False))
    print("Fold sizes (0 = official test holdout):", folds.fold.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
