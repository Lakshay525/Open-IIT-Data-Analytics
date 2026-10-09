"""Survey diagnostics and counterfactual visit attacks for Task 2.

The attack defence uses the same evidence extractor and no clean pin or
surveyed coordinate. Surveyed locations are used only after pin formation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from extract_evidence import DATA, ROOT, load_feature_frame, score_evidence_features


def weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2, side="left")])


def pin(group: pd.DataFrame, method: str, *, weight_col: str = "weight") -> np.ndarray | None:
    group = group[group[weight_col] > 0]
    if group.empty:
        return None
    w = group[weight_col].to_numpy(float)
    xy = group[["ev_x", "ev_y"]].to_numpy(float)
    if method == "weighted_mean":
        return np.average(xy, axis=0, weights=w)
    if method == "weighted_median":
        return np.array([weighted_median_1d(xy[:, j], w) for j in range(2)])
    if method != "independent_agent_cluster":
        raise ValueError(method)
    labels = DBSCAN(eps=100., min_samples=1).fit_predict(xy)
    clusters = []
    for label in set(labels):
        part = group.iloc[np.flatnonzero(labels == label)]
        clusters.append((float(part[weight_col].sum()), part.agent_id.nunique(), part))
    clusters.sort(key=lambda item: -item[0])
    top_weight, agents, top = clusters[0]
    next_weight = clusters[1][0] if len(clusters) > 1 else 0.
    if agents < 2 or (next_weight and top_weight < 1.5 * next_weight):
        return None
    centre = top[["ev_x", "ev_y"]].to_numpy(float)
    weights = top[weight_col].to_numpy(float)
    return np.array([weighted_median_1d(centre[:, j], weights) for j in range(2)])


def survey_diagnostics(evidence: pd.DataFrame, data_dir: Path = DATA):
    survey = pd.read_csv(data_dir / "surveyed_addresses.csv")
    visits = pd.read_csv(data_dir / "field_visits.csv")
    baseline = pd.read_csv(data_dir / "baseline_geocodes.csv")
    v = evidence.merge(survey, on="address_id").merge(
        visits[["visit_id", "checkin_x", "checkin_y"]], on="visit_id", validate="one_to_one")
    v = v.merge(baseline[["address_id", "geocoder_x", "geocoder_y"]],
                on="address_id", validate="many_to_one")
    # The official test labels have been exposed in earlier exploratory work.
    # Do not use them here to select rules or quote a new held-out result.
    dev = v[v.split.ne("test")].copy()
    dev["endpoint_error_m"] = np.hypot(dev.ev_x - dev.surveyed_x, dev.ev_y - dev.surveyed_y)
    dev["checkin_error_m"] = np.hypot(
        dev.checkin_x - dev.surveyed_x, dev.checkin_y - dev.surveyed_y)
    dev["within_uncertainty"] = np.where(
        dev.uncertainty_m.notna(),
        (dev.endpoint_error_m <= dev.uncertainty_m).astype(float), np.nan)
    parts = []
    for field in ("evidence_role", "outcome", "trail_type", "photo_reused_cross_address"):
        summary = dev.groupby(field, dropna=False).agg(
            visits=("visit_id", "size"), addresses=("address_id", "nunique"),
            median_endpoint_error_m=("endpoint_error_m", "median"),
            median_checkin_error_m=("checkin_error_m", "median"),
            median_uncertainty_m=("uncertainty_m", "median"),
            share_within_uncertainty=("within_uncertainty", "mean"),
            median_weight=("weight", "median")).reset_index()
        summary.insert(0, "dimension", field)
        summary = summary.rename(columns={field: "segment"})
        parts.append(summary)
    diagnostic = pd.concat(parts, ignore_index=True)
    rows = []
    for aid, group in dev.groupby("address_id"):
        truth = group[["surveyed_x", "surveyed_y"]].iloc[0].to_numpy(float)
        base = group[["geocoder_x", "geocoder_y"]].iloc[0].to_numpy(float)
        contact = group[group.evidence_role.eq("contact_at_address") & group.eligible_for_pin]
        extended = group[group.eligible_for_pin]
        row = {"address_id": aid, "split": group.split.iloc[0],
               "baseline_error_m": float(np.linalg.norm(base - truth)),
               "contact_visits": len(contact), "all_eligible_visits": len(extended)}
        for label, subset in (("contact_only", contact), ("extended", extended)):
            estimate = pin(subset, "weighted_median") if len(subset) else None
            row[f"{label}_error_m"] = (
                float(np.linalg.norm(estimate - truth)) if estimate is not None else np.nan)
        rows.append(row)
    pins = pd.DataFrame(rows)
    eligible = dev[dev.eligible_for_pin]
    quality = pd.DataFrame([{
        "sample": "development surveyed eligible visit rows",
        "visits": len(eligible), "addresses": eligible.address_id.nunique(),
        "median_endpoint_error_m": float(eligible.endpoint_error_m.median()),
        "median_checkin_error_m": float(eligible.checkin_error_m.median()),
        "uncertainty_coverage": float(eligible.within_uncertainty.mean()),
        "target_pseudo_quantile": .80,
        "spearman_uncertainty_vs_error": float(
            eligible[["uncertainty_m", "endpoint_error_m"]].corr(method="spearman").iloc[0, 1]),
        "spearman_weight_vs_error": float(
            eligible[["weight", "endpoint_error_m"]].corr(method="spearman").iloc[0, 1]),
    }])
    return dev, diagnostic, pins, quality


def attack_rows(scenario: str, targets: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    centres = localities.groupby("town_id")[["centroid_x", "centroid_y"]].median()
    if scenario == "home_checkin":
        n_copies, offsets, attacked = 1, (1500, 1200), set(targets.town_id)
    elif scenario == "tea_stall":
        n_copies, offsets, attacked = 3, (-1700, 1300), set(targets.town_id)
    elif scenario == "rogue_agent":
        n_copies, offsets = 2, (2200, -1800)
        attacked = {targets.town_id.value_counts().idxmax()}
    else:
        raise ValueError(scenario)
    rows = []
    for row in targets.itertuples():
        if row.town_id not in attacked:
            continue
        centre = centres.loc[row.town_id]
        agent = (f"SIM_HOME_{row.address_id}" if scenario == "home_checkin" else
                 f"SIM_{scenario}_{row.town_id}")
        for copy in range(n_copies):
            visit_id = f"SIM_{scenario}_{row.address_id}_{copy}"
            rows.append({
                "visit_id": visit_id, "account_id": row.account_id,
                "address_id": row.address_id, "agent_id": agent,
                "checkin_ts": f"2026-09-{20 + copy:02d} 12:00:00",
                "photo_hash": f"SYNTH_{visit_id}", "remark": "met customer at home",
                "outcome": "met_borrower", "dwell_s": 420, "town_id": row.town_id,
                "address_type": row.address_type, "split": row.split,
                "geocoder_x": row.geocoder_x, "geocoder_y": row.geocoder_y,
                "ev_x": float(centre.centroid_x + offsets[0]),
                "ev_y": float(centre.centroid_y + offsets[1]),
                "trail_accuracy_m": 4., "cluster_radius_m": 3.,
                "dwell_cluster_fraction": .9, "dwell_cluster_points": 8,
                "straightness": .8, "trail_type": "direct",
            })
    return pd.DataFrame(rows)


def stress_test(data_dir: Path = DATA):
    features = load_feature_frame(data_dir)
    clean = score_evidence_features(features)
    survey = pd.read_csv(data_dir / "surveyed_addresses.csv")
    addresses = pd.read_csv(data_dir / "addresses.csv")
    baseline = pd.read_csv(data_dir / "baseline_geocodes.csv")
    splits = pd.read_csv(data_dir / "splits.csv")
    localities = pd.read_csv(data_dir / "localities.csv")
    targets = survey[["address_id"]].merge(
        addresses[["address_id", "account_id", "town_id", "address_type"]],
        on="address_id").merge(splits, on="account_id").merge(
            baseline[["address_id", "geocoder_x", "geocoder_y"]], on="address_id")
    targets = targets[targets.split.ne("test") &
                      targets.address_id.isin(clean.loc[clean.eligible_for_pin, "address_id"])]
    truth = survey.set_index("address_id")
    clean_eligible = clean[clean.eligible_for_pin]
    reference = {
        aid: pin(group, "weighted_median")
        for aid, group in clean_eligible[clean_eligible.address_id.isin(targets.address_id)].groupby("address_id")}
    results, injections = [], []
    for scenario in ("home_checkin", "tea_stall", "rogue_agent"):
        fake = attack_rows(scenario, targets, localities)
        injections.append(fake.assign(scenario=scenario)[[
            "scenario", "visit_id", "address_id", "agent_id", "ev_x", "ev_y"]])
        # No clean estimate or truth is supplied to the detector; all trust
        # factors are recomputed from observed and injected visit features.
        attacked = score_evidence_features(pd.concat([features, fake], ignore_index=True))
        for aid in fake.address_id.unique():
            part = attacked[(attacked.address_id == aid) & attacked.eligible_for_pin].copy()
            unguarded = part.copy()
            unguarded["naive_weight"] = np.where(
                unguarded.visit_id.astype(str).str.startswith("SIM_"), .95,
                unguarded.weight)
            clean_pin = reference[aid]
            ground = truth.loc[aid, ["surveyed_x", "surveyed_y"]].to_numpy(float)
            for name, group, method, weight_col in (
                ("mean, high-trust injection", unguarded, "weighted_mean", "naive_weight"),
                ("median, high-trust injection", unguarded, "weighted_median", "naive_weight"),
                ("median, recomputed safeguards", part, "weighted_median", "weight"),
                ("independent-agent cluster, may abstain", part,
                 "independent_agent_cluster", "weight"),
            ):
                estimate = pin(group, method, weight_col=weight_col)
                results.append({
                    "scenario": scenario, "address_id": aid, "method": name,
                    "abstained": estimate is None,
                    "pin_shift_m": (float(np.linalg.norm(estimate - clean_pin))
                                    if estimate is not None else np.nan),
                    "error_before_m": float(np.linalg.norm(clean_pin - ground)),
                    "error_after_m": (float(np.linalg.norm(estimate - ground))
                                      if estimate is not None else np.nan),
                    "injected_weight_total": float(part.loc[
                        part.visit_id.astype(str).str.startswith("SIM_"), "weight"].sum()),
                    "fake_visit_count": int((fake.address_id == aid).sum()),
                })
    return pd.DataFrame(results), pd.concat(injections, ignore_index=True), clean


def main() -> None:
    evidence = pd.read_csv(ROOT / "evidence.csv")
    dev, diagnostic, pins, quality = survey_diagnostics(evidence)
    stress, injections, _ = stress_test()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    diagnostic.to_csv(out / "visit_quality_by_group.csv", index=False)
    pins.to_csv(out / "visit_pin_diagnostics.csv", index=False)
    quality.to_csv(out / "visit_uncertainty_diagnostics.csv", index=False)
    stress.to_csv(out / "integrity_stress_results.csv", index=False)
    injections.to_csv(out / "synthetic_attack_rows.csv", index=False)
    summary = stress.groupby(["scenario", "method"]).agg(
        affected_addresses=("address_id", "nunique"),
        reported_pins=("pin_shift_m", "count"),
        abstentions=("abstained", "sum"),
        median_pin_shift_m=("pin_shift_m", "median"),
        p90_pin_shift_m=("pin_shift_m", lambda s: s.quantile(.9)),
        median_error_after_m=("error_after_m", "median")).reset_index()
    summary.to_csv(out / "integrity_stress_summary.csv", index=False)
    print("Development visit diagnostics:", quality.to_string(index=False))
    print("Development pin coverage:", pins[[
        "contact_only_error_m", "extended_error_m"]].notna().sum().to_dict())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
