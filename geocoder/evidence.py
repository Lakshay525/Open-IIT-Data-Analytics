"""Extract one GPS endpoint, role, fitted uncertainty and trust weight per visit.

Uncertainty is fitted to leave-one-agent-out agreement on development
addresses. It is a proxy, not a probability or a verified fraud label.
"""

from __future__ import annotations

from .paths import DATA, OUT, RES

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor

OUTCOME_FACTOR = {
    "met_borrower": 1.0, "cash_collected": .9, "met_family": .5,
    "locked_premises": .25, "neighbour_says_shifted": .12,
    "no_such_person": .10,
}
CONTACT = {"met_borrower", "cash_collected", "met_family"}
STRUCTURE = {"locked_premises", "no_such_person"}
OFFSITE = re.compile(r"\b(?:at shop|dukaan pe mila|angadi nalli sikkidru)\b", re.I)
NOT_FOUND = re.compile(
    r"(?:address nahi mila|address sikkilla|could not trace address|"
    r"address not traceable|is address pe ghar nahi mila|address trace nahi hua)", re.I)
CORRECTION = re.compile(
    r"(?:actual house|asli ghar|nija mane|address wrong|address galat|"
    r"address tappu|house near|ghar .* (?:ke paas|peeche|saamne)|"
    r"mane .* (?:hattira|hinde|eduru))", re.I)
FEATURES = [
    "log_accuracy", "log_cluster_radius", "log_dwell", "cluster_fraction",
    "straightness", "prior_agreement", "photo_reuse", "offsite", "failed_structure",
    "shifted_structure", "trail_stationary", "trail_searching", "agent_integrity",
]


def extract_terminal_points(visits: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """Robust coordinate median of final GPS points near the trail's last point."""
    g = gps.copy()
    g["point_ts"] = pd.to_datetime(g.point_ts, errors="coerce")
    if g.point_ts.isna().any():
        raise ValueError("GPS point timestamps contain invalid values")
    g = g.sort_values(["visit_id", "point_ts", "seq"])
    visit_rows = visits.set_index("visit_id", drop=False)
    if visit_rows.index.has_duplicates:
        raise ValueError("Duplicate visits")
    rows = []
    for visit_id, trail in g.groupby("visit_id", sort=False):
        if visit_id not in visit_rows.index:
            raise ValueError(f"GPS visit {visit_id} missing from visits")
        meta = visit_rows.loc[visit_id]
        end = trail.point_ts.max()
        horizon_s = float(np.clip(meta.dwell_s, 120, 600))
        window = trail[trail.point_ts >= end - pd.to_timedelta(horizon_s, unit="s")]
        last = trail.iloc[-1]
        threshold_m = max(20., 2. * float(window.accuracy_m.median()))
        radius = np.hypot(window.x.to_numpy(float) - float(last.x),
                          window.y.to_numpy(float) - float(last.y))
        core = window.loc[radius <= threshold_m]
        if core.empty:
            core = trail.iloc[[-1]]
        cx, cy = float(core.x.median()), float(core.y.median())
        spread = np.hypot(core.x.to_numpy(float) - cx, core.y.to_numpy(float) - cy)
        xy = trail[["x", "y"]].to_numpy(float)
        path_len = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.
        net_len = float(np.hypot(cx - xy[0, 0], cy - xy[0, 1]))
        straightness = min(1., net_len / path_len) if path_len > 0 else 0.
        trail_type = ("stationary_or_sparse" if path_len < 80 else
                      "direct" if straightness >= .65 else "searching")
        rows.append({
            "visit_id": visit_id, "ev_x": cx, "ev_y": cy,
            "dwell_cluster_points": len(core),
            "dwell_cluster_fraction": len(core) / len(window),
            "cluster_radius_m": float(np.median(spread)),
            "straightness": straightness, "trail_type": trail_type,
            "trail_accuracy_m": float(window.accuracy_m.median()),
        })
    result = pd.DataFrame(rows)
    if result.visit_id.duplicated().any() or set(result.visit_id) != set(visits.visit_id):
        raise AssertionError("Expected exactly one endpoint per visit")
    return result


def _agent_integrity_factors(frame: pd.DataFrame, eps_m: float = 15.) -> pd.Series:
    """Soft-flag repeated stops across different addresses, using no truth."""
    factors = pd.Series(1., index=frame.index)
    for _, group in frame.groupby("agent_id", sort=False):
        if len(group) < 5:
            continue
        labels = DBSCAN(eps=eps_m, min_samples=5).fit_predict(group[["ev_x", "ev_y"]])
        for label in set(labels) - {-1}:
            members = group.index[labels == label]
            cluster = frame.loc[members]
            n_addresses = cluster.address_id.nunique()
            if n_addresses < 5:
                continue
            cx, cy = float(cluster.ev_x.median()), float(cluster.ev_y.median())
            gap = np.hypot(cluster.geocoder_x - cx, cluster.geocoder_y - cy)
            median_gap = float(gap.median())
            factor = (.20 if n_addresses >= 20 and median_gap >= 750 else
                      .45 if n_addresses >= 8 and median_gap >= 750 else
                      .65 if n_addresses >= 5 and median_gap >= 1000 else 1.)
            factors.loc[members] = np.minimum(factors.loc[members], factor)
    return factors


def _remark_fields(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    first = frame.remark.fillna("").astype(str).str.split(";").str[0]
    frame["offsite_meeting"] = first.str.contains(OFFSITE)
    frame["remark_not_found"] = first.str.contains(NOT_FOUND)
    frame["remark_correction_hint"] = frame.remark.fillna("").str.contains(CORRECTION)
    frame["landmark_hint"] = frame.remark.fillna("").apply(
        lambda x: x.split(";", 1)[1].strip()[:200]
        if ";" in x and CORRECTION.search(x.split(";", 1)[1]) else "")
    frame["remark_signal"] = np.select(
        [frame.offsite_meeting, frame.remark_not_found, frame.remark_correction_hint],
        ["offsite_meeting", "not_found_text", "landmark_hint"], default="none")
    return frame


def _prior_agreement(frame: pd.DataFrame) -> pd.Series:
    """Compare only earlier independent agents, never the current row itself."""
    factors = pd.Series(0.75, index=frame.index)
    prior: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for idx, row in frame.sort_values(["address_id", "checkin_ts", "visit_id"]).iterrows():
        previous = prior.get(row.address_id, {})
        independent = [xy for agent, history in previous.items()
                       if agent != row.agent_id for xy in [np.median(history, axis=0)]]
        if independent:
            px, py = np.median(independent, axis=0)
            gap = float(np.hypot(row.ev_x - px, row.ev_y - py))
            factors.loc[idx] = (1. if gap <= 50 else .75 if gap <= 150 else
                                .50 if gap <= 400 else .25)
        if (row.outcome in CONTACT and not row.offsite_meeting
                and not row.remark_not_found and not row.photo_reused_cross_address):
            prior.setdefault(row.address_id, {}).setdefault(row.agent_id, []).append(
                (float(row.ev_x), float(row.ev_y)))
    return factors


def _uncertainty_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_accuracy": np.log1p(frame.trail_accuracy_m.clip(lower=0)),
        "log_cluster_radius": np.log1p(frame.cluster_radius_m.clip(lower=0)),
        "log_dwell": np.log1p(frame.dwell_s.clip(lower=0)),
        "cluster_fraction": frame.dwell_cluster_fraction,
        "straightness": frame.straightness,
        "prior_agreement": frame.address_agreement_factor,
        "photo_reuse": frame.photo_reused_cross_address.astype(int),
        "offsite": frame.offsite_meeting.astype(int),
        "failed_structure": frame.outcome.isin(STRUCTURE).astype(int),
        "shifted_structure": frame.outcome.eq("neighbour_says_shifted").astype(int),
        "trail_stationary": frame.trail_type.eq("stationary_or_sparse").astype(int),
        "trail_searching": frame.trail_type.eq("searching").astype(int),
        "agent_integrity": frame.agent_integrity_factor,
    }, index=frame.index)[FEATURES].fillna(0.)


def _pseudo_labels(frame: pd.DataFrame) -> pd.Series:
    """Distance to a consensus of two or more *other* agents at one address."""
    labels = pd.Series(np.nan, index=frame.index)
    good_ref = frame.outcome.isin(CONTACT) & ~frame.offsite_meeting & ~frame.remark_not_found
    good_ref &= ~frame.photo_reused_cross_address
    for _, group in frame.groupby("address_id", sort=False):
        if group.agent_id.nunique() < 3:
            continue
        for agent, rows in group.groupby("agent_id"):
            reference = group[good_ref.loc[group.index] & group.agent_id.ne(agent)]
            if reference.agent_id.nunique() < 2:
                continue
            per_agent = reference.groupby("agent_id")[["ev_x", "ev_y"]].median()
            cx, cy = per_agent.median()
            candidates = rows[rows.outcome.isin(OUTCOME_FACTOR) &
                              ~rows.offsite_meeting & ~rows.remark_not_found]
            labels.loc[candidates.index] = np.hypot(candidates.ev_x - cx, candidates.ev_y - cy)
    return labels


def _fit_uncertainty(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Fit an 80th-percentile proxy without using surveyed coordinates."""
    labels = _pseudo_labels(frame)
    features = _uncertainty_features(frame)
    fitting = labels.notna() & frame.split.ne("test")
    if fitting.sum() < 100:
        raise ValueError("Insufficient independent-agent pseudo-labels for uncertainty fit")
    model = GradientBoostingRegressor(
        loss="quantile", alpha=.80, n_estimators=90, max_depth=2,
        min_samples_leaf=30, learning_rate=.05, random_state=20261006)
    model.fit(features.loc[fitting], np.log1p(labels.loc[fitting].clip(0, 10000)))
    predicted = np.expm1(model.predict(features)).clip(15., 5000.)
    uncertainty = pd.Series(
        np.maximum(predicted, 2 * frame.trail_accuracy_m.to_numpy(float)),
        index=frame.index)
    # Hard floors for known ambiguous evidence. These rules do not train on truth.
    uncertainty.loc[frame.photo_reused_cross_address] = np.maximum(
        uncertainty.loc[frame.photo_reused_cross_address], 1000.)
    uncertainty.loc[frame.offsite_meeting] = np.maximum(
        uncertainty.loc[frame.offsite_meeting], 1000.)
    uncertainty.loc[frame.trail_type.eq("stationary_or_sparse")] = np.maximum(
        uncertainty.loc[frame.trail_type.eq("stationary_or_sparse")], 250.)
    details = {
        "pseudo_label_visits": int(fitting.sum()),
        "pseudo_label_addresses": int(frame.loc[fitting, "address_id"].nunique()),
        "pseudo_label_80th_m": float(labels.loc[fitting].quantile(.80)),
        "fit_definition": "leave-one-agent-out median of at least two other contact agents",
        "training_population": "non-test accounts only; no surveyed coordinates",
        "uncertainty_interpretation": "proxy 80th percentile; surveyed coverage checked separately",
    }
    return uncertainty, details


def score_evidence_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recalculate all trust factors on a set of real or injected visits."""
    frame = frame.copy()
    frame["photo_reused_cross_address"] = frame.groupby("photo_hash").address_id.transform(
        "nunique").gt(1) & frame.photo_hash.notna()
    frame = _remark_fields(frame)
    frame["agent_integrity_factor"] = _agent_integrity_factors(frame)
    frame["address_agreement_factor"] = _prior_agreement(frame)
    frame["uncertainty_m"], details = _fit_uncertainty(frame)
    frame.attrs["uncertainty_fit"] = details

    frame["evidence_role"] = np.select(
        [frame.offsite_meeting & frame.outcome.isin(CONTACT),
         frame.outcome.eq("address_not_traceable") | frame.remark_not_found,
         frame.outcome.eq("neighbour_says_shifted"),
         frame.outcome.isin(STRUCTURE),
         frame.outcome.isin(CONTACT)],
        ["contact_elsewhere", "failed_search", "historical_structure",
         "structure_only", "contact_at_address"], default="none")
    candidate = frame.evidence_role.isin(
        {"historical_structure", "structure_only", "contact_at_address"})
    base = frame.outcome.map(OUTCOME_FACTOR).fillna(0.).astype(float)
    base.loc[~candidate] = 0.
    accuracy_factor = np.exp(-np.maximum(frame.trail_accuracy_m - 8., 0.) / 35.)
    dwell_factor = .25 + .75 * np.minimum(frame.dwell_s / 300., 1.)
    cluster_factor = .35 + .65 * frame.dwell_cluster_fraction.clip(0, 1)
    trail_factor = frame.trail_type.map(
        {"direct": 1., "searching": .60, "stationary_or_sparse": .10}).fillna(.10)
    uncertainty_factor = (80. / (80. + frame.uncertainty_m)).clip(0, 1)
    photo_factor = np.where(frame.photo_reused_cross_address, .05, 1.)
    frame["weight_before_agent_cap"] = (
        base * accuracy_factor * dwell_factor * cluster_factor * trail_factor
        * frame.address_agreement_factor * frame.agent_integrity_factor
        * uncertainty_factor * photo_factor).clip(0, 1)
    frame["agent_address_cap_factor"] = 1.
    for _, indices in frame.groupby(["address_id", "agent_id"], sort=False).groups.items():
        total = float(frame.loc[indices, "weight_before_agent_cap"].sum())
        if total > .35:
            frame.loc[indices, "agent_address_cap_factor"] = .35 / total
    frame["weight"] = (
        frame.weight_before_agent_cap * frame.agent_address_cap_factor).clip(0, 1)
    # Very small downweighted evidence remains visible for audit but cannot
    # create a stand-alone pin or count as surveyed-address coverage.
    frame["eligible_for_pin"] = candidate & frame.weight.ge(.001)
    frame["residence_target"] = frame.address_type.eq("residence")
    frame["negative_search_signal"] = frame.evidence_role.eq("failed_search")
    # This error-scale proxy describes candidates for an address pin. A
    # failed-search endpoint is not a claimed home location.
    frame.loc[~candidate, "uncertainty_m"] = np.nan
    return frame


def load_feature_frame(data_dir: Path = DATA) -> pd.DataFrame:
    """Read sources and extract GPS endpoint features before trust scoring."""
    visits = pd.read_csv(data_dir / "field_visits.csv")
    gps = pd.read_csv(data_dir / "visit_gps_points.csv")
    addresses = pd.read_csv(data_dir / "addresses.csv")
    baseline = pd.read_csv(data_dir / "baseline_geocodes.csv")
    splits = pd.read_csv(data_dir / "splits.csv")
    agents = pd.read_csv(data_dir / "agents.csv")
    if visits.visit_id.duplicated().any() or gps.duplicated(["visit_id", "seq"]).any():
        raise ValueError("Duplicate visit or GPS key")
    if not set(visits.address_id).issubset(set(addresses.address_id)):
        raise ValueError("Unresolved visit address")
    if not set(visits.agent_id).issubset(set(agents.agent_id)):
        raise ValueError("Unresolved visit agent")
    terminal = extract_terminal_points(visits, gps)
    frame = visits.merge(terminal, on="visit_id", validate="one_to_one")
    frame = frame.merge(
        addresses[["address_id", "town_id", "address_type"]],
        on="address_id", validate="many_to_one").merge(
            splits, on="account_id", validate="many_to_one").merge(
                baseline[["address_id", "geocoder_x", "geocoder_y"]],
                on="address_id", validate="many_to_one")
    return frame


def build_evidence(data_dir: Path = DATA, *, return_details: bool = False):
    frame = score_evidence_features(load_feature_frame(data_dir))
    details = frame.attrs.get("uncertainty_fit", {})
    columns = [
        "visit_id", "address_id", "ev_x", "ev_y", "weight", "trail_type",
        "uncertainty_m", "evidence_role", "negative_search_signal",
        "eligible_for_pin", "residence_target", "address_type", "outcome",
        "agent_id", "town_id", "split", "photo_reused_cross_address",
        "remark_signal", "landmark_hint", "trail_accuracy_m", "dwell_s",
        "dwell_cluster_points", "dwell_cluster_fraction", "cluster_radius_m",
        "straightness", "address_agreement_factor", "agent_integrity_factor",
        "agent_address_cap_factor",
    ]
    result = frame[columns].sort_values(["address_id", "visit_id"]).reset_index(drop=True)
    if result.visit_id.duplicated().any():
        raise AssertionError("Evidence output must contain one row per visit")
    return (result, details) if return_details else result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT / "evidence.csv")
    args = parser.parse_args()
    result, details = build_evidence(args.data_dir, return_details=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep enough precision that a tiny, nonzero photo penalty does not turn
    # into weight 0 while eligible_for_pin remains True in the exported CSV.
    result.to_csv(args.out, index=False, float_format="%.9g")
    pd.DataFrame([details]).to_csv(RES / "uncertainty_fit_summary.csv", index=False)
    print(f"Wrote {len(result):,} visits; eligible: {int(result.eligible_for_pin.sum())}")
    print("Evidence roles:", result.evidence_role.value_counts().to_dict())
    print("Uncertainty fit:", details)


if __name__ == "__main__":
    main()
