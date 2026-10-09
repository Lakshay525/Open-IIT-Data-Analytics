"""Validate the source tables without silently changing source values."""

from __future__ import annotations

from .paths import DATA, RES

from pathlib import Path

import numpy as np
import pandas as pd



TABLES = {
    "towns.csv": ("town_id",),
    "localities.csv": ("locality_id",),
    "landmarks_poi.csv": ("poi_id",),
    "baseline_geocodes.csv": ("address_id",),
    "surveyed_addresses.csv": ("address_id",),
    "visit_gps_points.csv": ("visit_id", "seq"),
    "addresses.csv": ("address_id",),
    "field_visits.csv": ("visit_id",),
    "agents.csv": ("agent_id",),
    "accounts.csv": ("account_id",),
    "splits.csv": ("account_id",),
}


def validate(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = {name: pd.read_csv(data_dir / name) for name in TABLES}
    report = []
    for name, keys in TABLES.items():
        df = tables[name]
        key_dupes = int(df.duplicated(list(keys)).sum())
        report.append({
            "file": name,
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_key_rows": key_dupes,
            "missing_cells": int(df.isna().sum().sum()),
            "missing_columns": "; ".join(f"{c}:{int(n)}" for c, n in df.isna().sum().items() if n),
            "primary_key": "+".join(keys),
            "status": "PASS" if key_dupes == 0 and df.duplicated().sum() == 0
                      and not df[list(keys)].isna().any().any() else "FAIL",
        })
    quality = pd.DataFrame(report)

    towns = tables["towns.csv"]
    localities = tables["localities.csv"]
    poi = tables["landmarks_poi.csv"]
    base = tables["baseline_geocodes.csv"]
    surveyed = tables["surveyed_addresses.csv"]
    gps = tables["visit_gps_points.csv"]
    addresses = tables["addresses.csv"]
    visits = tables["field_visits.csv"]
    agents = tables["agents.csv"]
    accounts = tables["accounts.csv"]
    splits = tables["splits.csv"]
    checks: list[dict] = []

    def add(name: str, observed: object, expected: str, ok: bool, note: str = "") -> None:
        checks.append({"check": name, "observed": observed, "expected": expected,
                       "status": "PASS" if ok else "FAIL", "note": note})

    def fk(name: str, values: pd.Series, parents: pd.Series) -> None:
        missing = sorted(set(values.dropna()) - set(parents.dropna()))
        add(name, len(missing), "0 unresolved keys", not missing,
            ", ".join(map(str, missing[:5])))

    fk("localities.town_id -> towns", localities.town_id, towns.town_id)
    fk("landmarks_poi.town_id -> towns", poi.town_id, towns.town_id)
    fk("baseline_geocodes.address_id -> addresses", base.address_id, addresses.address_id)
    fk("surveyed_addresses.address_id -> baseline", surveyed.address_id, base.address_id)
    fk("field_visits.address_id -> addresses", visits.address_id, addresses.address_id)
    fk("field_visits.address_id -> baseline", visits.address_id, base.address_id)
    fk("field_visits.agent_id -> agents", visits.agent_id, agents.agent_id)
    fk("visit_gps_points.visit_id -> field_visits", gps.visit_id, visits.visit_id)
    fk("addresses.account_id -> accounts", addresses.account_id, accounts.account_id)
    fk("splits.account_id -> accounts", splits.account_id, accounts.account_id)
    fk("addresses.account_id -> splits", addresses.account_id, splits.account_id)
    expected_accounts = visits[["visit_id", "address_id", "account_id"]].merge(
        addresses[["address_id", "account_id"]], on="address_id",
        suffixes=("_visit", "_address"), validate="many_to_one")
    add("Visit account matches address account",
        int((expected_accounts.account_id_visit == expected_accounts.account_id_address).sum()),
        f"{len(visits)} matching rows",
        bool((expected_accounts.account_id_visit == expected_accounts.account_id_address).all()))
    add("Known official splits", ", ".join(sorted(splits.split.unique())),
        "train, validation, test",
        set(splits.split) == {"train", "validation", "test"})
    surveyed_meta = surveyed.merge(addresses[["address_id", "account_id", "address_type"]],
                                   on="address_id", validate="one_to_one").merge(
        splits, on="account_id", validate="many_to_one")
    add("Surveyed official test rows", int(surveyed_meta.split.eq("test").sum()),
        "reported; reserve for final model assessment",
        bool(surveyed_meta.split.notna().all()),
        "The 15-row source test group is identified, but prior exploratory review exposed some labels")
    add("Surveyed address types", ", ".join(sorted(surveyed_meta.address_type.unique())),
        "residence only in supplied survey", set(surveyed_meta.address_type) == {"residence"})

    geo_numeric = base[["geocoder_x", "geocoder_y"]].to_numpy(float)
    truth_numeric = surveyed[["surveyed_x", "surveyed_y"]].to_numpy(float)
    gps_numeric = gps[["x", "y", "accuracy_m"]].to_numpy(float)
    add("Finite baseline coordinates", int(np.isfinite(geo_numeric).all()), "1", bool(np.isfinite(geo_numeric).all()))
    add("Finite surveyed coordinates", int(np.isfinite(truth_numeric).all()), "1", bool(np.isfinite(truth_numeric).all()))
    add("Finite GPS coordinates and accuracy", int(np.isfinite(gps_numeric).all()),
        "1", bool(np.isfinite(gps_numeric).all()))
    add("Positive GPS accuracy", int((gps.accuracy_m > 0).sum()), f"{len(gps)} positive rows", bool((gps.accuracy_m > 0).all()))
    add("Non-negative dwell seconds", int((visits.dwell_s >= 0).sum()), f"{len(visits)} non-negative rows", bool((visits.dwell_s >= 0).all()))
    precision_ok = set(base.precision.dropna()).issubset({"rooftop", "street", "locality", "pincode"})
    add("Known precision tiers", ", ".join(sorted(base.precision.unique())),
        "rooftop, street, locality, pincode", precision_ok)
    in_scope = set(base.address_id)
    out_addresses = addresses.loc[~addresses.address_id.isin(in_scope)]
    add("Addresses without baseline geocode", len(out_addresses), "reported as out of evaluated scope",
        len(out_addresses) == 237, "All are marked town_id=OUT" if set(out_addresses.town_id) == {"OUT"} else "Check town IDs")
    add("Baseline coverage for town addresses", int(addresses.town_id.isin(towns.town_id).sum() - len(set(addresses.loc[addresses.town_id.isin(towns.town_id), 'address_id']) - in_scope)),
        "all 2,880 in-town baseline addresses", set(addresses.loc[addresses.town_id.isin(towns.town_id), "address_id"]) == in_scope)

    # GPS sequence integrity is checked per visit; timestamps can repeat but seq is unique.
    seq_ok = True
    timestamp_ok = True
    for _, g in gps.sort_values(["visit_id", "seq"]).groupby("visit_id"):
        seq = g.seq.to_numpy()
        seq_ok &= bool(np.array_equal(seq, np.arange(len(seq))))
        times = pd.to_datetime(g.point_ts, errors="coerce")
        timestamp_ok &= bool(times.notna().all() and times.is_monotonic_increasing)
    add("Contiguous GPS sequence per visit", seq_ok, "seq starts at 0 and increments by 1", seq_ok)
    add("Ordered GPS timestamps per visit", timestamp_ok, "nondecreasing by seq", timestamp_ok)
    field_agents = set(agents.loc[agents.channel == "field", "agent_id"])
    add("Field-agent town metadata", int(agents.loc[agents.agent_id.isin(set(visits.agent_id)), "town_id"].notna().sum()),
        f"{len(field_agents)} field agents with town IDs", bool(agents.loc[agents.agent_id.isin(field_agents), "town_id"].notna().all()))

    # This dataset has projected-looking x/y, not latitude/longitude.
    add("Coordinate convention", "x/y local Cartesian; errors measured with Euclidean distance",
        "metres per Task PDF", True,
        "Inferred from task units and town-radius fields; no CRS/EPSG metadata supplied")
    checks_df = pd.DataFrame(checks)
    if (quality.status != "PASS").any() or (checks_df.status != "PASS").any():
        failed = quality.loc[quality.status != "PASS", "file"].tolist() + checks_df.loc[checks_df.status != "PASS", "check"].tolist()
        raise ValueError("Input validation failed: " + ", ".join(failed))
    return quality, checks_df


def main() -> None:
    quality, checks = validate()
    out = RES
    out.mkdir(exist_ok=True)
    quality.to_csv(out / "file_quality.csv", index=False)
    checks.to_csv(out / "validation_checks.csv", index=False)
    print("File quality summary")
    print(quality.to_string(index=False))
    print("\nCross-file and domain checks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
