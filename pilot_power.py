"""Pilot sample-size calculation from the supplied visit logs (feeds pilot_plan.md).

Outcome 1: share of visits ending 'address not traceable' (binary, clustered by territory = locality)
Outcome 2: productive visits per agent-day (continuous, clustered by agent)
ICC by the one-way ANOVA estimator; design effect DE = 1 + (m - 1) * ICC.
Run: python pilot_power.py  -> results/pilot_power.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CONTACT = {"met_borrower", "cash_collected", "met_family"}


def icc_anova(df: pd.DataFrame, cluster: str, y: str) -> tuple[float, float]:
    g = df.groupby(cluster)[y]
    k, n = g.ngroups, len(df)
    sizes = g.size().to_numpy(float)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    gm = df[y].mean()
    msb = (sizes * (g.mean() - gm) ** 2).sum() / (k - 1)
    msw = ((df[y] - g.transform("mean")) ** 2).sum() / (n - k)
    return float(max(0.0, (msb - msw) / (msb + (m0 - 1) * msw))), float(m0)


def main():
    fv = pd.read_csv(ROOT / "clean_data" / "field_visits.csv")
    feat = pd.read_csv(RES / "address_features.csv").set_index("address_id")
    fv["territory"] = fv.address_id.map(feat.loc_id).fillna(fv.address_id.map(feat.town_id))
    fv["nt"] = fv.outcome.eq("address_not_traceable").astype(float)
    fv["prod"] = fv.outcome.isin(CONTACT).astype(float)
    p0 = fv.nt.mean()
    icc_t, m_t = icc_anova(fv, "territory", "nt")
    za, zb = norm.ppf(0.975), norm.ppf(0.80)
    rows = []
    weeks = 6
    visits_per_territory = len(fv) / fv.territory.nunique() * weeks / 13.0   # logs span 13 weeks
    for rel in (0.10, 0.15, 0.20, 0.30):
        p1 = p0 * (1 - rel)
        n_ind = (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p0 - p1) ** 2
        de = 1 + (visits_per_territory - 1) * icc_t
        n_arm = n_ind * de
        rows.append({"outcome": "address-not-traceable share", "baseline": p0, "relative_reduction_to_detect": rel,
                     "visits_per_arm_if_unclustered": n_ind, "icc": icc_t, "visits_per_territory_in_6_weeks": visits_per_territory,
                     "design_effect": de, "visits_per_arm_clustered": n_arm, "territories_per_arm": n_arm / visits_per_territory})
    # productive visits per agent-day (field agents only)
    ad = fv.groupby(["agent_id", "visit_date"]).agg(prod=("prod", "sum"), visits=("prod", "size")).reset_index()
    icc_a, m_a = icc_anova(ad.assign(y=ad["prod"]), "agent_id", "y")
    mu, sd = ad["prod"].mean(), ad["prod"].std()
    for rel in (0.05, 0.10, 0.15):
        n_ind = 2 * (za + zb) ** 2 * sd ** 2 / (mu * rel) ** 2
        de = 1 + (m_a - 1) * icc_a
        rows.append({"outcome": "productive visits per agent-day", "baseline": mu, "relative_reduction_to_detect": rel,
                     "visits_per_arm_if_unclustered": n_ind, "icc": icc_a, "visits_per_territory_in_6_weeks": m_a,
                     "design_effect": de, "visits_per_arm_clustered": n_ind * de, "territories_per_arm": n_ind * de / m_a})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "pilot_power.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print(f"\nterritories (parsed locality) in the logs: {fv.territory.nunique()}; field agents: {fv.agent_id.nunique()}; "
          f"agent-days: {len(ad)}; mean productive per agent-day {mu:.2f}")


if __name__ == "__main__":
    main()
