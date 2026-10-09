"""Supporting analyses: field-impact estimate, learning curve, pilot sample-size calculation.

impact          How much could a better pin change in the field? Maps the visit-log gradient (old-pin error -> not-traceable
                rate, minutes) onto our error distribution. Observational, indicative only.
learning_curve  Accuracy for addresses with no visit as the system learns from more visited addresses.
pilot_power     Sample sizes for the field pilot, from the supplied logs (ICC by territory and by agent).

Run: python -m geocoder.analysis   ->  outputs/results/impact_*.csv, learning_curve.*, pilot_power.csv
"""
from __future__ import annotations

import os
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .model import Params, PinModel, load_all
from .paths import DATA, OUT, RES


# ======================================================================
# impact
# ======================================================================
BINS = [0, 100, 200, 400, 800, 1e9]
LABELS = ["<100 m", "100-200 m", "200-400 m", "400-800 m", ">800 m"]
CONTACT = {"met_borrower", "cash_collected", "met_family"}


def impact():
    fv = pd.read_csv(DATA / "field_visits.csv")
    base = pd.read_csv(DATA / "baseline_geocodes.csv").set_index("address_id")
    pins = pd.read_csv(OUT / "pins.csv").set_index("address_id")
    hp = pd.read_csv(RES / "hide_and_predict.csv")

    # distance the agent had to bridge: old pin -> confirmed location (own-visit pin)
    v = pins[pins.has_own_visit]
    gap = np.hypot(v.pin_x - base.loc[v.index, "geocoder_x"], v.pin_y - base.loc[v.index, "geocoder_y"])
    fv["gap_m"] = fv.address_id.map(gap)
    fv["minutes"] = (pd.to_datetime(fv.checkin_ts) - pd.to_datetime(fv.start_ts)).dt.total_seconds() / 60
    fv["not_traceable"] = fv.outcome.eq("address_not_traceable")
    fv["productive"] = fv.outcome.isin(CONTACT)
    x = fv.dropna(subset=["gap_m"]).copy()
    x["bin"] = pd.cut(x.gap_m, BINS, labels=LABELS)
    tab = x.groupby("bin", observed=True).agg(visits=("outcome", "size"), not_traceable=("not_traceable", "mean"),
                                              productive=("productive", "mean"), mean_minutes=("minutes", "mean"),
                                              median_minutes=("minutes", "median")).reset_index()
    tab.to_csv(RES / "impact_by_pin_error.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # apply the mapping to the old vs new error distributions on addresses without a visit
    bins = pd.cut(hp.err_old, BINS, labels=LABELS)
    new_bins = pd.cut(hp.err_model, BINS, labels=LABELS)
    m = tab.set_index("bin")
    keys = ["not_traceable", "productive", "mean_minutes"]
    rng = np.random.default_rng(11)

    def expect(bin_series, idx):
        b = bin_series.iloc[idx]
        return {k: float(m.loc[b, k].astype(float).mean()) for k in keys}

    n = len(hp)
    boots = {"old": [], "new": []}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        # also resample the bin table (visit-level) to carry its sampling noise
        xs = x.sample(len(x), replace=True, random_state=int(rng.integers(1e9)))
        mt = xs.groupby("bin", observed=True).agg(nt=("not_traceable", "mean"), pr=("productive", "mean"), mi=("minutes", "mean"))
        for tag, series in [("old", bins), ("new", new_bins)]:
            b = series.iloc[idx]
            vals = mt.reindex(b.values)
            boots[tag].append([vals.nt.mean(), vals.pr.mean(), vals.mi.mean()])
    out = []
    for tag in ("old", "new"):
        a = np.array(boots[tag])
        out.append({"pin": tag, "not_traceable_rate": a[:, 0].mean(), "productive_share": a[:, 1].mean(), "mean_minutes_per_visit": a[:, 2].mean()})
    res = pd.DataFrame(out).set_index("pin")
    # per-agent-day: fixed working time; productive visits/day = (W / minutes per visit) * productive share
    o, nw = np.array(boots["old"]), np.array(boots["new"])
    prod_day_old = (1 / o[:, 2]) * o[:, 1]
    prod_day_new = (1 / nw[:, 2]) * nw[:, 1]
    uplift = prod_day_new / prod_day_old - 1
    nt_drop = (o[:, 0] - nw[:, 0]) / o[:, 0]
    min_drop = (o[:, 2] - nw[:, 2]) / o[:, 2]
    summary = pd.DataFrame([
        {"metric": "address-not-traceable rate (old -> new)", "old": res.at["old", "not_traceable_rate"], "new": res.at["new", "not_traceable_rate"],
         "relative_change": -np.median(nt_drop), "ci95_low": -np.quantile(nt_drop, .975), "ci95_high": -np.quantile(nt_drop, .025)},
        {"metric": "minutes per visit (old -> new)", "old": res.at["old", "mean_minutes_per_visit"], "new": res.at["new", "mean_minutes_per_visit"],
         "relative_change": -np.median(min_drop), "ci95_low": -np.quantile(min_drop, .975), "ci95_high": -np.quantile(min_drop, .025)},
        {"metric": "productive visits per agent-day (relative uplift)", "old": 1.0, "new": 1 + np.median(uplift),
         "relative_change": np.median(uplift), "ci95_low": np.quantile(uplift, .025), "ci95_high": np.quantile(uplift, .975)},
    ])
    summary.to_csv(RES / "impact_estimate.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # share of addresses currently >400 m off that our pin brings within 200 m
    far = hp[hp.err_old > 400]
    print(f"\nAmong no-visit addresses whose old pin is >400 m off ({len(far)}): "
          f"{(far.err_model <= 200).mean():.0%} end up within 200 m of the confirmed location")
    pd.DataFrame([{"far_old_pin_addresses": len(far), "share_brought_within_200m": float((far.err_model <= 200).mean())}]
                 ).to_csv(RES / "impact_far_addresses.csv", index=False)

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    xl = np.arange(len(tab))
    ax[0].bar(xl, tab.not_traceable * 100, color="#c26a3d"); ax[0].set_title("'Address not traceable' rate (%)")
    ax[1].bar(xl, tab.mean_minutes, color="#3b6ea5"); ax[1].set_title("Minutes from start to check-in")
    ax[2].bar(xl, tab.visits, color="#7a8b99"); ax[2].set_title("Visits in bucket")
    for a in ax:
        a.set_xticks(xl); a.set_xticklabels(tab["bin"], rotation=25, fontsize=8); a.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("how far the OLD geocoder pin was from where the borrower was actually met", fontsize=9)
    plt.tight_layout(); plt.savefig(RES / "impact_by_pin_error.png", dpi=140); plt.close()


# ======================================================================
# learning curve
# ======================================================================
FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def learning_curve():
    feat, base, loc, lm, evid, sv, folds = load_all()
    ref = PinModel(Params.load()).fit(feat, base, evid, loc, lm)
    visited = sorted(ref._own)
    old = base.set_index("address_id")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(visited)
        n_test = int(0.3 * len(perm))
        T, pool = list(perm[:n_test]), list(perm[n_test:])
        tgt = np.array([ref._own[a][:2] for a in T], float)
        oldp = old.loc[T, ["geocoder_x", "geocoder_y"]].to_numpy(float)
        e_old = np.hypot(oldp[:, 0] - tgt[:, 0], oldp[:, 1] - tgt[:, 1])
        for f in FRACTIONS:
            keep = set(pool[: int(round(f * len(pool)))])
            ev = evid[evid.address_id.isin(keep)]
            m = PinModel(Params.load()).fit(feat, base, ev, loc, lm)
            pr = m.predict(T)
            e = np.hypot(pr.pin_x.to_numpy() - tgt[:, 0], pr.pin_y.to_numpy() - tgt[:, 1])
            rows.append({"seed": seed, "fraction_of_visits_known": f, "visited_addresses_known": len(keep),
                         "median_error_m": float(np.median(e)), "p90_error_m": float(np.quantile(e, 0.9)),
                         "within_100m": float((e <= 100).mean()), "old_geocoder_median_m": float(np.median(e_old)),
                         "old_geocoder_p90_m": float(np.quantile(e_old, 0.9))})
    d = pd.DataFrame(rows)
    d.to_csv(RES / "learning_curve_runs.csv", index=False)
    s = d.groupby("fraction_of_visits_known").agg(
        visited_addresses_known=("visited_addresses_known", "mean"), median_error_m=("median_error_m", "mean"),
        median_sd=("median_error_m", "std"), p90_error_m=("p90_error_m", "mean"), within_100m=("within_100m", "mean"),
        old_geocoder_median_m=("old_geocoder_median_m", "mean")).reset_index()
    s.to_csv(RES / "learning_curve.csv", index=False)
    print(s.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].errorbar(s.visited_addresses_known, s.median_error_m, yerr=s.median_sd, color="#3b6ea5", marker="o", capsize=3, label="pin model")
    ax[0].axhline(s.old_geocoder_median_m.iloc[0], color="#999999", ls="--", label="old geocoder")
    ax[0].set_xlabel("visited addresses the system has learned from"); ax[0].set_ylabel("median error, addresses with no visit (m)")
    ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("Median error falls as visits accumulate")
    ax[1].plot(s.visited_addresses_known, s.within_100m * 100, color="#2f7d5b", marker="o")
    ax[1].set_xlabel("visited addresses the system has learned from"); ax[1].set_ylabel("share within 100 m (%)"); ax[1].grid(alpha=.3)
    ax[1].set_title("More pins land within 100 m")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(RES / "learning_curve.png", dpi=140); plt.close()


# ======================================================================
# pilot power
# ======================================================================
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


def pilot_power():
    fv = pd.read_csv(DATA / "field_visits.csv")
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


def main():
    impact()
    pilot_power()
    learning_curve()


if __name__ == "__main__":
    main()
