"""Task 4: calibrated confidence radii, confidence tiers, landmark directions, PS2 export.

Inputs : pins.csv (Task 3), results/hide_and_predict.csv and results/pin_model_oof_errors.csv (Task 3)
Outputs: predictions.csv            main deliverable
         ps2_location_confidence.csv  hand-off for Problem Statement 2
         offline_pack/<town>.json   compact per-town bundle for the field app (works with no network)
         results/calibration_*.csv  coverage tables used by calibration.ipynb

Calibration method (split / normalised conformal prediction)
-------------------------------------------------------------
The pin model reports `raw_spread` (its own sigma). For a group g of addresses we collect the
normalised score  s = error / raw_spread  on data the model did NOT see, and set

        radius_p = Q_g(p) * raw_spread          Q_g(p) = finite-sample-corrected quantile of s

so "radius_90" contains the true location about 90% of the time *in that group*. Groups are the three
situations that behave differently:

    visit   the address has its own cleaned visit evidence
    novisit parsed locality, no visit of its own
    noloc   no locality in the text (pincode leaves several candidates)

and `novisit` is further split by precision tier when there are enough samples.
Calibration data (never the data used to report coverage):
    novisit / noloc : the 1,231-address hide-and-predict set (visit pseudo-labels, no surveyed truth)
    visit           : surveyed out-of-fold errors, leave-one-out (49 addresses -> wide intervals, stated)
Coverage is then reported on the surveyed addresses, which were not used to calibrate novisit/noloc.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "clean_data"
RES = ROOT / "results"

LEVELS = (0.5, 0.8, 0.9)
TIER_HIGH_M = 60.0       # radius_90 <= 60 m  -> high confidence
TIER_MED_M = 300.0       # radius_90 <= 300 m -> medium, else low
MIN_TIER_N = 150         # per-tier calibration needs at least this many samples
CONF_WITHIN_M = 100.0    # location_confidence = P(true location within 100 m)

# x east, y north is an ASSUMPTION (the source has no CRS). Flip here if CN confirms otherwise.
AXES = {"x_is_east": True, "y_is_north": True}

ACTIONS = {
    "high": ("VISIT_NOW", "Pin is reliable. Plan the visit; show directions in the field app."),
    "medium": ("VISIT_WITH_DIRECTIONS",
               "Likely right but not certain. Visit with the landmark directions; ask a neighbour at the pin first."),
    "low": ("VERIFY_FIRST",
            "Do not send an agent blind and do not mark 'not traceable'. Confirm a landmark with the borrower/"
            "reference or a quick local check first."),
}


# --------------------------------------------------------------------------
# conformal calibration
# --------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, p: float) -> float:
    """Finite-sample-corrected empirical quantile (split conformal)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(n, math.ceil((n + 1) * p))
    return float(s[k - 1])


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def assign_group(has_visit: pd.Series, n_hyp: pd.Series) -> pd.Series:
    g = np.where(has_visit, "visit", np.where(n_hyp > 1, "noloc", "novisit"))
    return pd.Series(g, index=has_visit.index)


class Calibrator:
    """Holds Q_g(p) for each (group, tier) cell with fallbacks to the pooled group."""

    def __init__(self):
        self.q: dict[tuple, dict[float, float]] = {}
        self.cdf: dict[tuple, np.ndarray] = {}
        self.n: dict[tuple, int] = {}

    def fit(self, df: pd.DataFrame):
        """df columns: group, precision, score"""
        for g, dg in df.groupby("group"):
            self._add((g, "*"), dg.score.to_numpy())
            for tier, dt in dg.groupby("precision"):
                if len(dt) >= MIN_TIER_N:
                    self._add((g, tier), dt.score.to_numpy())
        return self

    def _add(self, key, scores):
        self.q[key] = {p: conformal_quantile(scores, p) for p in LEVELS}
        self.cdf[key] = np.sort(scores)
        self.n[key] = len(scores)

    def key_for(self, group, tier):
        return (group, tier) if (group, tier) in self.q else (group, "*")

    def radius(self, group, tier, raw_spread, p):
        return self.q[self.key_for(group, tier)][p] * raw_spread

    def prob_within(self, group, tier, raw_spread, metres):
        """Calibrated P(error <= metres): empirical CDF of the normalised score."""
        c = self.cdf[self.key_for(group, tier)]
        r = metres / np.maximum(raw_spread, 1e-9)
        return np.searchsorted(c, r, side="right") / (len(c) + 1)


def loo_visit_scores(df_visit: pd.DataFrame):
    """Leave-one-out conformal radii for the small surveyed visit group."""
    out = {}
    s = df_visit.score.to_numpy()
    for i, a in enumerate(df_visit.index):
        rest = np.delete(s, i)
        out[a] = {p: conformal_quantile(rest, p) for p in LEVELS}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COMPASS_LOCAL = {
    "karnataka": {"N": "uttara", "NE": "eesanya", "E": "poorva", "SE": "agneya", "S": "dakshina", "SW": "nairutya",
                  "W": "pashchima", "NW": "vayavya"},
    "hindi": {"N": "uttar", "NE": "uttar-purab", "E": "purab", "SE": "dakshin-purab", "S": "dakshin",
              "SW": "dakshin-paschim", "W": "paschim", "NW": "uttar-paschim"},
}
STYLE_PHRASE = {  # near-landmark phrase by address style (Latin-script local flavour, for field staff)
    "karnataka": "{lm} hattira, {dist} m {dir}",
    "hindi": "{lm} ke paas, {dist} m {dir} ki taraf",
    "metro": "{dist} m {dir} of {lm}",
}
TYPE_LABEL = {
    "ration_shop": "Ration Shop", "govt_school": "Govt School", "bus_stop": "Bus Stop",
    "hanuman_temple": "Hanuman Temple", "masjid": "Masjid", "medical_store": "Medical Store",
    "ganesha_temple": "Ganesh Temple", "water_tank": "Water Tank", "milk_dairy": "Milk Dairy",
    "community_hall": "Community Hall", "park": "Park", "church": "Church", "post_office": "Post Office",
    "petrol_bunk": "Petrol Bunk",
}


def bearing_label(dx: float, dy: float) -> str:
    """Compass direction of the vector landmark -> pin."""
    if not AXES["x_is_east"]:
        dx = -dx
    if not AXES["y_is_north"]:
        dy = -dy
    ang = (math.degrees(math.atan2(dx, dy)) + 360) % 360  # 0 = north, clockwise
    return COMPASS[int(((ang + 22.5) % 360) // 45)]


def round_m(d: float) -> int:
    if d < 30:
        return int(max(10, round(d / 10) * 10))
    return int(round(d / 10) * 10) if d <= 200 else int(round(d / 50) * 50)


class DirectionBuilder:
    def __init__(self, landmarks: pd.DataFrame, localities: pd.DataFrame, towns: pd.DataFrame):
        self.lm = landmarks.reset_index(drop=True)
        self.lm_xy = self.lm[["x", "y"]].to_numpy(float)
        self.loc = localities
        self.style = towns.set_index("town_id").address_style.to_dict()
        self.by_town = {t: g.index.to_numpy() for t, g in self.lm.groupby("town_id")}
        lc = localities[["centroid_x", "centroid_y"]].to_numpy(float)
        names = []
        for r in self.lm.itertuples():
            cand = localities[localities.town_id == r.town_id]
            d = np.hypot(cand.centroid_x - r.x, cand.centroid_y - r.y)
            names.append(cand.locality_name.iloc[int(np.argmin(d.to_numpy()))])
        self.lm["near_locality"] = names

    def _label(self, idx: int, pin: np.ndarray, town_idx: np.ndarray) -> tuple[str, bool]:
        """Landmark display name, disambiguated by locality if the same name exists close to the pin."""
        row = self.lm.iloc[idx]
        same = town_idx[(self.lm.name.values[town_idx] == row["name"])]
        d = np.hypot(self.lm_xy[same, 0] - pin[0], self.lm_xy[same, 1] - pin[1])
        close = int((d < 900).sum())
        if close > 1:
            return f"{row['name']} ({row['near_locality']} side)", True
        return str(row["name"]), False

    def build(self, town, pin, radius90, tier, address_lm_type, street_bits, has_visit):
        town_idx = self.by_town.get(town)
        if town_idx is None:
            return "", "", {}
        d = np.hypot(self.lm_xy[town_idx, 0] - pin[0], self.lm_xy[town_idx, 1] - pin[1])
        order = np.argsort(d)
        chosen = []
        # 1) the landmark the address itself names, if one of that type is plausibly near the pin
        if isinstance(address_lm_type, str):
            for j in order[:12]:
                gi = town_idx[j]
                if self.lm.at[gi, "landmark_type"] == address_lm_type and d[j] <= max(350.0, 1.5 * radius90):
                    chosen.append((gi, d[j], True))
                    break
        # 2) fill with the nearest landmarks of other types (<= 3 total)
        used_types = {self.lm.at[c[0], "landmark_type"] for c in chosen}
        for j in order[:25]:
            if len(chosen) >= 3:
                break
            gi = town_idx[j]
            t = self.lm.at[gi, "landmark_type"]
            if t in used_types or d[j] > (700 if not chosen else 500):
                continue
            chosen.append((gi, d[j], False))
            used_types.add(t)
        parts_en, parts_loc, meta = [], [], []
        style = self.style.get(town, "metro")
        for k, (gi, dist, from_addr) in enumerate(chosen[:3]):
            name, dup = self._label(gi, pin, town_idx)
            dx, dy = pin[0] - self.lm_xy[gi, 0], pin[1] - self.lm_xy[gi, 1]
            dirn = bearing_label(dx, dy)
            m = round_m(dist)
            if m <= 25:
                en = f"at {name}"
                loc = en
            else:
                en = f"{m} m {dirn} of {name}"
                loc_dir = COMPASS_LOCAL.get(style, {}).get(dirn, dirn)
                loc = STYLE_PHRASE.get(style, STYLE_PHRASE["metro"]).format(lm=name, dist=m, dir=loc_dir)
            if k > 0:
                en = "also " + en
            parts_en.append(en)
            parts_loc.append(loc)
            meta.append({"poi_id": self.lm.at[gi, "poi_id"], "name": name, "dist_m": m, "dir": dirn,
                         "from_address": bool(from_addr), "ambiguous_name": bool(dup)})
        en_txt = "; ".join(parts_en) if parts_en else "No landmark within 700 m"
        loc_txt = "; ".join(parts_loc) if parts_loc else ""
        if street_bits:
            en_txt += f". Address: {street_bits}"
        if tier == "low":
            en_txt = f"APPROXIMATE (within ~{int(round(radius90, -1)):d} m): " + en_txt + ". Verify with a local before walking in."
        elif tier == "medium":
            en_txt = f"Likely area (within ~{int(round(radius90, -1)):d} m): " + en_txt
        return en_txt, loc_txt, {"landmarks": meta}


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def street_summary(f) -> str:
    bits = []
    if pd.notna(f.get("cross")) and pd.notna(f.get("main")):
        bits.append(f"{ordinal(f['cross'])} Cross, {ordinal(f['main'])} Main")
    if pd.notna(f.get("block")) and pd.notna(f.get("road")):
        bits.append(f"Block {str(f['block']).upper()}, Road {int(f['road'])}")
    if pd.notna(f.get("gali")):
        bits.append(f"Gali {int(f['gali'])}" + (f", Ward {int(f['ward'])}" if pd.notna(f.get("ward")) else ""))
    return "; ".join(bits)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    pins = pd.read_csv(ROOT / "pins.csv").set_index("address_id")
    hp = pd.read_csv(RES / "hide_and_predict.csv").set_index("address_id")
    oof = pd.read_csv(RES / "pin_model_oof_errors.csv").set_index("address_id")
    feat = pd.read_csv(RES / "address_features.csv").set_index("address_id")
    base = pd.read_csv(DATA / "baseline_geocodes.csv").set_index("address_id")
    addr = pd.read_csv(DATA / "addresses.csv").set_index("address_id")
    landmarks = pd.read_csv(DATA / "landmarks_poi.csv")
    localities = pd.read_csv(DATA / "localities.csv")
    towns = pd.read_csv(DATA / "towns.csv")
    sv = pd.read_csv(DATA / "surveyed_addresses.csv").set_index("address_id")

    pins["precision"] = base.loc[pins.index, "precision"]
    pins["group"] = assign_group(pins.has_own_visit, pins.n_locality_hyp)

    # ---- calibration data
    hp["score"] = hp.err_model / hp.raw_spread
    hp["group"] = np.where(hp.n_locality_hyp > 1, "noloc", "novisit")
    cal_hp = hp[["group", "precision", "score"]]
    sv_df = oof[["error_m"]].join(pins[["raw_spread", "group", "precision", "has_own_visit"]])
    sv_df["score"] = sv_df.error_m / sv_df.raw_spread
    visit_cal = sv_df[sv_df.group == "visit"]

    calib = Calibrator().fit(cal_hp)
    # visit group is calibrated on surveyed visited addresses (LOO for their own rows)
    vcal = Calibrator()
    vcal._add(("visit", "*"), visit_cal.score.to_numpy())
    calib.q.update(vcal.q), calib.cdf.update(vcal.cdf), calib.n.update(vcal.n)
    loo = loo_visit_scores(visit_cal)

    # ---- radii for every address
    for p in LEVELS:
        pins[f"r{int(p*100)}"] = [calib.radius(g, t, s, p) for g, t, s in zip(pins.group, pins.precision, pins.raw_spread)]
    # surveyed visited rows: leave-one-out radii (their own error must not set their own radius)
    for a in loo.index:
        for p in LEVELS:
            pins.at[a, f"r{int(p*100)}"] = loo.at[a, p] * pins.at[a, "raw_spread"]
    pins["location_confidence"] = [float(calib.prob_within(g, t, s, CONF_WITHIN_M))
                                   for g, t, s in zip(pins.group, pins.precision, pins.raw_spread)]
    pins["p_within_250"] = [float(calib.prob_within(g, t, s, 250.0))
                            for g, t, s in zip(pins.group, pins.precision, pins.raw_spread)]
    # visited surveyed rows: LOO probability as well
    for a in loo.index:
        sc = np.sort(np.delete(visit_cal.score.to_numpy(), list(visit_cal.index).index(a)))
        pins.at[a, "location_confidence"] = np.searchsorted(sc, CONF_WITHIN_M / pins.at[a, "raw_spread"], side="right") / (len(sc) + 1)
        pins.at[a, "p_within_250"] = np.searchsorted(sc, 250.0 / pins.at[a, "raw_spread"], side="right") / (len(sc) + 1)

    # a radius is never smaller than GPS-level noise
    pins["r50"] = pins.r50.clip(lower=5.0)
    pins["r90"] = np.maximum(pins.r90, pins.r50 * 1.2)
    pins["r80"] = np.clip(pins.r80, pins.r50, pins.r90)

    pins["confidence_tier"] = np.where(pins.r90 <= TIER_HIGH_M, "high", np.where(pins.r90 <= TIER_MED_M, "medium", "low"))
    pins["suggested_action"] = pins.confidence_tier.map(lambda t: ACTIONS[t][0])

    # ---- reason codes (explainability for agents, supervisors, auditors)
    reasons = []
    for a, r in pins.iterrows():
        f = feat.loc[a]
        codes = []
        if r.has_own_visit:
            codes.append("CONFIRMED_BY_FIELD_VISIT")
        elif r.n_locality_hyp > 1:
            codes.append("NO_LOCALITY_IN_TEXT")
        else:
            if str(r.clues_used).find("grid") >= 0 or str(r.clues_used).find("cm") >= 0 or str(r.clues_used).find("br") >= 0:
                codes.append("INFERRED_FROM_STREET_NEIGHBOURS")
            elif str(r.clues_used).find("lm") >= 0 or str(r.clues_used).find("poi") >= 0:
                codes.append("INFERRED_FROM_LANDMARK")
            else:
                codes.append("LOCALITY_LEVEL_ONLY")
        if r.precision == "pincode" and not r.has_own_visit:
            codes.append("OLD_GEOCODER_AT_PINCODE_CENTRE")
        reasons.append("|".join(codes))
    pins["reason_codes"] = reasons

    # ---- directions
    db = DirectionBuilder(landmarks, localities, towns)
    en, loc_txt, lm_meta = [], [], []
    for a, r in pins.iterrows():
        f = feat.loc[a]
        t, tl, m = db.build(addr.at[a, "town_id"], np.array([r.pin_x, r.pin_y]), r.r90, r.confidence_tier,
                            f.get("lm_type"), street_summary(f), bool(r.has_own_visit))
        en.append(t), loc_txt.append(tl), lm_meta.append(m)
    pins["directions"] = en
    pins["directions_local"] = loc_txt

    # ---- assemble deliverables
    out = pd.DataFrame({
        "address_id": pins.index,
        "pin_x": pins.pin_x.round(1).values, "pin_y": pins.pin_y.round(1).values,
        "radius_50": pins.r50.round(1).values, "radius_90": pins.r90.round(1).values,
        "directions": pins.directions.values, "confidence_tier": pins.confidence_tier.values,
        "suggested_action": pins.suggested_action.values,
        "location_confidence": pins.location_confidence.round(3).values,
        "radius_80": pins.r80.round(1).values,
        "reason_codes": pins.reason_codes.values, "method_used": pins.method_used.values,
        "raw_spread": pins.raw_spread.round(1).values,
        "town_id": addr.loc[pins.index, "town_id"].values, "precision_old_geocoder": pins.precision.values,
        "old_pin_x": base.loc[pins.index, "geocoder_x"].round(1).values,
        "old_pin_y": base.loc[pins.index, "geocoder_y"].round(1).values,
        "directions_local": pins.directions_local.values,
        "has_own_visit": pins.has_own_visit.values,
        "pin_source": pins.pin_source.values,
    })
    out.to_csv(ROOT / "predictions.csv", index=False)

    ps2 = pd.DataFrame({
        "address_id": pins.index,
        "location_confidence": pins.location_confidence.round(3).values,
        "p_within_250m": pins.p_within_250.round(3).values,
        "radius_90_m": pins.r90.round(1).values,
        "confidence_tier": pins.confidence_tier.values,
        "hard_to_find_flag": (pins.confidence_tier.eq("low") & ~pins.has_own_visit).astype(int).values,
        "reason_codes": pins.reason_codes.values,
    })
    ps2["ps2_guidance"] = np.where(
        ps2.hard_to_find_flag == 1,
        "LOCATION_UNRESOLVED: do not treat a failed visit here as evidence the address is invalid",
        np.where(ps2.confidence_tier == "high", "LOCATION_CONFIRMED_OR_RELIABLE: a failed visit is informative",
                 "LOCATION_APPROXIMATE: a failed visit is weak evidence; retry with directions"))
    ps2.to_csv(ROOT / "ps2_location_confidence.csv", index=False)

    # ---- offline pack per town
    (ROOT / "offline_pack").mkdir(exist_ok=True)
    for town, g in out.groupby("town_id"):
        pack = {
            "town_id": town, "units": "metres, local x/y (x east, y north assumed)",
            "addresses": [{"id": r.address_id, "x": r.pin_x, "y": r.pin_y, "r50": r.radius_50, "r90": r.radius_90,
                           "tier": r.confidence_tier, "dir": r.directions} for r in g.itertuples()],
            "landmarks": landmarks[landmarks.town_id == town][["poi_id", "name", "x", "y"]].round(1).to_dict("records"),
        }
        (ROOT / "offline_pack" / f"{town}.json").write_text(json.dumps(pack, separators=(",", ":")), encoding="utf-8")

    # ---- calibration report (held-out) -----------------------------------------
    rows = []
    # novisit / noloc: calibrated on hide-and-predict, scored on surveyed OOF
    ev = sv_df.join(pins[["r50", "r80", "r90", "group", "precision"]], rsuffix="_p")
    ev["town_id"] = addr.loc[ev.index, "town_id"]
    ev["tier"] = pins.loc[ev.index, "confidence_tier"]
    for p, col in [(0.5, "r50"), (0.8, "r80"), (0.9, "r90")]:
        ev[f"in{int(p*100)}"] = ev.error_m <= ev[col]

    def add(name, d, dim):
        for p in (50, 80, 90):
            k, n = int(d[f"in{p}"].sum()), len(d)
            lo, hi = wilson(k, n)
            rows.append({"dimension": dim, "segment": name, "n": n, "nominal": p / 100, "coverage": k / n if n else np.nan,
                         "ci95_low": lo, "ci95_high": hi, "median_radius_90": float(d.r90.median()) if n else np.nan,
                         "median_error": float(d.error_m.median()) if n else np.nan})
    add("all surveyed", ev, "overall")
    for g, d in ev.groupby("group"):
        add(g, d, "group")
    for g, d in ev.groupby("town_id"):
        add(g, d, "town")
    for g, d in ev.groupby("precision"):
        add(g, d, "old_geocoder_tier")
    for g, d in ev.groupby("tier"):
        add(g, d, "confidence_tier")
    calt = pd.DataFrame(rows)
    calt.to_csv(RES / "calibration_coverage.csv", index=False)
    ev.to_csv(RES / "calibration_surveyed_detail.csv")

    # reliability curve for the held-out groups (nominal vs observed)
    rel = []
    vs = visit_cal.score.to_numpy()
    nv = ev[ev.group != "visit"]
    vv = ev[ev.group == "visit"]
    for p in np.linspace(0.1, 0.95, 18):
        cov_nv = np.mean([e <= conformal_quantile(calib.cdf[calib.key_for(g, t)], p) * sp
                          for e, g, t, sp in zip(nv.error_m, nv.group, nv.precision, nv.raw_spread)])
        cov_v = np.mean([e <= conformal_quantile(np.delete(vs, i), p) * sp
                         for i, (e, sp) in enumerate(zip(vv.error_m, vv.raw_spread))])
        rel.append({"curve": "unvisited (calibrated on hide-and-predict, scored on surveyed)", "nominal": p, "observed": cov_nv, "n": len(nv)})
        rel.append({"curve": "visited (leave-one-out)", "nominal": p, "observed": cov_v, "n": len(vv)})
    pd.DataFrame(rel).to_csv(RES / "calibration_reliability.csv", index=False)

    # calibration set quantiles (for audit)
    qrows = [{"group": k[0], "tier": k[1], "n": calib.n[k], **{f"Q{int(p*100)}": v for p, v in calib.q[k].items()}}
             for k in sorted(calib.q)]
    pd.DataFrame(qrows).to_csv(RES / "calibration_quantiles.csv", index=False)

    # ---- tier quality
    tq = ev.groupby("tier").agg(n=("error_m", "size"), median_err=("error_m", "median"),
                                within_100=("error_m", lambda x: (x <= 100).mean()), within_250=("error_m", lambda x: (x <= 250).mean()))
    tq.to_csv(RES / "confidence_tier_quality.csv")

    # ---- directions sanity
    ds = directions_checks(pins, out, feat, sv, landmarks, db)
    pd.DataFrame([ds]).to_csv(RES / "directions_checks.csv", index=False)

    print(f"predictions.csv: {len(out)} rows | tiers: {out.confidence_tier.value_counts().to_dict()}")
    print(calt[(calt.dimension.isin(['overall', 'group'])) & (calt.nominal.isin([0.5, 0.9]))].round(3).to_string(index=False))
    print(tq.round(2).to_string())
    print("directions checks:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ds.items()})


def directions_checks(pins, out, feat, sv, landmarks, db):
    """Objective checks that directions are consistent with the address text and uniquely identifiable."""
    res = {}
    res["share_with_a_landmark"] = float((~out.directions.str.contains("No landmark")).mean())
    # does the nearest same-type POI to the true location match the landmark our pin would pick?
    hit, tot, near_truth, near_pin = 0, 0, 0, 0
    for a in sv.index:
        lt = feat.at[a, "lm_type"]
        if not isinstance(lt, str):
            continue
        town = out.set_index("address_id").at[a, "town_id"]
        sub = landmarks[(landmarks.town_id == town) & (landmarks.landmark_type == lt)]
        if sub.empty:
            continue
        t = sv.loc[a].to_numpy(float)
        p = out.set_index("address_id").loc[a, ["pin_x", "pin_y"]].to_numpy(float)
        dt = np.hypot(sub.x - t[0], sub.y - t[1])
        dp = np.hypot(sub.x - p[0], sub.y - p[1])
        tot += 1
        hit += int(dt.idxmin() == dp.idxmin())
        near_truth += int(dt.min() <= 300)
        near_pin += int(dp.min() <= 300)
    res["surveyed_with_address_landmark"] = tot
    res["address_landmark_within_300m_of_truth"] = near_truth / max(tot, 1)
    res["address_landmark_within_300m_of_our_pin"] = near_pin / max(tot, 1)
    res["same_landmark_chosen_for_pin_and_truth"] = hit / max(tot, 1)
    amb = out.directions.str.contains(r"\(.* side\)")
    res["share_directions_needing_locality_disambiguation"] = float(amb.mean())
    return res


if __name__ == "__main__":
    main()
