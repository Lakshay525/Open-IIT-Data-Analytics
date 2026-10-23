"""Query engine behind the live demo API.

Fits the same pin model as the pipeline (about 0.4 s) and answers two questions:
  locate(text, town)            where is this free-text address? (pin, calibrated radii, tier, directions)
  what_if_visit(address, x, y)  if an agent confirms the borrower at (x, y), how do the neighbours' pins move?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parent
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from geocoder import confidence as conf            # noqa: E402
from geocoder.model import Params, PinModel        # noqa: E402
from geocoder.parser import parse_addresses        # noqa: E402

D = LIB / "data"
NEW_ID = "NEW-0001"
_cache: dict = {}


def _load() -> dict:
    if _cache:
        return _cache
    _cache["feat"] = pd.read_csv(D / "address_features.csv")
    _cache["base"] = pd.read_csv(D / "baseline_geocodes.csv")
    _cache["loc"] = pd.read_csv(D / "localities.csv")
    _cache["lm"] = pd.read_csv(D / "landmarks_poi.csv")
    _cache["towns"] = pd.read_csv(D / "towns.csv")
    _cache["evid"] = pd.read_csv(D / "evidence.csv")
    _cache["addr"] = pd.read_csv(D / "addresses.csv").set_index("address_id")
    _cache["truth"] = pd.read_csv(D / "surveyed_addresses.csv")
    _cache["quant"] = pd.read_csv(D / "calibration_quantiles.csv")
    _cache["params"] = Params.load(D / "pin_model_params.json")
    _cache["dirs"] = conf.DirectionBuilder(_cache["lm"], _cache["loc"], _cache["towns"])
    return _cache


def _fit(feat=None, base=None, evid=None, truth=None) -> PinModel:
    c = _load()
    return PinModel(c["params"]).fit(c["feat"] if feat is None else feat, c["base"] if base is None else base,
                                     c["evid"] if evid is None else evid, c["loc"], c["lm"], truth=truth)


def _radii(group: str, tier: str, spread: float) -> tuple[float, float, float]:
    q = _load()["quant"]
    row = q[(q.group == group) & (q.tier == tier)]
    if row.empty:
        row = q[(q.group == group) & (q.tier == "*")]
    r50 = max(float(row.Q50.iloc[0]) * spread, 5.0)
    r90 = max(float(row.Q90.iloc[0]) * spread, r50 * 1.2)
    r80 = float(np.clip(float(row.Q80.iloc[0]) * spread, r50, r90))
    return r50, r80, r90


def _tier(r90: float) -> str:
    return "high" if r90 <= conf.TIER_HIGH_M else ("medium" if r90 <= conf.TIER_MED_M else "low")


def _reasons(clues: str, n_hyp: int, precision: str, visit: bool) -> list[str]:
    codes = []
    if visit:
        codes.append("CONFIRMED_BY_FIELD_VISIT")
    elif n_hyp > 1:
        codes.append("NO_LOCALITY_IN_TEXT")
    elif any(k in clues for k in ("grid", "cm", "br")):
        codes.append("INFERRED_FROM_STREET_NEIGHBOURS")
    elif "lm" in clues or "poi" in clues:
        codes.append("INFERRED_FROM_LANDMARK")
    else:
        codes.append("LOCALITY_LEVEL_ONLY")
    if precision == "pincode" and not visit:
        codes.append("OLD_GEOCODER_AT_PINCODE_CENTRE")
    return codes


def locate(text: str, town_id: str) -> dict:
    """Locate a free-text address inside one of the three towns."""
    c = _load()
    text = " ".join(str(text).split())[:300]
    if town_id not in set(c["towns"].town_id):
        return {"error": f"unknown town {town_id}"}
    if len(text) < 6:
        return {"error": "please enter a fuller address"}
    row = pd.DataFrame([{"address_id": NEW_ID, "town_id": town_id, "address_text": text}])
    parsed = parse_addresses(row, c["loc"])
    p = parsed.iloc[0]
    loc = c["loc"]
    # what the commercial geocoder would have done: the locality centre, or the pincode centre when no locality is named
    if pd.notna(p.get("loc_id")):
        cen = loc[loc.locality_id == p.loc_id].iloc[0]
        old, precision = (float(cen.centroid_x), float(cen.centroid_y)), "locality"
    else:
        pin = p.get("pincode_text")
        sub = loc[(loc.town_id == town_id) & (loc.pincode.astype(str) == str(pin))] if pd.notna(pin) else loc[loc.town_id == town_id]
        if sub.empty:
            sub = loc[loc.town_id == town_id]
        old, precision = (float(sub.centroid_x.mean()), float(sub.centroid_y.mean())), "pincode"
    feat = pd.concat([c["feat"], parsed], ignore_index=True)
    base = pd.concat([c["base"], pd.DataFrame([{"address_id": NEW_ID, "geocoder_x": old[0], "geocoder_y": old[1], "precision": precision}])], ignore_index=True)
    model = _fit(feat=feat, base=base)
    pr = model.predict([NEW_ID]).iloc[0]
    spread = float(pr.raw_spread)
    group = "noloc" if int(pr.n_locality_hyp) > 1 else "novisit"
    r50, r80, r90 = _radii(group, precision, spread)
    tier = _tier(r90)
    pin_xy = np.array([float(pr.pin_x), float(pr.pin_y)])
    street = conf.street_summary(p)
    en, local, meta = c["dirs"].build(town_id, pin_xy, r90, tier, p.get("lm_type"), street, False)
    keys = {k: (None if pd.isna(p.get(k)) else str(p.get(k))) for k in ("loc_id", "pincode_text", "cross", "main", "gali", "ward", "block", "road", "lm_type", "lm_rel")}
    loc_name = None
    if keys["loc_id"]:
        loc_name = str(loc[loc.locality_id == keys["loc_id"]].locality_name.iloc[0])
    return {
        "text": text, "town_id": town_id, "parsed": keys, "locality_name": loc_name, "script": str(p.get("script")),
        "old_pin": {"x": round(old[0], 1), "y": round(old[1], 1), "precision": precision},
        "pin": {"x": round(float(pin_xy[0]), 1), "y": round(float(pin_xy[1]), 1)},
        "moved_m": round(float(np.hypot(pin_xy[0] - old[0], pin_xy[1] - old[1])), 1),
        "raw_spread": round(spread, 1), "radius_50": round(r50, 1), "radius_80": round(r80, 1), "radius_90": round(r90, 1),
        "tier": tier, "action": conf.ACTIONS[tier][0], "advice": conf.ACTIONS[tier][1],
        "method": str(pr.method_used), "clues": str(pr.clues_used), "n_candidate_localities": int(pr.n_locality_hyp),
        "reason_codes": _reasons(str(pr.clues_used), int(pr.n_locality_hyp), precision, False),
        "directions": en, "directions_local": local, "landmarks": meta.get("landmarks", []),
    }


def what_if_visit(address_id: str, x: float, y: float, uncertainty: float = 60.0) -> dict:
    """Log a confirmed visit for one address and report how the pins of its unvisited neighbours move."""
    c = _load()
    if address_id not in set(c["feat"].address_id):
        return {"error": f"unknown address {address_id}"}
    model = _fit()
    f = model.feat
    loc_id = f.at[address_id, "loc_id"]
    neigh = [a for a in f.index[(f.loc_id == loc_id)] if a != address_id and a not in model._own] if pd.notna(loc_id) else []
    ids = [address_id] + neigh
    before = model.predict(ids).set_index("address_id")
    new = pd.DataFrame([{"visit_id": "VS-LIVE", "address_id": address_id, "ev_x": float(x), "ev_y": float(y), "weight": 0.5,
                         "eligible_for_pin": True, "evidence_role": "contact_at_address", "uncertainty_m": float(uncertainty)}])
    model.update(new)
    after = model.predict(ids).set_index("address_id")
    truth = c["truth"].set_index("address_id")
    out = []
    for a in neigh:
        b, af = before.loc[a], after.loc[a]
        moved = float(np.hypot(af.pin_x - b.pin_x, af.pin_y - b.pin_y))
        if moved < 3:
            continue
        item = {"id": a, "text": str(c["addr"].at[a, "address_text"]), "before": [round(float(b.pin_x), 1), round(float(b.pin_y), 1)],
                "after": [round(float(af.pin_x), 1), round(float(af.pin_y), 1)], "moved_m": round(moved, 1),
                "spread_before": round(float(b.raw_spread), 1), "spread_after": round(float(af.raw_spread), 1)}
        if a in truth.index:
            t = truth.loc[a]
            item["truth"] = [round(float(t.surveyed_x), 1), round(float(t.surveyed_y), 1)]
            item["err_before"] = round(float(np.hypot(b.pin_x - t.surveyed_x, b.pin_y - t.surveyed_y)), 1)
            item["err_after"] = round(float(np.hypot(af.pin_x - t.surveyed_x, af.pin_y - t.surveyed_y)), 1)
        out.append(item)
    out.sort(key=lambda d: -d["moved_m"])
    t_b, t_a = before.loc[address_id], after.loc[address_id]
    return {
        "address_id": address_id, "visit": [round(float(x), 1), round(float(y), 1)],
        "target": {"before": [round(float(t_b.pin_x), 1), round(float(t_b.pin_y), 1)], "after": [round(float(t_a.pin_x), 1), round(float(t_a.pin_y), 1)],
                   "spread_before": round(float(t_b.raw_spread), 1), "spread_after": round(float(t_a.raw_spread), 1)},
        "neighbours_checked": len(neigh), "neighbours_moved": len(out), "neighbours": out[:30],
        "locality": str(c["loc"][c["loc"].locality_id == loc_id].locality_name.iloc[0]) if pd.notna(loc_id) else None,
    }


def health() -> dict:
    c = _load()
    return {"ok": True, "addresses": int(len(c["feat"])), "visits": int(len(c["evid"]))}


def dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8")
