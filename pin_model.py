"""Task 3b: the pin model.

What it is
----------
A *precision-weighted evidence-fusion* model (empirical-Bayes / Gaussian product of
experts) with robust gating. Every clue about where an address is gives a guess
(mu) and an honest uncertainty (sigma); the model combines them, trusting the
precise clues more and dropping a clue that contradicts the rest.

    slot            clue                                             typical sigma
    --------------  -----------------------------------------------  -------------
    visit           this address's own cleaned field-visit evidence   10-30 m
    cm / br         same locality + cross&main / block&road           50-100 m
    gali            same locality + gali (+ward)                       ~200 m
    lmcm / lmbr     same street key AND same landmark type            tight
    lm              same locality + same landmark type                 ~170 m
    locgrp          known pins in the same locality                    ~300 m
    loccent         locality centroid                                  ~300 m
    poi             landmark POIs of the named type near the old pin   ~200 m
    baseline        old geocoder pin (sigma depends on precision tier) 20-1500 m

"Neighbour" clues use the *known locations* table: cleaned visit pins of other
addresses (about 43% of addresses) plus, for evaluation, surveyed truths of
training folds only. Because one confirmed visit becomes a clue for every address
sharing its street / landmark, each new visit improves many addresses, and
`update()` adds evidence without retraining.

Addresses whose text has no locality are handled with *locality hypotheses*
(candidate localities from the pincode) scored by Bayesian evidence.

All sigmas / gate settings live in `Params` and are fitted by `tune_pin_model.py`
using only leave-one-out pseudo-labels from field visits -- never surveyed truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "clean_data"

SLOTS = ["baseline", "visit", "grid", "cm", "br", "gali", "lmcm", "lmbr", "lm", "locgrp", "loccent", "poi"]
SLOT_IDX = {s: i for i, s in enumerate(SLOTS)}
VISIT_GATE_WEAK = 2.5
NEIGHBOUR_SLOTS = ["cm", "br", "gali", "lmcm", "lmbr", "lm", "locgrp"]
TIERS = ["rooftop", "street", "locality", "pincode"]
METHOD_OF = {"baseline": "baseline", "visit": "own_visit", "grid": "street_grid", "cm": "street_key", "br": "street_key",
             "gali": "street_key", "lmcm": "street_key", "lmbr": "street_key", "lm": "landmark_group",
             "locgrp": "locality_group", "loccent": "locality_centroid", "poi": "landmark_poi"}


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    # sigma (m) of the old geocoder pin by precision tier
    sb_rooftop: float = 10.0
    sb_street: float = 115.0
    sb_locality: float = 330.0
    sb_pincode: float = 1250.0
    # base sigma (m) of each neighbour / locality clue
    a_cm: float = 55.0
    a_br: float = 85.0
    a_gali: float = 200.0
    a_lmcm: float = 55.0
    a_lmbr: float = 85.0
    a_lm: float = 170.0
    a_locgrp: float = 300.0
    a_loccent: float = 320.0
    a_poi: float = 220.0
    # street-number grid regression: sigma^2 = (a_grid * predictive_spread)^2 + f_grid^2
    a_grid: float = 0.8
    f_grid: float = 30.0
    # own visit: sigma^2 = a_visit^2 + (b_visit * spread)^2
    a_visit: float = 15.0
    b_visit: float = 0.6
    # neighbour clue: sigma^2 = (a*(1 + c_n/n))^2 + (b_scatter*scatter)^2
    c_n: float = 0.8
    b_scatter: float = 0.5
    # robust gate: drop the clue most inconsistent with the others if above k sigma
    gate_k: float = 3.0
    # posterior spread inflation (calibration is finished in Task 4)
    inflate: float = 1.0
    # locality-hypothesis softness (>1 flattens)
    hyp_temp: float = 1.0
    # which clues are switched on (used by the ablation study)
    enabled: tuple = tuple(SLOTS)

    def sb(self, tier: str) -> float:
        return getattr(self, f"sb_{tier}")

    @classmethod
    def load(cls, path: Path | None = None) -> "Params":
        path = path or ROOT / "results" / "pin_model_params.json"
        p = cls()
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            for f in fields(cls):
                if f.name in data and f.name != "enabled":
                    setattr(p, f.name, float(data[f.name]))
        return p

    def save(self, path: Path) -> None:
        d = asdict(self)
        d.pop("enabled")
        Path(path).write_text(json.dumps(d, indent=2))

    TUNABLE = ("sb_street", "sb_locality", "a_cm", "a_br", "a_gali", "a_lmcm", "a_lmbr", "a_lm", "a_locgrp",
               "a_loccent", "a_poi", "a_grid", "f_grid", "c_n", "b_scatter", "gate_k")


# (name, columns, kernel bandwidth per column). A huge bandwidth = one global linear fit per locality
# (right for numbered cross/main grids); a small block bandwidth fits each block separately (T3 blocks).
GRID_FS = [("cm", ["cross", "main"], (1e3, 1e3)), ("br", ["blocknum", "road"], (0.45, 2.5)),
           ("gw", ["gali", "ward"], (1e3, 1e3)), ("g", ["gali"], (1e3,))]


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    c = np.cumsum(weights[o])
    return float(values[o][min(np.searchsorted(c, c[-1] / 2.0), len(values) - 1)])


def _nz(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and v == v and v != ""


# --------------------------------------------------------------------------
# fusion (vectorised, shared by tuning and prediction)
# --------------------------------------------------------------------------
@dataclass
class Candidates:
    mu: np.ndarray       # (N, S, 2)
    s: np.ndarray        # (N, S) scatter of the clue's own members
    n: np.ndarray        # (N, S) number of members
    valid: np.ndarray    # (N, S) bool
    tier: np.ndarray     # (N,) int index into TIERS
    gate_scale: np.ndarray | None = None  # (N, S) >1 makes a clue harder to gate away


def clue_sigma(c: Candidates, p: Params) -> np.ndarray:
    N, S = c.valid.shape
    sig = np.full((N, S), np.inf)
    sb = np.array([p.sb(t) for t in TIERS])
    sig[:, SLOT_IDX["baseline"]] = sb[c.tier]
    for slot in NEIGHBOUR_SLOTS + ["poi"]:
        j = SLOT_IDX[slot]
        a = getattr(p, f"a_{slot}")
        n = np.maximum(c.n[:, j], 1.0)
        sig[:, j] = np.sqrt((a * (1.0 + p.c_n / n)) ** 2 + (p.b_scatter * c.s[:, j]) ** 2)
    j = SLOT_IDX["grid"]
    sig[:, j] = np.sqrt((p.a_grid * c.s[:, j]) ** 2 + p.f_grid ** 2)
    j = SLOT_IDX["loccent"]
    sig[:, j] = p.a_loccent
    j = SLOT_IDX["visit"]
    sig[:, j] = np.sqrt(p.a_visit ** 2 + (p.b_visit * c.s[:, j]) ** 2)
    return sig


def fuse(c: Candidates, p: Params, return_detail: bool = False):
    """Robust precision-weighted fusion. Returns (mu (N,2), sigma (N,), weights (N,S))."""
    N, S = c.valid.shape
    sig = clue_sigma(c, p)
    on = np.zeros(S, bool)
    for s in p.enabled:
        on[SLOT_IDX[s]] = True
    valid = c.valid & on[None, :] & np.isfinite(sig)
    # the old pin is the fallback: always available
    w = np.where(valid, 1.0 / np.maximum(sig, 1e-6) ** 2, 0.0)
    mu = c.mu
    valid0 = valid.copy()
    gs = c.gate_scale if c.gate_scale is not None else np.ones((N, S))
    for _ in range(3):  # drop at most three inconsistent clues
        W = w.sum(1, keepdims=True)
        F = (w[:, :, None] * mu).sum(1) / np.maximum(W, 1e-12)
        Wm = W - w
        with np.errstate(divide="ignore", invalid="ignore"):
            Fm = (W * F)[:, None, :] - w[:, :, None] * mu
            Fm = Fm / np.where(Wm > 0, Wm, np.nan)[:, :, None]
            sm2 = np.where(Wm > 0, 1.0 / Wm, np.nan)
            r = np.linalg.norm(mu - Fm, axis=2) / np.sqrt(np.where(valid, sig ** 2, np.nan) + sm2)
        r = np.where(valid, r / gs, -np.inf)
        r = np.nan_to_num(r, nan=-np.inf, neginf=-np.inf)
        worst = r.argmax(1)
        rmax = r[np.arange(N), worst]
        # never gate a clue away when only one clue remains
        drop = (rmax > p.gate_k) & (valid.sum(1) > 1)
        if not drop.any():
            break
        valid[np.arange(N)[drop], worst[drop]] = False
        w = np.where(valid, w, 0.0)
    W = w.sum(1)
    F = (w[:, :, None] * mu).sum(1) / np.maximum(W, 1e-12)[:, None]
    m = valid.sum(1)
    chi2 = (w * np.linalg.norm(mu - F[:, None, :], axis=2) ** 2).sum(1)
    var = 1.0 / np.maximum(W, 1e-12) + chi2 / (np.maximum(W, 1e-12) * np.maximum(m - 1, 1))
    spread = np.sqrt(var) * p.inflate
    if return_detail:
        # Robust log-evidence of ALL clues (including gated ones) given the fused point: a clue that
        # had to be thrown away pays an outlier penalty, so a hypothesis cannot win by discarding
        # the clues that contradict it.
        eps, area = 0.05, 1e8
        q = np.linalg.norm(mu - F[:, None, :], axis=2) ** 2 / np.maximum(sig, 1e-6) ** 2
        dens = (1 - eps) * np.exp(-0.5 * q) / (2 * np.pi * np.maximum(sig, 1e-6) ** 2) + eps / area
        logev = np.where(valid0, np.log(dens), 0.0).sum(1)
        return F, spread, w, logev
    return F, spread, w


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class PinModel:
    DEFAULT_SB = {"rooftop": 30.0, "street": 120.0, "locality": 380.0, "pincode": 1300.0}
    POI_SPREAD = 150.0

    def __init__(self, params: Params | None = None):
        self.p = params or Params.load()

    # ---------------- fitting -------------------------------------------------
    def fit(self, features: pd.DataFrame, baseline: pd.DataFrame, evidence: pd.DataFrame,
            localities: pd.DataFrame, landmarks: pd.DataFrame, truth: pd.DataFrame | None = None):
        """features: address_features.csv. truth: surveyed rows allowed as *neighbour knowledge*
        (callers must pass training-fold rows only)."""
        feats = features.copy()
        for c in ["cross", "main", "gali", "ward", "road"]:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")
        feats["blocknum"] = feats["block"].map(lambda z: float(ord(str(z).lower()) - 96) if isinstance(z, str) and len(z) == 1 else np.nan)
        self.feat = feats.set_index("address_id")
        self.base = baseline.set_index("address_id")
        self.loc = localities.set_index("locality_id")
        self.pin_locs = localities.groupby(localities.pincode.astype(str)).locality_id.apply(list).to_dict()
        self.town_locs = localities.groupby("town_id").locality_id.apply(list).to_dict()
        self.poi = {k: g[["x", "y"]].to_numpy(float) for k, g in landmarks.groupby(["town_id", "landmark_type"])}
        self.addr_ids = list(self.feat.index)
        wc = self.feat.dropna(subset=["loc_id", "ward"])
        self.ward_cnt = wc.groupby(["loc_id", "ward"]).size().to_dict()
        self._own = {}
        self._evidence_cache = evidence.copy()
        self._set_evidence(self._evidence_cache)
        self._set_truth(truth)
        self._rebuild_known()
        return self

    def _set_evidence(self, evidence: pd.DataFrame):
        E = evidence[evidence.eligible_for_pin]
        self._own = {}
        for a, q in E.groupby("address_id"):
            w = q.weight.to_numpy(float)
            x = weighted_median(q.ev_x.to_numpy(float), w)
            y = weighted_median(q.ev_y.to_numpy(float), w)
            d = np.hypot(q.ev_x.to_numpy(float) - x, q.ev_y.to_numpy(float) - y)
            spread = float(np.sqrt((w * d ** 2).sum() / max(w.sum(), 1e-12))) if len(q) > 1 else 0.0
            self._own[a] = (x, y, len(q), spread, float(w.sum()))

    def _set_truth(self, truth: pd.DataFrame | None):
        self._truth = {}
        if truth is not None and len(truth):
            for r in truth.itertuples():
                self._truth[r.address_id] = (float(r.surveyed_x), float(r.surveyed_y))

    def update(self, new_evidence: pd.DataFrame):
        """Online loop: add visit evidence; every address sharing a key benefits immediately."""
        merged = pd.concat([self._evidence_cache, new_evidence], ignore_index=True)
        self._evidence_cache = merged.drop_duplicates("visit_id", keep="last")
        self._set_evidence(self._evidence_cache)
        self._rebuild_known()

    def _rebuild_known(self):
        rows = {}
        for a, (x, y, n, spread, W) in self._own.items():
            if a in self.feat.index:
                sig = float(np.sqrt(15.0 ** 2 + (0.6 * spread) ** 2))
                rows[a] = (x, y, sig)
        for a, (x, y) in self._truth.items():
            rows[a] = (x, y, 5.0)
        self.known_ids = list(rows)
        self.known_row = {a: i for i, a in enumerate(self.known_ids)}
        arr = np.array([rows[a] for a in self.known_ids], float) if rows else np.zeros((0, 3))
        self.known_xy = arr[:, :2]
        self.known_w = 1.0 / np.maximum(arr[:, 2], 3.0) ** 2 if len(arr) else np.zeros(0)
        self.index: dict[tuple, list[int]] = {}
        for a in self.known_ids:
            f = self.feat.loc[a]
            if not _nz(f.loc_id):
                continue
            for _, key in self._keys(f, f.loc_id):
                self.index.setdefault(key, []).append(self.known_row[a])
        self._lp_cache = {}
        # per (locality, feature-set) design matrices for the street-number grid regression
        tmp: dict[tuple, list] = {}
        for a in self.known_ids:
            f = self.feat.loc[a]
            if not _nz(f.loc_id):
                continue
            for fsname, cols, _h in GRID_FS:
                if all(_nz(f.get(c)) for c in cols):
                    tmp.setdefault((f.loc_id, fsname), []).append((self.known_row[a], [float(f[c]) for c in cols]))
        self.grid = {k: (np.array([r for r, _ in v]), np.array([x for _, x in v])) for k, v in tmp.items()}

    # ---------------- keys ----------------------------------------------------
    @staticmethod
    def _keys(f, l):
        ks = []
        cross, main, gali, ward = f.get("cross"), f.get("main"), f.get("gali"), f.get("ward")
        blk, rd, lm = f.get("block"), f.get("road"), f.get("lm_type")
        if _nz(cross) and _nz(main):
            ks.append(("cm", ("cm", l, str(cross), str(main))))
        if _nz(blk) and _nz(rd):
            ks.append(("br", ("br", l, str(blk), str(rd))))
        if _nz(gali):
            ks.append(("gali", ("g", l, str(gali), str(ward) if _nz(ward) else "")))
        if _nz(lm):
            if _nz(cross) and _nz(main):
                ks.append(("lmcm", ("lmcm", l, str(lm), str(cross), str(main))))
            if _nz(blk) and _nz(rd):
                ks.append(("lmbr", ("lmbr", l, str(lm), str(blk), str(rd))))
            ks.append(("lm", ("lm", l, str(lm))))
        ks.append(("locgrp", ("loc", l)))
        return ks

    def _group(self, key, exclude_row):
        rows = self.index.get(key)
        if not rows:
            return None
        if exclude_row is not None:
            rows = [r for r in rows if r != exclude_row]
            if not rows:
                return None
        P = self.known_xy[rows]
        w = self.known_w[rows]
        mu = np.array([weighted_median(P[:, 0], w), weighted_median(P[:, 1], w)])
        s = float(np.sqrt(np.mean(((P - mu) ** 2).sum(1)))) if len(rows) > 1 else 0.0
        return mu, len(rows), s

    def _locgrp(self, l, exclude_row):
        # locality group is big: cache the unexcluded median, recompute only when excluding a member
        key = ("loc", l)
        if exclude_row is None or exclude_row not in (self.index.get(key) or []):
            if key not in self._lp_cache:
                self._lp_cache[key] = self._group(key, None)
            return self._lp_cache[key]
        return self._group(key, exclude_row)

    def _grid_clue(self, l, f, ex):
        """Robust per-locality linear regression of position on street numbers.

        Layouts are numbered grids (Nth cross / Mth main), so position is a smooth function of the
        numbers: this predicts addresses on streets nobody has visited yet."""
        best = None
        for fsname, cols, h in GRID_FS:
            if any(not _nz(f.get(c)) for c in cols):
                continue
            G = self.grid.get((l, fsname))
            if G is None:
                continue
            rows, X = G
            if ex is not None:
                keep = rows != ex
                rows, X = rows[keep], X[keep]
            n, k = len(rows), len(cols) + 1
            if n < k + 3:
                continue
            x0f = np.array([float(f[c]) for c in cols])
            kw = np.exp(-0.5 * (((X - x0f) / np.array(h)) ** 2).sum(1))
            if (kw > 0.1).sum() < k + 2:
                continue
            A = np.c_[np.ones(n), X - x0f]  # local-linear: the intercept is the prediction at the target
            Y = self.known_xy[rows]
            wt = np.ones(n)
            ridge = np.diag([0.0] + [1e-2] * (k - 1))
            for _ in range(4):
                ww = kw * wt
                AtW = A.T * ww
                coef = np.linalg.solve(AtW @ A + ridge, AtW @ Y)
                r = np.linalg.norm(Y - A @ coef, axis=1)
                scale = max(1.4826 * np.median(r[kw > 0.1]), 15.0)
                wt = np.minimum(1.0, 1.5 * scale / np.maximum(r, 1e-9))
            ww = kw * wt
            AtW = A.T * ww
            lev = float(np.linalg.solve(AtW @ A + ridge, np.eye(k)[0])[0])
            res2 = float((ww * r ** 2).sum() / max(ww.sum() - k, 1.0))
            spread = float(np.sqrt(res2 * (1.0 + lev)))
            if best is None or spread < best[2]:
                best = (coef[0], 1, spread)
        return best

    # ---------------- candidates ---------------------------------------------
    def _poi_clue(self, town, lm, pin, tier):
        P = self.poi.get((town, lm))
        if P is None or len(P) == 0:
            return None
        d2 = ((P - pin) ** 2).sum(1)
        tier_sb = self.DEFAULT_SB[tier]
        w = np.exp(-0.5 * d2 / (tier_sb ** 2 + self.POI_SPREAD ** 2))
        if w.sum() < 1e-4:
            return None
        w = w / w.sum()
        mu = (w[:, None] * P).sum(0)
        s = float(np.sqrt((w * ((P - mu) ** 2).sum(1)).sum()))
        return mu, 1, s

    def _candidates_for(self, a, l, hide_own, exclude_self=True):
        """Return arrays for one (address, locality hypothesis)."""
        S = len(SLOTS)
        mu = np.zeros((S, 2))
        s = np.zeros(S)
        n = np.ones(S)
        valid = np.zeros(S, bool)
        f = self.feat.loc[a]
        b = self.base.loc[a]
        pin = np.array([b.geocoder_x, b.geocoder_y], float)
        i = SLOT_IDX["baseline"]
        mu[i], valid[i] = pin, True
        if not hide_own and a in self._own:
            x, y, cnt, spread, W = self._own[a]
            i = SLOT_IDX["visit"]
            mu[i], s[i], n[i], valid[i] = (x, y), spread, cnt, True
        ex = self.known_row.get(a) if exclude_self else None
        if _nz(l):
            for slot, key in self._keys(f, l):
                g = self._locgrp(l, ex) if slot == "locgrp" else self._group(key, ex)
                if g is None:
                    continue
                i = SLOT_IDX[slot]
                mu[i], n[i], s[i], valid[i] = g[0], g[1], g[2], True
            g = self._grid_clue(l, f, ex)
            if g is not None:
                i = SLOT_IDX["grid"]
                mu[i], n[i], s[i], valid[i] = g[0], g[1], g[2], True
            c = self.loc.loc[l]
            i = SLOT_IDX["loccent"]
            mu[i], valid[i] = (c.centroid_x, c.centroid_y), True
        if _nz(f.get("lm_type")):
            g = self._poi_clue(f.town_id, f.lm_type, pin, b.precision)
            if g is not None:
                i = SLOT_IDX["poi"]
                mu[i], n[i], s[i], valid[i] = g[0], g[1], g[2], True
        gs = np.ones(S)
        if valid[SLOT_IDX["visit"]]:
            _, _, cnt, spread, _ = self._own[a]
            # a tight cluster of >=2 agreeing visit rows is ground truth about the place: never gate it away;
            # a single weak row can still be overruled by overwhelming contrary evidence.
            gs[SLOT_IDX["visit"]] = 1e3 if (cnt >= 2 and spread < 60) else VISIT_GATE_WEAK
        return mu, s, n, valid, TIERS.index(b.precision), gs

    def _ward_prior(self, hyps, ward):
        if not _nz(ward):
            return np.ones(len(hyps)) / len(hyps)
        c = np.array([self.ward_cnt.get((h, float(ward)), 0) + 0.5 for h in hyps])
        return c / c.sum()

    def locality_hypotheses(self, a):
        f = self.feat.loc[a]
        if _nz(f.loc_id):
            return [f.loc_id]
        pin = f.get("pincode_text")
        if _nz(pin) and str(int(float(pin))) in self.pin_locs:
            return list(self.pin_locs[str(int(float(pin)))])
        return list(self.town_locs.get(f.town_id, []))

    def build_candidates(self, ids, hide_own=False, exclude_self=True):
        """Single-hypothesis candidates for addresses with a parsed locality (used by tuning)."""
        mus, ss, ns, vs, ts, gss = [], [], [], [], [], []
        for a in ids:
            l = self.feat.at[a, "loc_id"]
            mu, s, n, v, t, gs = self._candidates_for(a, l if _nz(l) else None, hide_own, exclude_self)
            mus.append(mu), ss.append(s), ns.append(n), vs.append(v), ts.append(t), gss.append(gs)
        return Candidates(np.array(mus), np.array(ss), np.array(ns), np.array(vs), np.array(ts), np.array(gss))

    # ---------------- prediction ---------------------------------------------
    def predict(self, ids=None, hide_own=False, exclude_self=True, params: Params | None = None) -> pd.DataFrame:
        p = params or self.p
        ids = list(self.addr_ids if ids is None else ids)
        out = []
        for a in ids:
            hyps = self.locality_hypotheses(a)
            cands = [self._candidates_for(a, l, hide_own, exclude_self) for l in (hyps if hyps else [None])]
            C = Candidates(np.array([c[0] for c in cands]), np.array([c[1] for c in cands]),
                           np.array([c[2] for c in cands]), np.array([c[3] for c in cands]),
                           np.array([c[4] for c in cands]), np.array([c[5] for c in cands]))
            has_visit = (not hide_own) and a in self._own
            if len(hyps) > 1 and not has_visit:
                # The text names no locality and nothing on the ground pins it down: the pincode leaves
                # several candidate localities. Evidence from a wrong locality would only add false
                # confidence, so use locality-level clues, weight hypotheses by the ward prior (the
                # only locality-specific text signal) and report the wide spread honestly.
                q = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__ if k != "enabled"})
                q.enabled = tuple(x for x in p.enabled if x in ("baseline", "loccent", "locgrp"))
                F, spread, w = fuse(C, q)
                pw = self._ward_prior(hyps, self.feat.at[a, "ward"])
            else:
                F, spread, w, logev = fuse(C, p, return_detail=True)
                pw = np.exp((logev - logev.max()) / p.hyp_temp) if len(hyps) > 1 else np.ones(1)
                pw = pw / pw.sum()
            if len(hyps) > 1:
                mu = (pw[:, None] * F).sum(0)
                between = float((pw * ((F - mu) ** 2).sum(1)).sum())
                spr = float(np.sqrt((pw * spread ** 2).sum() + between))
                top = int(pw.argmax())
                loc_used = hyps[top]
                wsel = (pw[:, None] * w).sum(0)
            else:
                mu, spr, loc_used, wsel = F[0], float(spread[0]), (hyps[0] if hyps else None), w[0]
            share = wsel / max(wsel.sum(), 1e-12)
            j = int(np.argmax(share))
            method = METHOD_OF[SLOTS[j]] if share[j] >= 0.45 else "blend"
            used = ",".join(SLOTS[k] for k in np.argsort(-share) if share[k] >= 0.05)
            out.append({"address_id": a, "pin_x": float(mu[0]), "pin_y": float(mu[1]), "raw_spread": float(spr),
                        "method_used": method, "clues_used": used, "locality_used": loc_used,
                        "n_locality_hyp": len(hyps)})
        return pd.DataFrame(out)


# --------------------------------------------------------------------------
# data loading shared by scripts
# --------------------------------------------------------------------------
def load_all(data_dir: Path = DATA, feature_path: Path | None = None):
    feature_path = feature_path or ROOT / "results" / "address_features.csv"
    features = pd.read_csv(feature_path)
    baseline = pd.read_csv(data_dir / "baseline_geocodes.csv")
    localities = pd.read_csv(data_dir / "localities.csv")
    landmarks = pd.read_csv(data_dir / "landmarks_poi.csv")
    evidence = pd.read_csv(ROOT / "evidence.csv")
    surveyed = pd.read_csv(data_dir / "surveyed_addresses.csv")
    folds = pd.read_csv(ROOT / "folds.csv")
    return features, baseline, localities, landmarks, evidence, surveyed, folds
