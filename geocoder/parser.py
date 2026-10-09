"""Rule-based multilingual address parser.

Turns free-text Indian addresses (English / Hinglish / Kanglish, Latin, Devanagari
and Kannada script, abbreviated and misspelt) into structured fields that the pin
model uses as *keys* for finding neighbouring addresses:

    locality_id, pincode, cross, main, gali, block, road, ward,
    building, lm_type, lm_rel

Design notes
- Deliberately rule-based: the text is templated enough that a dictionary plus
  fuzzy matching is accurate, auditable (every field traces to a phrase) and works
  offline. No coordinates are used anywhere in this file.
- Locality is matched fuzzily *inside the address's own town*, and narrowed by the
  pincode in the text when one is present and valid.
- Words that belong to the matched locality name are removed before landmark
  parsing, so "Green Park Layout" is not read as a park landmark.

Run:  python -m geocoder.parser   ->  outputs/results/address_features.csv
"""

from __future__ import annotations

from .paths import DATA, RES

import difflib
import re

import pandas as pd


# --------------------------------------------------------------------------
# Dictionaries
# --------------------------------------------------------------------------
ABBREV = {
    # nagar
    "ngr": "nagar", "nagr": "nagar", "naagr": "nagar", "ngar": "nagar", "naagar": "nagar",
    "nagra": "nagar", "ngaar": "nagar", "nager": "nagar", "nagarr": "nagar",
    # layout / colony / enclave / mohalla
    "layt": "layout", "lyt": "layout", "lout": "layout", "layot": "layout", "lay": "layout",
    "col": "colony", "colny": "colony", "collony": "colony", "colon": "colony", "cly": "colony",
    "enclv": "enclave", "encl": "enclave", "mohala": "mohalla", "mhla": "mohalla",
    "gardens": "gardens", "gdns": "gardens",
}
GENERIC_LOC = {"nagar", "colony", "layout", "enclave", "basti", "mohalla", "puri", "gardens"}

LANDMARKS: dict[str, str] = {
    "hanuman_temple": r"han+u?m[aā]+n|hanumn|hnuman|anjan[a-z]*\s*(?:gudi|swamy|temple|mandir)|हनुमान|ಹನುಮ|ಆಂಜನೇಯ",
    "ganesha_temple": r"gan[ae]+sh|gaanesh|ganapath|ganpat|गणेश|ಗಣಪತಿ|ಗಣೇಶ|\bganesh",
    "masjid": r"masjid|masidi|mazjid|mosque|मस्जिद|ಮಸೀದಿ|ಮಸ್ಜಿದ",
    "church": r"ch[uo]r?ch|churc\b|girja|गिरजा|चर्च|ಚರ್ಚ್",
    "govt_school": r"school|skool|schol|shaale|shale|स्कूल|विद्यालय|ಶಾಲೆ",
    "ration_shop": r"ration|\bpds\b|nyaya\s*bele|nyayabele|fair\s*price|राशन|ನ್ಯಾಯಬೆಲೆ|ಪಡಿತರ",
    "bus_stop": r"bus\s*(?:stop|stand|stnd|adda|nild\w*)|\bbus\b|बस|ಬಸ್|\badda\b",
    "milk_dairy": r"milk|dairy|doodh|haalina|halina|दूध|डेयरी|ಹಾಲಿನ|ಡೈರಿ",
    "medical_store": r"medical|pharmacy|chemist|dawai|dawa\b|मेडिकल|दवाई|ಮೆಡಿಕಲ್|ಔಷಧ",
    "petrol_bunk": r"petrol|\bbunk\b|\bpump\b|पेट्रोल|पंप|ಪೆಟ್ರೋಲ್|ಬಂಕ್",
    "post_office": r"post\s*off|anche|dak\s*ghar|dakghar|डाकघर|डाक|ಅಂಚೆ",
    "water_tank": r"water\s*tank|overhead|neerina|paani|pani\s*ki|tanki|टंकी|ಟ್ಯಾಂಕ್|ನೀರಿನ",
    "community_hall": r"communit|mantapa|mandapa|samudaya|bhavana|barat|कम्युनिटी|सामुदायिक|बारात|ಕಲ್ಯಾಣ|ಸಮುದಾಯ|\bhall\b",
    "park": r"\bpark\b|bagicha|udyan|garden|children|बगीचा|पार्क|ಪಾರ್ಕ್|ಉದ್ಯಾನ",
}
_LM_RE = {k: re.compile(v, re.I) for k, v in LANDMARKS.items()}
# a landmark word that is only meaningful together with another (avoid "tank" alone etc.)

# canonical words used for a typo-tolerant fallback ("Chruch", "Psot Office", "Medcial", "Scheol")
FUZZY_VOCAB = {
    "church": "church", "masjid": "masjid", "mosque": "masjid", "masjad": "masjid", "school": "govt_school",
    "ration": "ration_shop", "dairy": "milk_dairy", "milk": "milk_dairy", "medical": "medical_store",
    "pharmacy": "medical_store", "petrol": "petrol_bunk", "ganesh": "ganesha_temple", "hanuman": "hanuman_temple",
    "community": "community_hall", "tank": "water_tank", "overhead": "water_tank", "post": "post_office",
}


def _fuzzy_landmark(tokens: list[str]):
    best = (0.0, None)
    for t in tokens:
        if len(t) < 4 or not t.isascii() or not t.isalpha():
            continue
        for w, ty in FUZZY_VOCAB.items():
            sc = difflib.SequenceMatcher(None, t, w).ratio()
            if sc > best[0]:
                best = (sc, ty)
    return best[1] if best[0] >= 0.78 else None


RELATIONS = {
    "opposite": r"opp\b|opposite|samne|saamne|sammne|eduru|edhuru|सामने|ಎದುರು",
    "behind": r"behind|bhd\b|peeche|piche|pichhe|pichche|hinde|hindhe|पीछे|पिछे|ಹಿಂದೆ",
    "beside": r"beside|bagal|bagl|pakka|pakk\b|baju|adj\b|adjacent|next\s*to|बगल|ಪಕ್ಕ",
    "near": r"near|\bnr\b|naer|close\s*to|ke\s*pa+s|hattira|hatt?ir|paas|pass\b|के\s*पास|ಹತ್ತಿರ|\bnear",
}
_REL_RE = {k: re.compile(v, re.I) for k, v in RELATIONS.items()}

BUILDING_SUFFIX = r"(?:complex|apts|apartments?|residency|nilaya|nivas|kunj|towers?|villa|heights|plaza|sadan|bhavan)"
_BUILDING_RE = re.compile(r"((?:[a-z]{3,}\s+){1,2})" + BUILDING_SUFFIX + r"\b", re.I)

_PIN_RE = re.compile(r"\b(\d{5,6})\b")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"[#,()\-/.]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _canon_tokens(tokens: list[str]) -> list[str]:
    return [ABBREV.get(t, t) for t in tokens]


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _match_locality(tokens: list[str], cands: pd.DataFrame):
    """Return (locality_id, score, margin, matched_token_positions)."""
    scored = []
    for _, loc in cands.iterrows():
        name = _canon_tokens(_norm(loc.locality_name).split())
        wsum, acc, used = 0.0, 0.0, []
        for nt in name:
            w = 0.4 if nt in GENERIC_LOC else 1.0
            best, pos = 0.0, None
            for i, t in enumerate(tokens):
                if len(t) < 2 and not nt.isdigit():
                    continue
                s = _tok_sim(nt, t)
                if nt.isdigit():
                    s = 1.0 if t == nt else 0.0
                if s > best:
                    best, pos = s, i
            # a distinctive token must match well, or the whole locality is rejected
            if w == 1.0 and best < 0.78:
                best = 0.0
            wsum += w
            acc += w * best
            if best >= 0.78 and pos is not None:
                used.append(pos)
        scored.append((acc / wsum, loc.locality_id, used))
    scored.sort(key=lambda x: -x[0])
    top = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top[0] < 0.72:
        return None, top[0], top[0] - second, []
    return top[1], top[0], top[0] - second, top[2]


def _first(pattern: str, text: str, flags=re.I):
    m = re.search(pattern, text, flags)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.lower()
    return None


# --------------------------------------------------------------------------
# main parse
# --------------------------------------------------------------------------
def parse_addresses(addresses: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    locs_by_town = {t: g.reset_index(drop=True) for t, g in localities.groupby("town_id")}
    valid_pins = set(localities.pincode.astype(str))
    pin_towns = localities.assign(pc=localities.pincode.astype(str)).groupby("pc").town_id.first().to_dict()
    rows = []
    for r in addresses.itertuples():
        raw = str(r.address_text)
        text = _norm(raw)
        # ---- pincode (tolerate one-digit typos: 97204 -> 970204 not recoverable, drop)
        pin = None
        for m in _PIN_RE.finditer(text):
            if m.group(1) in valid_pins:
                pin = m.group(1)
                break
        town = r.town_id
        cands = locs_by_town.get(town)
        if cands is None:
            rows.append({"address_id": r.address_id})
            continue
        if pin and pin_towns.get(pin) == town:
            sub = cands[cands.pincode.astype(str) == pin]
            sub_all = cands
        else:
            pin = pin if (pin and pin_towns.get(pin) == town) else None
            sub = cands
            sub_all = cands
        tokens = _canon_tokens(text.split())
        lid, sc, mg, used = _match_locality(tokens, sub)
        loc_src = "text+pincode" if (pin and lid) else ("text" if lid else None)
        if lid is None and pin is not None:
            # pincode narrowed to one locality?
            if len(sub) == 1:
                lid, loc_src = sub.locality_id.iloc[0], "pincode_unique"
        if lid is None and pin is not None:
            # maybe the locality is in the town but outside the pincode (typo'd pincode)
            lid2, sc2, mg2, used2 = _match_locality(tokens, sub_all)
            if lid2 is not None and sc2 >= 0.85:
                lid, sc, mg, used, loc_src = lid2, sc2, mg2, used2, "text"
        used_set = set(used)
        lm_text = " ".join(t for i, t in enumerate(tokens) if i not in used_set)
        # ---- landmark
        found = []
        for k, rx in _LM_RE.items():
            m = rx.search(lm_text)
            if m:
                found.append((m.start(), k))
        found.sort()
        lm_type = found[0][1] if found else None
        if lm_type is None and any(rx.search(lm_text) for rx in _REL_RE.values()):
            lm_type = _fuzzy_landmark(lm_text.split())
            if lm_type:
                found = [(0, lm_type)]
        lm_types = "|".join(k for _, k in found)
        # relation word nearest to the landmark word
        lm_rel = None
        if lm_type:
            for k, rx in _REL_RE.items():
                if rx.search(lm_text):
                    lm_rel = k
                    break
        # ---- street keys (use the un-tokenised text so "5th cross" stays intact)
        cross = _first(r"(\d+)\s*(?:st|nd|rd|th)?\s*(?:cross|crs|crss|cros|cr\b|x\b)", text)
        main = _first(r"(\d+)\s*(?:st|nd|rd|th)?\s*(?:main|mainn|mn\b)", text)
        gali = _first(r"(?:gali|galli|glai|gai|गली)\s*(?:no|नं|number)?\s*(\d+)", text)
        block = _first(r"\b([a-f])\s*(?:blk|block)\b", text) or _first(r"(?:blk|block)\s*([a-f])\b", text)
        road = (_first(r"(?:\brd|\broad)\s*(\d+)", text)
                or _first(r"(\d+)\s*(?:st|nd|rd|th)?\s*(?:rd|road)\b", text))
        if road is not None and cross is not None and main is not None:
            road = None  # "7th main rd" style: not a numbered road
        if cross is not None and int(cross) > 20:  # "122th Cross": a typo, not a street number
            cross = None
        ward = _first(r"ward\s*(\d+)", text)
        # ---- building
        bm = _BUILDING_RE.search(text)
        building = None
        if bm:
            nm = re.sub(r"\s+", " ", bm.group(1).strip())
            suf = re.search(BUILDING_SUFFIX, bm.group(0), re.I).group(0).lower()
            suf = {"apts": "apartments", "apartment": "apartments"}.get(suf, suf)
            # drop words that belong to the locality name or the road vocabulary
            nm_tokens = [t for t in nm.split() if t not in {"no", "house", "blk", "block", "rd", "road", "main", "cross"}
                         and not t.isdigit()]
            if nm_tokens:
                building = " ".join(_canon_tokens(nm_tokens)) + " " + suf
        script = "kn" if re.search(r"[ಀ-೿]", raw) else ("hi" if re.search(r"[ऀ-ॿ]", raw) else "latin")
        rows.append({
            "address_id": r.address_id, "town_id": town, "pincode_text": pin,
            "loc_id": lid, "loc_score": round(sc, 3), "loc_margin": round(mg, 3), "loc_src": loc_src,
            "cross": cross, "main": main, "gali": gali, "block": block, "road": road,
            "ward": ward, "building": building, "lm_type": lm_type, "lm_types": lm_types,
            "lm_rel": lm_rel, "script": script,
        })
    out = pd.DataFrame(rows)
    return out


def _check_pincode_to_locality(parsed: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    """Where the locality is missing but the pincode maps to several localities,
    keep the pincode as a coarser key (done in the model via `pincode_text`)."""
    return parsed


def main() -> None:
    addr = pd.read_csv(DATA / "addresses.csv")
    addr = addr[addr.town_id != "OUT"]
    loc = pd.read_csv(DATA / "localities.csv")
    parsed = parse_addresses(addr, loc)
    out = RES / "address_features.csv"
    out.parent.mkdir(exist_ok=True)
    parsed.to_csv(out, index=False)
    print(f"parsed {len(parsed):,} addresses -> {out}")
    cov = {c: round(parsed[c].notna().mean(), 3)
           for c in ["pincode_text", "loc_id", "cross", "main", "gali", "block", "road", "ward", "building", "lm_type"]}
    print("field coverage:", cov)


if __name__ == "__main__":
    main()
