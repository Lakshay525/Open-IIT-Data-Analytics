"""Checks for the live web app: the API engine and the static data it serves.  Run: python -m tests.test_web

The engine is imported from web/api/_lib exactly as the Vercel functions import it, so this also proves the deployable bundle is complete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
sys.path.insert(0, str(WEB / "api" / "_lib"))
import engine  # noqa: E402  (must come before any other `geocoder` import: it brings its own copy)


def test_bundle_is_complete():
    needed = ["index.html", "styles.css", "app.js", "data/stats.json", "data/demo_cases.json", "data/town_T1.json", "data/town_T2.json", "data/town_T3.json"]
    missing = [n for n in needed if not (WEB / "public" / n).exists()]
    assert not missing, missing
    for n in ("locate.py", "visit.py", "health.py"):
        assert (WEB / "api" / n).exists(), n
    assert (WEB / "requirements.txt").exists() and (WEB / "vercel.json").exists()
    assert engine.health()["ok"]
    print("PASS web bundle complete; engine loads (2,880 addresses, 5,578 visits)")


def test_locate_on_held_out_addresses():
    cases = json.loads((WEB / "public" / "data" / "demo_cases.json").read_text(encoding="utf-8"))["examples"]
    better = 0
    for ex in cases:
        r = engine.locate(ex["text"], ex["town"])
        assert "error" not in r and r["tier"] in ("high", "medium", "low")
        assert r["radius_90"] >= r["radius_50"] > 0 and r["directions"]
        import math
        err = math.hypot(r["pin"]["x"] - ex["truth"][0], r["pin"]["y"] - ex["truth"][1])
        better += err < ex["old_error"] + 1
    assert better >= 4, f"only {better}/6 held-out examples improved on the old pin"
    print(f"PASS locate: {better}/{len(cases)} held-out addresses beat the old geocoder pin")


def test_locate_rejects_bad_input():
    assert "error" in engine.locate("hi", "T1")
    assert "error" in engine.locate("some long enough address text", "T9")
    r = engine.locate("Gali No. 4, Ward 1, Devgarh Nagar", "T2")      # no locality named: must still answer, with a wide radius
    assert "error" not in r and r["radius_90"] > 300 and r["tier"] == "low"
    print("PASS locate handles bad input and locality-free addresses")


def test_visit_moves_neighbours():
    sc = json.loads((WEB / "public" / "data" / "demo_cases.json").read_text(encoding="utf-8"))["scenarios"]
    s = sc[0]
    r = engine.what_if_visit(s["id"], s["visit"][0], s["visit"][1])
    assert "error" not in r and r["neighbours_moved"] >= 3
    assert r["target"]["spread_after"] < r["target"]["spread_before"]
    assert "error" in engine.what_if_visit("AD999999", 0, 0)
    print(f"PASS visit: one confirmed visit re-estimated {r['neighbours_moved']} neighbours; the address's own uncertainty fell {r['target']['spread_before']:.0f} -> {r['target']['spread_after']:.0f} m")


if __name__ == "__main__":
    test_bundle_is_complete()
    test_locate_on_held_out_addresses()
    test_locate_rejects_bad_input()
    test_visit_moves_neighbours()
    print("all web checks passed")
