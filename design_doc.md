# Design document - Address geocoder that learns from field visits (PS3)

Audience: CreditNirvana (CN) product, field-ops and platform teams. Everything here is implemented in this package except where marked **(to build in CN)**.

## 1. What the system does

For every account address it produces: a **pin**, a **calibrated confidence radius** (50% and 90%), a **confidence tier** with a suggested action, **plain-text landmark directions** that work offline, and a **location-confidence score for PS2**. Every confirmed field visit becomes new evidence, so pins for neighbouring addresses improve without retraining.

```
 CN field app --visit logs, GPS trails, outcomes, remarks-->  evidence extraction (Task 2)  --> one trusted point + role + uncertainty per visit
 CN address records --address text, old pin----------------->  address parser               --> locality, street numbers, landmark type
                                                                    |                              |
                                                                    +------------> pin model (Task 3) <--- known locations (visit pins)
                                                                                         |
                                                              calibration + directions (Task 4)
                                                                                         |
   +---------------+----------------+-------------------+-----------------+-------------+
   field app      visit planner    address records     PS2 (RPC model)     dashboards
 (offline pack)  (visit / verify)  (write-back, share)  (location conf.)  (quality, trends)
```

## 2. Components and contracts

| Component | Module | Input | Output | Notes |
|---|---|---|---|---|
| Evidence extraction | `extract_evidence.py` | `field_visits`, `visit_gps_points` | `evidence.csv` (one row per visit) | Role (contact / structure-only / historical / failed search), uncertainty, integrity factors. Raw trails can be deleted after this step. |
| Address parser | `address_parser.py` | `addresses.address_text`, localities | `address_features.csv` | Rule-based, multilingual, typo tolerant; every field traces to a phrase. |
| Pin model | `pin_model.py` | features, old pin, evidence, landmarks | `pins.csv` (pin, `raw_spread`, `method_used`, `clues_used`) | Fusion of clues; `fit`, `predict`, `update`. |
| Calibration + directions | `confidence_directions.py` | `pins.csv`, calibration sets | `predictions.csv`, `ps2_location_confidence.csv`, `offline_pack/*.json` | Conformal radii, tiers, reason codes, directions. |
| Orchestration | `run_pipeline.py` | raw CSVs | everything above + tests + notebooks | One command. |

Column contracts are in `FORMAT_SHEET.md`.

## 3. How CN's systems plug in

| CN system | What it reads | How |
|---|---|---|
| **Field app** | pin, `radius_50/90`, `confidence_tier`, `directions` | Per-territory JSON pack (`offline_pack/<town>.json`, ~200 bytes per address) synced when online, read from device storage offline. Map shows the pin with the 50/90% circles; "directions" is plain text. After a visit the app uploads outcome + GPS trail as it does today. |
| **Visit planner** | `confidence_tier`, `suggested_action`, `radius_90` | `VISIT_NOW` (high) enters routing normally; `VISIT_WITH_DIRECTIONS` (medium) enters routing with a larger stop-time allowance; `VERIFY_FIRST` (low) is routed to a verification task (call borrower/reference for a landmark, or a quick local check) *before* a costly visit is planned. A small exploration share of low-confidence addresses is still visited so the model keeps learning (see section 6). |
| **Address records** | `pin`, `radius_90`, `pin_source`, `reason_codes` | Written back per address and shared across accounts at the same place (`update()` makes one confirmed visit a clue for every address on that street). **(write-back job to build in CN)** |
| **PS2 (right-party-contact)** | `ps2_location_confidence.csv` | `location_confidence` = calibrated P(true location within 100 m); `hard_to_find_flag` = low confidence and no visit. PS2 must read a failed visit on a flagged address as "couldn't find it", not "doesn't exist". |
| **Dashboards** | `predictions.csv`, rolling error log | Quality: median / P90 distance between each previous pin and the next confirmed visit (self-auditing, no survey needed), calibration coverage, share by tier; Operations: "not traceable" rate and productive visits per agent-day by territory. **(dashboards to build in CN)** |

## 4. Run modes

* **Nightly batch** (minutes for millions of rows - see section 7): re-extract evidence for yesterday's visits, `update()` the model, regenerate `predictions.csv` and the offline packs.
* **On visit close (optional)**: `PinModel.update(new_evidence)` then re-predict only the addresses sharing the visit's street key / landmark. Measured here: fitting all 2,880 addresses takes 0.4 s, predicting one address about 1 ms, and an update about 0.4 s (it rebuilds the key index; an incremental index would make it constant-time).
* **Monthly**: refit sigmas (`tune_pin_model.py`) and recalibrate radii (`confidence_directions.py`); compare calibration coverage with the previous month and alert if the 90% radius covers less than ~85% or more than ~97% of newly confirmed visits.

## 5. Why this design (decisions and trade-offs)

* **Evidence fusion, not a black-box model.** The clues are few, heterogeneous and have known reliability; fusion by inverse variance with robust gating is accurate here, trains in seconds, and every pin can be explained as "this visit + these two neighbours + this grid fit". That meets the "explainable and auditable" requirement without a post-hoc explainer.
* **Street-number grid regression.** In numbered layouts (Nth cross / Mth main, block / road) the numbers are a coordinate system. A per-locality robust local-linear fit locates streets nobody has visited - the single biggest accuracy gain after the visits themselves.
* **Conformal calibration on data the model did not see.** The radius for unvisited addresses is set on a 1,231-address hide-and-predict set, then *checked* on the surveyed addresses.
* **Visits that disagree with text are protected.** A tight cluster of agreeing visits is ground truth about where the borrower lives, even if the KYC locality is wrong; text-derived clues are gated instead.
* **Honest abstention.** Pincode-level addresses with no locality in the text cannot be resolved from text; the model stays near the old pin, reports a wide radius, and the planner verifies first.

## 6. Feedback loops and bias

Historical visits reflect the incumbent policy (which addresses were visited, by whom). Mitigations built in: (1) the model is evaluated on addresses *without* visits (hide-and-predict) and on independent survey truth, not on visited addresses alone; (2) the per-agent cap and cross-address photo-reuse penalties stop one agent steering a pin; (3) we recommend an **exploration budget** of about 5% of low-confidence addresses visited regardless of tier so the model keeps receiving evidence where it is least certain; (4) calibration is re-checked monthly by territory.

## 7. Scale and cost

* Complexity: building the key index is linear in known addresses; each prediction is a handful of dictionary lookups, one small weighted regression (tens of points) and a fusion step. The 2,880-address pipeline (excluding notebooks and the optional stress test) runs in about two minutes on a laptop; at a few million addresses, partition by territory (the model is naturally territory-local) and run territories in parallel.
* Storage: one derived evidence row per visit replaces a raw GPS trail of about 29 points, so far less location data needs to be retained.
* No external geocoding or map API is required, so there is no per-call cost and no address text leaves CN's environment.

## 8. Offline behaviour

The pack contains, per address: pin, 50% and 90% radii, tier and directions; plus the town's landmarks. All directions are generated server-side as text, so the app needs no routing or map service. Tier colours and the circles are drawn on whatever base map the app already uses (or none: the demo map shows the pin, circles and landmarks with no tiles). Sync is a small JSON diff per territory.

## 9. Failure modes and fallbacks

| Situation | Behaviour |
|---|---|
| Parser finds no locality | Several candidate localities (by pincode); stays near the old pin; wide radius; tier `low`; `NO_LOCALITY_IN_TEXT` |
| Visit contradicts everything else | A single weak visit can be overruled; a tight cluster of agreeing visits cannot |
| Fake / copied visits | Cross-address photo reuse and repeated-stop clusters are down-weighted to zero influence; per-agent cap 0.35 per address |
| New town / no visits yet | Falls back to landmarks and the old pin (cold-start median 239 m vs 318 m old on the test split of visited addresses), improving as visits arrive |
| Unusual coordinates / CRS mismatch | Pipeline refuses to run if coordinates are non-finite; `AXES` switch for east/north orientation |

## 10. What we need from CN to go live

1. Confirm the coordinate system (CRS) and that local `x` = east, `y` = north.
2. Real landmark names per territory (the supplied list has 240 generic POIs; directions improve with a denser list).
3. A territory-level retention policy for raw GPS trails and photos (see `COMPLIANCE.md`).
4. Review of the Hinglish / Kanglish direction phrases by native speakers.
5. Pilot territories and a randomisation unit (see `pilot_plan.md`).
