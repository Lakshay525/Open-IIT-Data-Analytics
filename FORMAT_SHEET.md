# Shared format sheet (column contracts between tasks)

Coordinates everywhere: local metres, `x` east, `y` north (assumed; no CRS in the source). One row per key; all files UTF-8 CSV.

| File | Key | Columns | Produced by -> used by |
|---|---|---|---|
| `evidence.csv` | `visit_id` | `visit_id, address_id, ev_x, ev_y, weight, trail_type` (+ `uncertainty_m, evidence_role, eligible_for_pin, outcome, agent_id, town_id, ...`) | Task 2 -> Task 3 |
| `results/address_features.csv` | `address_id` | `town_id, pincode_text, loc_id, loc_score, cross, main, gali, block, road, ward, building, lm_type, lm_rel, script` | parser -> Task 3 |
| `pins.csv` | `address_id` | `address_id, pin_x, pin_y, raw_spread, method_used` (+ `clues_used, locality_used, n_locality_hyp, pin_source, has_own_visit`) | Task 3 -> Task 4. `raw_spread` = per-axis sigma (m). The 100 surveyed rows are out-of-fold. |
| `predictions.csv` | `address_id` | `address_id, pin_x, pin_y, radius_50, radius_90, directions, confidence_tier` (+ `suggested_action, location_confidence, radius_80, reason_codes, method_used, raw_spread, town_id, precision_old_geocoder, old_pin_x, old_pin_y, directions_local, has_own_visit, pin_source`) | Task 4 -> field app, planner, records, dashboards |
| `ps2_location_confidence.csv` | `address_id` | `location_confidence, p_within_250m, radius_90_m, confidence_tier, hard_to_find_flag, reason_codes, ps2_guidance` | Task 4 -> PS2 |
| `offline_pack/<town>.json` | town | `addresses[{id,x,y,r50,r90,tier,dir}], landmarks[{poi_id,name,x,y}]` | Task 4 -> field app |

Definitions
* `radius_50` / `radius_90`: the true location lies inside with ~50% / ~90% probability, calibrated by conformal prediction (`confidence_directions.py`).
* `confidence_tier`: `high` radius_90 <= 60 m (visit now); `medium` <= 300 m (visit with directions); `low` otherwise (verify first).
* `location_confidence`: calibrated probability that the true location is within 100 m of the pin.
* `hard_to_find_flag`: tier `low` and no field visit of its own; tells PS2 "could not find" is not "does not exist".
* `method_used`: dominant clue (`own_visit`, `street_grid`, `street_key`, `landmark_group`, `landmark_poi`, `locality_centroid`, `baseline`, `blend`).
* `reason_codes` (pipe-separated): `CONFIRMED_BY_FIELD_VISIT`, `INFERRED_FROM_STREET_NEIGHBOURS`, `INFERRED_FROM_LANDMARK`, `LOCALITY_LEVEL_ONLY`, `NO_LOCALITY_IN_TEXT`, `OLD_GEOCODER_AT_PINCODE_CENTRE`.
