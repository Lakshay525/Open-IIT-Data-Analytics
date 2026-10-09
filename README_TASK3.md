# Task 3 - Pin model

Gives every one of the 2,880 addresses a pin, an honest uncertainty (`raw_spread`) and a record of which
clues produced it. Task 4 turns `raw_spread` into calibrated radii and writes directions.

```bash
pip install -r requirements.txt
python run_task3.py            # parse -> CV -> ablation -> hide-and-predict -> pins.csv
python tune_pin_model.py       # optional: refit sigmas on leave-one-out visit pseudo-labels
python test_task3.py           # leakage, online-update, visit-protection and output-contract checks
python create_task3_notebook.py  # rebuild notebooks/pin_model_experiments.ipynb
```

## What the model is

Not a neural net and not one big regression. It is a **precision-weighted evidence-fusion model**: each clue
gives a guess and an honest sigma, clues are combined by inverse variance, and a clue that contradicts the
rest is dropped. Every number is traceable to a clue, which suits the RBI/DPDP "explainable and auditable" rule.

| clue (slot) | what it is | typical sigma |
|---|---|---|
| `visit` | the address's own cleaned visit pin (Task 2 evidence) | 10-30 m |
| `cm`, `br`, `gali` | known pins of other addresses on the same street key (cross+main, block+road, gali) | 55-85 m (gali off) |
| `grid` | **street-number grid regression**: cross/main (T1) and block/road (T3) numbers are a coordinate system, so a robust local-linear fit per locality predicts streets nobody has visited | 30-100 m |
| `lmcm`, `lmbr`, `lm` | same street key / same locality **and** same landmark type | 150-200 m |
| `poi` | landmark POIs of the named type near the old pin | ~100-200 m |
| `locgrp`, `loccent` | known pins in the locality; locality centroid | ~1 km (weak, safety net) |
| `baseline` | old geocoder pin; sigma by precision tier | 25 m ... 1.25 km |

`address_parser.py` supplies the keys: locality (fuzzy, within the town, narrowed by pincode), cross/main/gali/
block/road/ward, landmark type (English/Hindi/Kannada, Latin and native script, typo-tolerant) and relation word.

**Learning loop.** `PinModel.update(new_evidence)` adds visits without retraining; every address that shares a
street key, landmark or grid benefits at once (verified in `test_task3.py`).

**No locality in the text** (about 10% of addresses, almost all pincode tier): the pincode leaves 3-5 candidate
localities. Nothing in the text separates them, so the model stays near the old pin, weights candidates by the
ward prior (T2 wards are locality-specific) and reports a wide `raw_spread`. A visit on the address resolves it.

## How the numbers were obtained (no leakage)

- **Fitting** uses only leave-one-out *visit pseudo-labels* (1,134 visited addresses): hide an address's own
  visits, predict it from everything else, minimise a Huber loss. Surveyed truth is never used for fitting.
  2-fold check across localities: tuned on one half, scored on the other (`results/tuning_report.csv`).
- **Evaluation** on the 100 surveyed addresses is out-of-fold with the shared `folds.csv` (fold 0 = official
  test, folds 1-5 = development). Surveyed truth enters only as neighbour knowledge from folds not being scored.
  `test_task3.py` proves that changing a fold's own truth, or the test fold's truth, changes no other prediction.
- **Caveat carried from Task 1-2:** the official-test labels were exposed in earlier exploratory work, so fold 0 is
  a sanity check, not a blind test.

## Results

100 surveyed addresses, out-of-fold (`results/pin_model_scores.csv`, bootstrap intervals inside):

| | old geocoder | pin model |
|---|---:|---:|
| All 100: median / P90 | 376 / 839 m | **46 / 369 m** |
| Dev folds (85) | 363 / 748 m | 39 / 351 m |
| Official-test fold (15, see caveat) | 456 / 2,845 m | 68 / 2,473 m |
| Within 50 / 100 / 250 m (all 100) | 5% / 9% / 35% | **51% / 65% / 86%** |
| Visited addresses (49): median | 384 m | **11 m** |
| Unvisited addresses (51): median / P90 | 363 / 653 m | **132 / 535 m** |

Ablation (all 100, median / P90): old pin 376/839 -> +own visit 126/625 -> +locality groups 119/550 ->
+street-key neighbours 69/550 -> +grid regression 37/537 -> +landmarks & POIs **46/369**.
On the 100 labels the landmark layer moves the median up a little (noise: a handful of addresses) and cuts P90;
on the 1,231-address hide-and-predict check every layer helps (`results/pin_model_ablation.csv`).

**The number that matters for the ~57% of addresses with no visit** - hide-and-predict on 1,231 visited addresses
(their own visits hidden, no surveyed truth involved): median **320 -> 126 m**, P90 797 -> 607 m, share within
100 m 12% -> 42%. Locality-tier addresses improve most (354 -> 139 m); street tier 121 -> 73 m.

`raw_spread` is informative: Spearman with error 0.89 on the surveyed set and 0.69 on hide-and-predict; about 90%
of errors fall inside 2.15 x `raw_spread` (the Gaussian ideal). It is *not* a calibrated radius; Task 4 does that.

## What it cannot do

- **Pincode-tier addresses with no locality in the text** stay near the old pin (median ~1.3 km): the data does not
  contain the information. They dominate the test-fold P90. Route them to "verify first", or let one visit fix them.
- **Truth that disagrees with the text** (KYC says one locality, the borrower is elsewhere): neighbours pull the
  wrong way; a tight cluster of agreeing visits is protected from being overruled, a single weak visit is not.
- Surveyed truth covers residences only; office / native-village accuracy is unvalidated.
- 100 labels is small. Treat strata under ~10 addresses (rooftop 1, pincode 10) as anecdotes.
- Coordinates are local metres with no CRS, so "metres" is inferred from the task sheet.

## Files

| file | purpose |
|---|---|
| `address_parser.py` | rule-based multilingual parser -> `results/address_features.csv` |
| `pin_model.py` | the model (`Params`, `PinModel.fit/predict/update`, `fuse`) |
| `tune_pin_model.py` | fits sigmas on visit pseudo-labels -> `results/pin_model_params.json` |
| `run_task3.py` | CV, ablation, hide-and-predict, writes `pins.csv` |
| `test_task3.py` | leakage / online-update / visit-protection / output checks |
| `pins.csv` | **deliverable**: `address_id, pin_x, pin_y, raw_spread, method_used` (+ `clues_used, locality_used, n_locality_hyp, pin_source, has_own_visit`). The 100 surveyed rows are out-of-fold so `eval.py` scores the file directly |
| `results/pin_model_oof_errors.csv` | per-address out-of-fold errors for Task 4's conformal calibration |
| `notebooks/pin_model_experiments.ipynb` | tables and plots |

## Hand-off

- **Task 4:** calibrate radii on `pin_model_oof_errors.csv` (surveyed) plus `hide_and_predict.csv` (1,231 labels,
  much larger), by `method_used` and `precision`; directions from POIs near `pin_x, pin_y`. `raw_spread` is the
  natural scale; low-`raw_spread` own-visit pins need almost no radius, pincode-without-locality needs a large one.
- **Task 5:** `PinModel.update()` is the "every visit improves the next guess" story; `n_locality_hyp > 1` flags
  addresses to verify before a visit is planned.
