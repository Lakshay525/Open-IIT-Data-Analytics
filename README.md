# Address geocoder that learns from field visits

Open IIT Data Analytics | Team AnalyticsOnTop | Problem Statement 3 (CreditNirvana)

Commercial geocoders drop Indian addresses at the pincode or locality centre, so field agents start hundreds of metres from the house and one visit in four ends "address not traceable". This project gives every address a **pin**, a **calibrated confidence radius**, a **confidence tier that says what to do** (visit now / visit with directions / verify first) and **offline landmark directions**, and it **improves as field visits come in**: a confirmed visit becomes evidence for every address on the same street.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py              # everything from the raw CSVs, about 5 minutes
python run_pipeline.py --fast       # skips the stress test, learning curve and notebook
python run_pipeline.py --retune     # also refits the model's error settings
```

Then open `outputs/demo_map.html` (a single offline page: old pin vs our pin vs true location, confidence circles, directions).

## Results (synthetic data, 100 surveyed addresses, all out-of-fold)

| | Old geocoder | This project |
|---|---:|---:|
| Median / 90th-percentile error | 376 / 839 m | **46 / 369 m** |
| Within 50 / 100 / 250 m | 5 / 9 / 35% | **51 / 65 / 86%** |
| Addresses with a field visit (49), median | 384 m | **11 m** |
| Addresses without a visit (51), median | 363 m | **132 m** |
| No-visit addresses, large check (1,231), median | 321 m | **126 m** |
| 90% radius contains the truth | n/a | **91%** (95% CI 84-95%) |

More numbers are written to `outputs/summary.md` on every run.

## How it works

1. **Evidence** (`geocoder/evidence.py`): turns each noisy GPS trail into one trusted point with a role (met borrower, locked house, met at a shop, failed search...) and an uncertainty; reused photos and repeated stops are down-weighted and one agent's influence is capped (`geocoder/integrity.py` stress-tests this with synthetic fake visits).
2. **Parser** (`geocoder/parser.py`): rule-based, multilingual (English, Hinglish, Kanglish; Latin, Devanagari, Kannada script; typo tolerant). Extracts locality, cross/main, gali/ward, block/road and landmark type.
3. **Pin model** (`geocoder/model.py`): combines clues by trust (inverse variance) and drops a clue that contradicts the rest. Clues: the address's own visit, neighbours on the same street, a **street-number grid** (cross/main numbers are coordinates, so a per-locality regression places streets nobody has visited), same-landmark neighbours and the old pin. `PinModel.update()` adds new visits without retraining. Error settings are fitted by hiding each visited address's own visit and predicting it (`geocoder/tuning.py`); surveyed truth is never used for fitting.
4. **Confidence** (`geocoder/confidence.py`): split conformal prediction turns the model's uncertainty into 50% / 90% radii, calibrated on data the model never saw; adds tiers, reason codes, text directions that work offline, a PS2 export and per-town offline packs.
5. **Analyses** (`geocoder/analysis.py`): indicative field impact, learning curve (error falls as visits accumulate), pilot sample sizes. `geocoder/present.py` builds figures, the demo map and the summary. `geocoder/evaluate.py` runs the cross-validated evaluation and writes `outputs/pins.csv`.

## Repository layout

```
run_pipeline.py        one command that rebuilds everything
geocoder/              the code (one module per stage; scoring.py is the shared metre-error scorer)
tests/                 data checks, leakage checks, calibration and direction checks
data/                  supplied inputs (synthetic)
outputs/               pins.csv, predictions.csv, ps2_location_confidence.csv, evidence.csv, folds.csv,
                       offline_pack/, demo_map.html, summary.md, results/ (diagnostic tables and figures)
notebooks/             analysis.py (source) and analysis.ipynb (executed)
docs/                  design.md (integration + file formats), pilot_plan.md, compliance.md
```

Main deliverable columns are described in `docs/design.md` (appendix). `outputs/predictions.csv` holds `address_id, pin_x, pin_y, radius_50, radius_90, directions, confidence_tier` plus reason codes and a suggested action.

## Evaluation rules

- `outputs/folds.csv` is frozen: the 15 official-test addresses are fold 0 and the rest form five development folds with whole localities held out together. Surveyed truth enters the model only as neighbour knowledge from folds that are not being scored; `tests/test_model.py` proves a fold's predictions cannot depend on its own labels.
- The official-test labels were seen during early exploration, so fold 0 is a sanity check, not a blind result.

## Tests

`python -m tests.test_data`, `python -m tests.test_model`, `python -m tests.test_outputs` (the pipeline runs them last).

## Known limits

- All data is synthetic, so real-world accuracy is unproven until a field pilot (`docs/pilot_plan.md`).
- About 10% of addresses name no locality, only a pincode. Several localities about 1.4 km apart stay possible, so these stay near the old pin, are flagged low confidence and sent to verify-first. They set the 90th percentile.
- Only 100 surveyed addresses (residences only), so small strata give wide intervals.
- Coordinates are assumed to be x = east, y = north (no CRS in the data); change `AXES` in `geocoder/confidence.py` if not. Hinglish/Kanglish direction phrases need native-speaker review.
