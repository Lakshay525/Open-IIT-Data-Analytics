# Address geocoder that learns from field visits

Open IIT Data Analytics | Team AnalyticsOnTop | Problem Statement 3 (CreditNirvana)

Commercial geocoders drop Indian addresses at the pincode or locality centre, so field agents start hundreds of metres from the house and one visit in four ends "address not traceable". This project gives every address a **pin**, a **calibrated confidence radius**, a **confidence tier that says what to do** (visit now / visit with directions / verify first) and **offline landmark directions**, and it **improves as field visits come in**: a confirmed visit becomes evidence for every address on the same street.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py              # everything from the raw CSVs, about 6 minutes
python run_pipeline.py --fast       # skips the stress test, learning curve and notebook
python run_pipeline.py --retune     # also refits the model's error settings
```

Then open `outputs/demo_map.html` (a single offline page: old pin vs our pin vs true location, confidence circles, directions).

## Live demo (web app)

`web/` is a deployable prototype: a front end plus Python serverless functions running the same model. Try an address, or confirm a visit and watch the neighbouring pins move.

```bash
pip install numpy pandas
python web/dev_server.py          # http://127.0.0.1:8000
```

Deploy on Vercel with Root Directory = `web` (steps in `docs/deploy.md`).

## Results (synthetic data, 100 surveyed addresses, all out-of-fold)

| | Old geocoder | This project |
|---|---:|---:|
| Median / 90th-percentile error | 376 / 839 m | **46 / 369 m** |
| Within 50 / 100 / 250 m | 5 / 9 / 35% | **51 / 65 / 86%** |
| Addresses with a field visit (49), median | 384 m | **11 m** |
| Addresses without a visit (51), median | 363 m | **132 m** |
| No-visit addresses, large check (1,231), median | 321 m | **126 m** |
| 90% radius contains the truth | n/a | **91%** (95% CI 84-95%) |

Every number on this page is regenerated in `outputs/summary.md` (which also compares the model with simpler and gradient-boosting alternatives). Evaluation rules and leakage controls: [`docs/methods.md`](docs/methods.md) section 9.

## How it works

Five stages, each one module (details and equations in [`docs/methods.md`](docs/methods.md)):

| Stage | Module | One line |
|---|---|---|
| Evidence | `evidence.py`, `integrity.py` | Each GPS trail becomes one trusted point with a role, an uncertainty (quantile model) and a trust weight; reused photos and repeated stops are down-weighted; a fake-visit stress test measures the damage. |
| Parser | `parser.py` | Rule-based, multilingual, typo tolerant: locality, cross/main, gali/ward, block/road, landmark type. |
| Pin model | `model.py`, `tuning.py` | Precision-weighted fusion of up to 12 clues with a robust gate; a per-locality street-number regression places streets nobody has visited; `update()` adds a visit without retraining; settings are fitted on visit pseudo-labels, never on surveyed truth. |
| Confidence | `confidence.py` | Split conformal 50% / 90% radii per group, tiers, reason codes, offline directions, PS2 export, per-town packs. |
| Evidence of benefit | `evaluate.py`, `benchmark.py`, `analysis.py` | Leakage-safe CV and ablation, comparison with simpler and ML alternatives, learning curve, impact mapping, pilot sample sizes. |

## Repository layout

```
run_pipeline.py        one command that rebuilds everything
geocoder/              the code (one module per stage; scoring.py is the shared metre-error scorer)
tests/                 data checks, leakage checks, calibration, directions and web-app checks
web/                   live demo: public/ (front end), api/ (serverless functions), dev_server.py, vercel.json
data/                  supplied inputs (synthetic)
outputs/               pins.csv, predictions.csv, ps2_location_confidence.csv, evidence.csv, folds.csv,
                       offline_pack/, demo_map.html, summary.md, results/ (diagnostic tables and figures)
notebooks/             analysis.py (source) and analysis.ipynb (executed)
docs/                  methods.md (equations, fitting, evaluation, benchmark), design.md (integration + file formats),
                       pilot_plan.md, compliance.md, deploy.md
```

Main deliverable columns are described in `docs/design.md` (appendix). `outputs/predictions.csv` holds `address_id, pin_x, pin_y, radius_50, radius_90, directions, confidence_tier` plus reason codes and a suggested action.

## Tests

`python -m tests.test_data`, `python -m tests.test_model`, `python -m tests.test_outputs`, `python -m tests.test_web` (the pipeline runs them last).

## Limits (full list in [`docs/methods.md`](docs/methods.md) section 10)

- All data is synthetic, so real-world accuracy is unproven until a field pilot (`docs/pilot_plan.md`).
- About 10% of addresses name no locality, only a pincode. Several localities about 1.4 km apart stay possible, so these stay near the old pin, are flagged low confidence and sent to verify-first. They set the 90th percentile.
- Only 100 surveyed addresses (residences only), so small strata give wide intervals.
- Coordinates are assumed to be x = east, y = north (no CRS in the data); change `AXES` in `geocoder/confidence.py` if not. Hinglish/Kanglish direction phrases need native-speaker review.
