# Address geocoder: reviewed Tasks 1 and 2

This is a reproducible **data/scoring and visit-evidence package**, not the final
pin model. The supplied CreditNirvana data is invented. The primary task sheet
specifies Task 1 (data/scoring) and Task 2 (visit cleaning); the separate review
PDF motivated the revisions documented in [REVIEW_NOTES.md](REVIEW_NOTES.md).

## Start here

Use Python 3.10+ from this folder. Install the packages in `requirements.txt`
and run:

```bash
python validate_inputs.py
python eval.py
python extract_evidence.py
python integrity_analysis.py
python verify_package.py
```

`verify_package.py` executes both notebooks and checks critical outputs.
You can open `notebooks/baseline_report.ipynb` and
`notebooks/integrity_stress_test.ipynb` in Jupyter for an interactive walkthrough.
Relative paths work when Jupyter starts in the package folder or the notebooks
folder. The latter command takes roughly half a minute on this synthetic data.

## Task 1: data and fair scoring

- `clean_data/` contains the six geocoder tables and five shared tables:
  addresses, field visits, agents, accounts and official account splits. Source
  rows are retained. The 237 `OUT` addresses remain in the address table but
  lack baseline geocodes; they are outside the 2,880-address geocoder population.
- `validate_inputs.py` writes `results/file_quality.csv` and
  `results/validation_checks.csv`. It checks IDs, cross-file links, GPS
  sequence/time, units, official splits and visit-account consistency. Optional
  fields such as PTP IDs or salary-credit day can be blank.
- `eval.py` scores one prediction per surveyed address by Euclidean
  coordinate error in metres. It reports median, P90, shares within 50, 100
  and 250 m, and address-bootstrap 95% intervals. Results are split by
  precision, town, official split, any visit and contact visit. The detailed
  tables are `results/baseline_scores.csv` and
  `results/baseline_address_errors.csv`.
- `folds.csv` assigns all 100 surveyed addresses: official test accounts
  have fold **0** (15 addresses), and the remaining 85 form five development
  folds of 17. Each fold holds out complete proxy locality groups. Groups
  come from the nearest locality centroid to the **old geocoder pin**, never
  from the surveyed truth. Neighboring localities can still leak spatial
  information, so inspect model-specific duplicates and features later.

### Baseline benchmark

| Population | N | Median error | P90 error |
| --- | ---: | ---: | ---: |
| All surveyed addresses | 100 | 376.4 m | 839.2 m |
| With any field visit | 49 | 383.7 m | 1,769.6 m |
| Without a field visit | 51 | 362.9 m | 653.5 m |

Among all 100 addresses, **5%**, **9%** and **35%** of baseline pins lie
within 50, 100 and 250 m of the surveyed point. The 100-row baseline is a
descriptive reference and agrees with the ~376 m value in the task sheet.
The visited/unvisited contrast is not a causal estimate of visit impact.
Pincode addresses have a 1,375.8 m median but only ten surveyed examples;
the rooftop tier has one. Intervals for strata below ten are suppressed.
The source does not provide an EPSG/CRS code. Metre units are inferred from
the task and town-radius fields.

**Test-set caution:** Earlier exploratory review of the supplied data
exposed some official-test survey labels. The revised pipeline does not fit
uncertainty or select visit rules on those labels, but the team's test set is
no longer fully blind. Freeze the approach before any final test comparison,
and disclose this limitation.

## Task 2: one point, role and uncertainty per visit

`extract_evidence.py` converts 160,406 GPS points from 5,578 visits into
`evidence.csv` (one row per visit). It takes a short window at the end of the
trail, selects points near its endpoint, and uses coordinate medians to form
`ev_x, ev_y`. The file includes the required
`visit_id,address_id,ev_x,ev_y,weight,trail_type` fields and diagnostics.

The new fields are important:

| Field | Interpretation |
| --- | --- |
| `evidence_role` | Contact at address, structure only, former structure, contact elsewhere, or failed search |
| `uncertainty_m` | Fitted error-scale proxy in metres for possible pin evidence; blank for failed searches and offsite meetings |
| `weight` | Relative influence after GPS, role, agreement, repeat-stop, photo and uncertainty factors |
| `eligible_for_pin` | May contribute to an address pin after a minimum useful weight of 0.001 |
| `negative_search_signal` | Search failed; retain for review, never interpret as proof of where the truth is |
| `photo_reused_cross_address` | Exact photo hash occurs on distinct addresses |
| `remark_signal`, `landmark_hint` | Simple multilingual remark flags and an unparsed hint for a later landmark model |
| `residence_target`, `address_type` | Distinguish residence from office/native-village addresses |

`met_borrower`, `cash_collected` and `met_family` can support a pin;
family contact is weaker evidence. `locked_premises` and
`no_such_person` can provide **low-weight structure-only** location evidence
when the remark does not contradict the outcome. `neighbour_says_shifted`
is tagged as possible *historical* address evidence and has a small weight:
do not treat it as confirmation of the borrower's current residence.
Meetings at a shop and failed/contradictory searches cannot move the
residence pin. `address_not_traceable` produces only a failed-search signal.
The endpoint alone does not justify excluding the old geocoder region.

One agent can contribute at most **0.35 absolute total weight per address**
before aggregation. This is not a 35% cap on a normalized pin estimate.
Identical photos reused across addresses are strongly downweighted. Very
small weights remain visible for audit but do not count as usable pin evidence.
The remark detector uses explicit Kannada, Hindi and English phrases; it
is not a complete language model.

### Uncertainty method and diagnostic

The script fits a simple quantile model using **651 visits at 175
development addresses**. For a target visit, the training proxy is its
distance from the median location reported by at least two *other* agents
at the same address. No surveyed coordinates train this model, and official
test accounts do not train it. A bad cluster shared by multiple agents can
still fool this proxy.

`results/uncertainty_fit_summary.csv` records the fit. On **127 eligible
visit rows at 38 development surveyed addresses**, the measured endpoint
falls within the reported `uncertainty_m` about **95%** of the time.
This is a conservative, selected sample, not validated calibration on a
representative population. The fitted uncertainty has Spearman correlation
about **0.28** with surveyed endpoint error; the final weight has little
error ranking on this already-filtered subset. Treat the weight as a
transparent influence rule, not a probability of truth.

The corresponding visit endpoint median error is **8.4 m**, compared with
**23.9 m** for the raw check-in on those same eligible rows. These
are selected visits; they do not show improvement for all 2,880 addresses.
Address-level diagnostics are in `results/visit_pin_diagnostics.csv`.
On the 38 development surveyed addresses with usable evidence, a weighted
median of visits has about **6.9 m median error**, versus **362.5 m**
for the baseline on that exact subset. It is an evidence-derived diagnostic,
not a cross-validated final geocoder. All surveyed truth addresses are
residences, so office/native-village pin accuracy is unknown.

### Photo reuse and stress tests

`results/visit_quality_by_group.csv` and `results/weight_vs_error.png`
contain an observational photo-reuse comparison. Eight development-survey
visits have a photo hash reused across addresses; their median endpoint
error is about **6.25 km**, versus **12.7 m** for 190 other surveyed visit
rows. Eight is a small sample, and reuse alone is not a fraud label.

`integrity_analysis.py` and `notebooks/integrity_stress_test.ipynb`
inject synthetic home, shared-stop and rogue-agent visits. They recalculate
all trust factors from the contaminated visits; the defence does not see a
pre-attack clean pin. `results/integrity_stress_summary.csv` reports median
and P90 pin shift, the number of pins emitted, and abstentions. Detailed
rows are in `results/integrity_stress_results.csv`; injected coordinates
are in `results/synthetic_attack_rows.csv`.

The safeguarded weighted median often has **0 m median shift**, but its P90
shift still reaches about **3.72 km (home), 1.40 km (shared stop), and
2.54 km (rogue)**. An independent-agent cluster option reduces measured
P90 among *reported* pins to a few tens of metres in these fixtures, while
abstaining on **22/38**, **22/38** and **7/16** affected addresses. Do not
compare its low tail shift without stating its coverage loss. These are
controlled sensitivity tests, not a real-world fake-visit detection rate.

## For the next teammate

1. Keep `folds.csv` fixed. Train/select models only on folds 1-5.
   Fold 0 is the official test group; prior exposure must be disclosed.
2. Join `evidence.csv` to addresses using `address_id`; use only
   `eligible_for_pin` rows for address-pin aggregation, and inspect
   `evidence_role` when the target is *current borrower residence*.
3. Use `landmark_hint` as unparsed text input, not a coordinate or a
   validated direction. Use the failed-search signal for review/difficulty
   features, not a geometric exclusion zone.
4. Evaluate later pins with `eval.py` on truly held-out predictions. Report
   median, P90, within-radius shares, subgroup sizes and coverage/abstention.
5. If updating online, compute repeat-stop, photo reuse, prior agreement
   and fitted uncertainty from data available **as of the visit time**.
   The included batch analysis can see later visits.

For production data, access to raw trails and photo/remark fields should be
restricted. Keep derived evidence needed for audit and modelling; delete raw
trails after extraction and verification under the organisation's approved
retention and legal policy. This synthetic bundle retains raw
trails so a reviewer can reproduce results.

Tasks 3-5 (the final pin model, calibrated confidence and demo) are not part
of this package.
