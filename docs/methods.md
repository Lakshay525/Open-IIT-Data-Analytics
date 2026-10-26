# Methods: how every number is produced

This is the technical reference. `README.md` says what the project does, `docs/design.md` says how it plugs into CreditNirvana's systems, and `outputs/summary.md` holds the headline numbers (regenerated on every run; this document does not repeat them). Numbers quoted below come from the run of 26 Oct 2026 and are marked as such.

Notation. An address `a` has a set of clues `i`. Each clue gives a location guess `mu_i` (2-vector, metres, x east, y north) and a 1-D error scale `sigma_i`. All distances are Euclidean in the supplied local metre grid.

## 1. Pipeline at a glance

```
raw tables ──► validation ──► scoring (frozen folds) ──► evidence ─┐
                                                                   ├─► pin model ──► calibration ──► outputs
address text ──► parser ───────────────────────────────────────────┘      ▲   │            │
                                                                          │   └─ update()  └─ directions, tiers, PS2 file
                                         tuning (fits the sigmas) ────────┘
```

| Stage | Module | Statistical idea |
|---|---|---|
| Visit cleaning | `evidence.py` | Dwell-cluster endpoint, DBSCAN repeated-stop flag, role rules, quantile regression for per-visit error, multiplicative trust weights |
| Address parsing | `parser.py` | Rules + fuzzy matching; no learning |
| Pin model | `model.py` | Precision-weighted fusion (Gaussian product of experts) with a leave-one-out robust gate |
| Tuning | `tuning.py` | Huber loss on hide-and-predict pseudo-labels, Powell search |
| Confidence | `confidence.py` | Split conformal prediction, per group and precision tier |
| Evaluation | `evaluate.py`, `benchmark.py` | Frozen locality-grouped folds, ablation ladder, estimator comparison |

## 2. Visit cleaning (`evidence.py`)

**One point per visit.** Within a window at the end of the GPS trail (length = dwell time clipped to 120 to 600 s) the model keeps the points within `max(20 m, 2 x median GPS accuracy)` of the last fix (the dwell cluster) and takes their coordinate-wise median. The cluster's share of the window, its radius, the trail's straightness (net distance over path length) and a trail type (`direct`, `searching`, `stationary_or_sparse`) feed the weights.

**Repeated-stop detector.** For each agent with at least five visits, DBSCAN (eps 15 m, min 5 points) over the agent's visit endpoints finds places where the agent "visited" five or more different addresses. If that place is 750 m or more from those addresses' old pins, the visits get an integrity factor of 0.65, 0.45 or 0.20 depending on how many addresses it spans. It uses no truth.

**Role.** Rules over the outcome code and the remark text (English, Hinglish, Kanglish regexes) assign one of `contact_at_address`, `structure_only`, `historical_structure`, `contact_elsewhere`, `failed_search`. Only the first three can become a pin; `contact_elsewhere` and `failed_search` never move a residence.

**Per-visit uncertainty.** A gradient-boosted *quantile* regression (alpha = 0.80, 90 trees, depth 2, 13 features) predicts the 80th percentile of `log(1 + d)` where `d` is the distance between this visit's endpoint and the consensus of at least two *other* agents at the same address (leave-one-agent-out). Features: GPS accuracy, cluster radius, dwell, cluster fraction, trail straightness, agreement with other visits, photo reuse, off-site flag, failed/shifted structure flags, trail type, agent integrity. Predictions are floored at twice the GPS accuracy and at 250 to 1000 m for known-ambiguous evidence (reused photo, off-site meeting, sparse trail). No surveyed coordinate is used. This is the one place the project uses supervised ML, because the target (agent disagreement) is observed and the features are numerous.

**Trust weight.** A product of factors in [0, 1]:

```
w = outcome_factor  x exp(-max(acc - 8, 0)/35)  x (0.25 + 0.75 min(dwell/300, 1))  x (0.35 + 0.65 cluster_fraction)
    x trail_factor  x agreement  x agent_integrity  x 80/(80 + uncertainty_m)  x (0.05 if photo reused across addresses else 1)
```

Outcome factors: met borrower 1.0, cash collected 0.9, met family 0.5, locked premises 0.25, neighbour says shifted 0.12, no such person 0.10. After that, the total weight one agent contributes to one address is capped at 0.35. A visit with `w < 0.001` stays in the table for audit but cannot create a pin.

## 3. Address parser (`parser.py`)

Normalise the text (case, script, punctuation), tokenise, and map tokens to canonical forms with dictionaries for abbreviations and relation words in English, Hinglish and Kanglish (Latin, Devanagari and Kannada script). Localities and landmark types are matched with a token-similarity score (`difflib`) so "Haanuman Temple" and "Psot Office" resolve. Output fields: locality, cross, main, gali, ward, block, road, landmark type, relation word, script. Rules were chosen over a language model because the text is templated, nothing leaves the machine, and every extracted field traces to a phrase. Limit: it will not generalise to free-form addresses without extending the dictionaries.

## 4. The pin model (`model.py`)

### 4.1 Fusion

With weights `w_i = 1/sigma_i^2` and `W = sum w_i`:

```
x_hat = sum_i w_i mu_i / W
```

The reported spread is not the textbook `1/W` alone, because clues that disagree should widen it. The model uses

```
var = 1/W + chi2 / (W (m - 1)),   chi2 = sum_i w_i ||mu_i - x_hat||^2,   spread = inflate * sqrt(var)
```

where `m` is the number of clues kept. This is the usual Birge-ratio style inflation: if the clues agree, `chi2/(m-1)` is about 1 and the spread is about `sqrt(2/W)`; if they conflict it grows. `inflate` (1.37 fitted) absorbs the remaining correlation between clues. The final radius is set separately by conformal calibration (section 7), so `spread` only needs to *rank* addresses by difficulty, not be a probability.

### 4.2 Clues and their sigmas

| Slot | Clue (`mu`) | `sigma` |
|---|---|---|
| `baseline` | old geocoder pin | `sb[precision tier]`: rooftop 10, street 84, locality 848, pincode 1,250 m (fitted, run of 26 Oct) |
| `visit` | weighted median of the address's own eligible visits | `sqrt(a_visit^2 + (b_visit * scatter)^2)` = 15 m floor |
| `cm`, `br`, `gali` | weighted median of known pins sharing locality + cross/main, block/road, gali/ward | `sqrt((a (1 + c_n/n))^2 + (b_scatter * scatter)^2)` |
| `lmcm`, `lmbr`, `lm` | same, additionally sharing the landmark type (`lm`: locality only) | same form |
| `locgrp`, `loccent` | known pins in the locality; locality centroid | `a_locgrp`, `a_loccent` |
| `poi` | landmarks of the named type near the old pin, Gaussian-weighted | same form |
| `grid` | street-number regression (4.4) | `sqrt((a_grid * predictive spread)^2 + f_grid^2)` |

`n` is the number of known pins behind the clue and `scatter` their RMS distance from the median, so a street with one known house is trusted less than one with six that agree. "Known pins" are cleaned visit pins of other addresses (about 43% of addresses) plus, during evaluation only, surveyed truths of training folds. The address's own pin is always excluded (`exclude_self`).

### 4.3 Robust gate

Fusion is vulnerable to one wrong clue (a mis-parsed locality, a copied visit). For each valid clue the model computes the leave-one-out fusion `x_hat_{-i}` and the standardised conflict

```
r_i = ||mu_i - x_hat_{-i}|| / sqrt(sigma_i^2 + 1/W_{-i}),   then divided by a per-clue gate scale
```

The clue with the largest `r_i` is dropped if `r_i > k` (k = 3.19 fitted). This repeats at most three times and never leaves fewer than one clue. Gate scale is 1 for ordinary clues; for the address's own visit it is 1,000 when at least two visits agree within 60 m (a tight cluster is ground truth about where the borrower lives and must survive even a wrong KYC locality) and 2.5 when there is a single weak visit.

### 4.4 Street-number grid

In numbered layouts (Nth cross and Mth main; block and road) position is a smooth function of the street numbers. For each locality and number-set, known pins are regressed on the numbers with a **local-linear kernel regression**:

* design `A = [1, X - x0]` so the intercept is the prediction at the target `x0`;
* Gaussian kernel weights with a bandwidth per column (effectively infinite for cross/main, which gives one global linear fit per locality; (0.45, 2.5) for block/road, which fits each block separately);
* **iteratively reweighted** (4 passes) with Huber-like weights `min(1, 1.5 s / r)` where `s = max(1.4826 MAD, 15 m)`, so a wrongly placed neighbour does not drag the fit;
* a ridge of 0.01 on the slope terms for stability;
* predictive spread `sqrt(res^2 (1 + leverage))`, with the leverage of the target point, so extrapolating to a corner of the grid is penalised.

Among the available number-sets the one with the smallest predictive spread is used. A fit needs at least `k + 3` known points and `k + 2` with non-negligible kernel weight. This clue is the largest single gain after the visits themselves (dropping it raises the hide-and-predict median by about 38 m in the sensitivity table, section 8).

### 4.5 No locality in the text

When the parser finds no locality and the address has no visit, the pincode leaves several candidate localities. Using neighbour clues from a wrong locality would add false confidence, so the model fuses only `baseline`, `locgrp` and `loccent` per hypothesis, weights hypotheses by a ward prior (counts of the ward in each candidate locality, plus 0.5) and reports the mixture mean with the between-hypothesis variance added to the spread. For addresses with a parsed locality the single hypothesis is used. Where several hypotheses and some evidence exist, hypotheses are scored by a robust log-evidence in which every clue (including gated ones) pays an outlier mixture penalty (eps = 0.05), so a hypothesis cannot win by discarding the clues that contradict it.

### 4.6 Online update

`update(new_evidence)` merges evidence by `visit_id`, rebuilds the known-pin table and the key index (about 0.4 s for all 2,880 addresses; an incremental index would make it constant time). Every address sharing a street key, landmark key or locality benefits at the next `predict`. Nothing is retrained; the sigmas stay fixed.

## 5. Fitting the sigmas (`tuning.py`)

**Pseudo-labels.** For each visited address the model hides that address's own visit and predicts it from everything else. The target is the cleaned visit pin itself (8 to 10 m from survey truth where both exist). This gives 1,134 labels with a parsed locality, about 11 times the surveyed set, and keeps the surveyed truth clean for evaluation.

**Objective.** Mean Huber loss with scale 120 m on the Euclidean error, plus a ridge `0.002 ||log(theta) - log(theta_0)||^2` toward the defaults, so a poorly identified sigma cannot run away. The 16 tunable settings (`Params.TUNABLE`) are optimised in log space with Powell's method from three starts.

**Honest check.** Splitting localities into two halves, tuning on one and scoring on the other gives a median of about 126 m and 116 m on the two halves, against 113 m for the in-sample fit (`outputs/results/tuning_report.csv`). So roughly 10 m of the in-sample figure is overfit, and the held-out-locality figure is the one to quote for unseen localities.

**What tuning tells us.** Some fitted sigmas are very large (`a_gali` about 3.8 km, `a_locgrp` about 1.7 km), meaning those clues carry almost no weight once the street-number grid, street keys and landmarks are present. Section 8 confirms they can be removed with no change. We keep them because they cost nothing and act as safety nets in other towns.

## 6. Where the old pin helps and where it hurts

The old geocoder's own `precision` field (rooftop, street, locality, pincode) sets its sigma. Rooftop pins (about 17 m median error) are almost never overruled; locality and pincode pins are treated as weak, nearly uninformative clues. Section 8 shows the pattern: where the old pin was already at street or rooftop precision, the model rarely improves on it and sometimes does worse (31% worse than the old pin where only the baseline clue was available at locality precision).

## 7. Calibrated confidence (`confidence.py`)

The model's `raw_spread` is turned into radii with **split conformal prediction**.

* Score: `s = e / raw_spread`, with `e` the distance error on data the model did not see.
* Radius at level `p` in {0.5, 0.9}: `r_p = Q_p(s) * raw_spread`, where `Q_p` is the empirical quantile at level `ceil((n + 1) p) / n` (finite-sample correction).
* Scores are pooled in three **groups** that behave differently: addresses with their own visit (leave-one-out scores), addresses without a visit (scores from the 1,231-address hide-and-predict set), addresses with no locality in the text. Within a group, a precision-tier cell is used when it has at least `MIN_TIER_N` scores, otherwise the pooled group.
* Verification never shares data with calibration: no-visit radii are fitted on the pseudo-label set and checked on the surveyed addresses; visited addresses use leave-one-out.
* `location_confidence` is the empirical CDF of the normalised score at `100 m / raw_spread`: a calibrated probability of being within 100 m, passed to PS2.
* Coverage intervals are Wilson 95% intervals.

Tiers are fixed cuts on the 90% radius (high 60 m, medium 300 m), and reason codes are deterministic functions of the dominant clue and the group, so every tier and every code can be audited.

Limits: conformal coverage is marginal, not conditional. The overall 90% coverage holds, but a small stratum (the six no-locality surveyed addresses) can be far off and its interval is wide. Exchangeability between calibration and deployment addresses is assumed; a new town or a drift in visit practice breaks it, which is why the pilot plan recalibrates monthly and alerts when coverage leaves 85% to 97%.

## 8. Is the model actually better than alternatives? (`benchmark.py`)

All estimators below receive the same clues from the same `PinModel`, and are scored on two sets: the visited addresses with their own visit hidden (n = 1,134, target = cleaned visit pin) and the surveyed addresses, out-of-fold (n = 91 with a parsed locality). Median / 90th percentile in metres, run of 26 Oct 2026 (`outputs/summary.md` regenerates this):

| Estimator | Visited, visit hidden | Surveyed, out-of-fold |
|---|---:|---:|
| Old pin | 302 / 580 | 342 / 625 |
| Locality centroid | 339 / 545 | 366 / 536 |
| Street-key median | 211 / 525 | 66 / 418 |
| GP on street numbers (kriging) | 209 / 533 | 57 / 436 |
| Gradient-boosted trees on parsed features | 137 / 410 | 50 / 310 |
| Equal-weight average of clues | 229 / 446 | 230 / 496 |
| Fusion, no gate | 115 / 419 | 47 / 341 |
| **Fusion with gate (shipped)** | **113 / 383** | **39 / 224** |

Reading it honestly:

* The cleverness is in combining clues. No single clue family gets under 200 m on the visited set; equal weighting is worse than the best single clue; weighting by reliability roughly halves the error.
* **Gradient boosting is a serious competitor** on the median (137 m against 113 m on 1,134 addresses; 50 m against 39 m on 91, a difference within noise at that size). The fusion model's edge is clearer in the tail (P90 383 m against 410 m, and 224 m against 310 m), and it additionally gives an explainable pin (which clues, which weights), a calibrated spread and an update without retraining. The gradient-boosting numbers show the case for the fusion model is *not* "ML cannot do this".
* The robust gate helps mostly in the tail (P90 419 m to 383 m on visited).

**Sensitivity** (`benchmark_sensitivity.csv`, visited set): multiplying every sigma by 0.5 to 2 changes the median by 4 to 14 m; moving the gate threshold between 2.5 and 6 changes it by under 2 m; the fitted setting is at or near the optimum of both. Dropping one clue at a time: the street-number grid costs +38 m, block/road +6 m, landmark group +6 m; cross/main, gali, landmark-street keys, locality group and locality centre cost about 0 (redundant given the others).

**Error taxonomy** (`benchmark_error_taxonomy.csv`): the model is better than the old pin by more than 25 m for 83% of locality-precision addresses placed by the street grid (median 350 m to 97 m) but is worse for 31% of the 64 addresses where only the old pin and locality clues exist, and the landmark-POI method is the weakest of the clue families (median 213 m).

## 9. Evaluation protocol and leakage controls (`scoring.py`, `evaluate.py`)

* **Folds** are frozen in `outputs/folds.csv`: the 15 official-test addresses are fold 0; the other 85 form five development folds, with whole localities held out together so neighbours cannot leak across folds. `run_pipeline.py` hashes the file before and after a run and reports any change.
* **Surveyed truth enters the model only as neighbour knowledge from folds that are not being scored**. `tests/test_model.py` corrupts a fold's truth and asserts that its predictions do not change.
* Fitting (sigmas, uncertainty model) uses only visit pseudo-labels. Calibration and verification use different data.
* Three test sets answer different questions: the 100 surveyed addresses (independent truth but few); the 1,134 or 1,231 hide-and-predict addresses (many labels but the label is itself a visit); the larger set is used for choosing layers and radii because 100 labels are too few to separate small differences.
* The official-test labels were seen during early exploration, so fold 0 is a sanity check, not a blind result.
* Confidence intervals are address-level bootstrap intervals (`scoring.py`).

## 10. Weaknesses of the method itself

* **Independence is assumed, and is false.** Neighbour clues share members (the same known pins feed `cm`, `lmcm` and `locgrp`), so the product of experts double-counts. The count term `(1 + c_n/n)`, the scatter term, the gate and the inflation factor are practical fixes, not a joint model; correlated clues are why the `1/W` spread is only used to rank difficulty.
* **Isotropic Gaussian errors.** Street-aligned errors (along a road rather than across it) are real; a covariance per clue would be a natural extension.
* **The label is a visit, not a doorstep.** Pseudo-labels carry the agent's stopping bias (a few metres on this data). Where agents systematically stop at a gate or shop, pins inherit that.
* **Regular layouts only.** The street-number grid helps where numbers encode position (T1, T3) and does little where they do not.
* **Visit feedback.** The policy decides which addresses get visited; the evaluation on addresses without visits and on independent survey truth limits but cannot remove that selection effect (`docs/design.md` section 6 recommends an exploration share).
* **Synthetic data.** Every number here is on synthetic data; real address text will be messier and the parser dictionaries will need extending.

## 11. Complexity and cost

Known-pin index: linear in known addresses. One prediction: a handful of dictionary lookups, up to four small weighted regressions of tens of points, and a fusion of at most 12 clues. Measured on a laptop: whole pipeline about two minutes (excluding stress test and benchmark), one address about 1 ms, an update about 0.4 s. The model is territory-local, so a deployment partitions by territory and runs partitions in parallel.

## 12. Reproducing everything

```
python run_pipeline.py            # all stages, including benchmark.py (about 6 minutes)
python run_pipeline.py --fast     # skips stress test, benchmark, learning curve, notebook
python -m geocoder.benchmark      # only the comparison (about 40 s)
```
