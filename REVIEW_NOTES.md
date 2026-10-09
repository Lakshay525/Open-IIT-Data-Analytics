# Review decisions and open risks

This log compares the separate review PDF with the revised implementation.
It records where the data supports a change and where a proposed interpretation
would be too strong. All quoted visit-error diagnostics below are development
survey observations unless explicitly marked otherwise.

| Suggestion | Decision and evidence |
| --- | --- |
| Use failed visits | Applied selectively. `locked_premises` trails have 7.9 m median endpoint error in 43 development surveyed rows. `no_such_person` has only three such rows; both are marked structure-only and downweighted. `neighbour_says_shifted` is marked historical and weak; it does not certify a current residence. A contradictory “address not found” remark removes a visit from pin use. Usable surveyed-address coverage rises from 34 contact-at-address to 43 with the gated structure/history signals, not the review's suggested 46. The extra nine are not model accuracy evidence. |
| Treat `address_not_traceable` as negative | Applied as a **failed-search flag**, not a spatial exclusion claim. In development survey rows, the endpoint error for the `address_not_traceable` outcome has a 751 m median and the broader failed-search role a 1.29 km median. Its endpoint does not identify a safe exclusion region around the baseline. |
| Photo reuse | Applied cross-address exact-hash flag, severe weight penalty and uncertainty floor. The source has 172 visits across 11 repeated hashes; eight development-survey rows have a roughly 6.25 km median endpoint error versus 12.7 m for 190 others. This is a very small, non-random diagnostic sample. A low retained weight remains auditable but falls below the 0.001 pin-use threshold. |
| Remarks and landmarks | Applied explicit English/Hindi/Kannada rules for shop meetings, “not found” contradictions and correction hints. Meetings at shops are excluded from residence-pin evidence; correction text is retained as `landmark_hint` for Task 3. Text is not converted into a coordinate here. Rules can miss paraphrases and require manual spot checks before operational use. |
| Fitted uncertainty | Applied an 80th-quantile proxy fitted to leave-one-agent-out consensus at development addresses, then checked on surveyed development visits. On 127 eligible rows the uncertainty radius covers about 95% of observed endpoint errors. Spearman correlation between uncertainty and observed error is about +0.28. The final influence weight does not rank error well on that already-filtered sample, so neither field is advertised as a calibrated probability. Multi-agent agreement is an imperfect pseudo-label and may be wrong collectively. |
| Evaluation improvements | Added within-50/100/250 m shares and intervals, visit-status baseline cuts, account split metadata and grouped development folds. Official test accounts are fold 0; five development folds have 17 addresses each, grouped by the old pin's nearest locality centroid. The grouping is a proxy because source address rows have no locality ID. All-100 baseline remains descriptive. Prior exploratory access to official-test survey labels prevents a truly blind final test claim. |
| Real duplicate-photo test | Added the observational, surveyed-row error comparison and a weight-vs-error plot. Photo reuse is a warning signal, not a confirmed fraud label or a causal test of the downweighting rule. |
| Stronger synthetic attacks | Fixed a major flaw in the first notebook: it used the pre-attack clean pin to assign a fake point's agreement penalty. The revision injects plausible visit features and recomputes all safeguards without a clean reference; clean/truth points are used only to measure outcomes afterwards. Safeguarded median still has kilometre-scale P90 shifts. An independent-agent cluster method avoids large shifts on emitted pins in these fixtures but abstains on many addresses. The fixtures are synthetic endpoint-feature injections, not replayed raw GPS trails. |
| Stationary trails / address types / retention | Stationary/sparse visits have a .10 trail factor and 250 m uncertainty floor. Office and native-village rows are tagged; survey truth covers residences only. Documentation distinguishes restricted raw-trail handling in production from keeping synthetic raw trails in the reproducible bundle. |

## Additional issues found in the first package

- The earlier explanation called the agent cap “35% of total evidence.”
  The code actually caps one agent's **absolute weight sum at 0.35 per
  address**. The revised explanation states this correctly.
- The first folds ignored `splits.csv`, even though the shared data README
  says all three problem statements use it. `accounts.csv` and
  `splits.csv` are now included and their joins validated.
- The first stress comparison overestimated defence because a clean visit
  median informed attack detection. The new stress test does not use it
  until scoring outcomes.
- The first pipeline gave every non-contact visit zero pin weight, losing
  some real structure observations. The revision preserves the distinction
  between a located structure and a confirmed current borrower residence.
- Rounding trust weights to three decimals can create a row that says
  `eligible_for_pin=True` but displays `weight=0`. CSV output now keeps
  nine significant digits, and eligibility requires weight at least 0.001.

## Before a final external performance claim

1. Ask the problem owner for the actual coordinate reference system and
   whether the synthetic surveyed truth represents a current or historical
   residence when a neighbour says the borrower shifted.
2. Hand-check multilingual remark patterns and a sample of flagged photo
   hashes. An exact hash reused across addresses is unusual but not proof
   of agent misconduct.
3. Test the entire Task 3 model with leakage-safe out-of-fold predictions.
   The ~7 m visit-derived median applies only where visits exist and should
   never be extrapolated to all 2,880 addresses.
4. Calibrate prediction radii on data that did not choose the model/rules,
   and report both accuracy and abstention. Because test labels have already
   been seen in exploratory work, consider obtaining a new blinded survey
   sample if a strict external performance claim is required.
5. For a live deployment, reimplement all global checks as time-respecting
   features; the current batch view can use visits that happened later.
