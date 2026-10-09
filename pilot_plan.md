# Pilot plan - proving the geocoder in the field

The offline results in this package show the pins are more accurate. They do **not** prove that agents find more borrowers; that needs a controlled field trial. This plan is built from the supplied visit logs (numbers below come from `pilot_power.py`, `impact_analysis.py`).

## 1. Question and hypotheses

* **H1 (primary):** territories using our pins, confidence tiers and directions have a lower share of visits ending *address not traceable* than control territories.
* **H2 (primary):** field agents complete more *productive visits per agent-day* (contact outcomes: met borrower / cash collected / met family).
* **H3 (secondary):** less time from start to check-in; fewer visits needed per productive contact; lower cost per productive visit (CN's range: Rs 150-400).
* **Guardrails:** zero third-party debt-disclosure incidents; no rise in borrower complaints; no fall in recovered amount per productive visit; agent feedback on directions.

Indicative effect to plan for (observational, from the supplied logs, **not** a pilot result): the not-traceable rate rises from about 11-12% when the old pin was within 200 m of the borrower to 41% when it was more than 800 m away, and minutes per visit rise from about 9 to 28. Mapping our pin errors on addresses without a visit onto that gradient suggests roughly **-17% relative not-traceable rate** (95% interval -21% to -12%), **-11% minutes per visit** and **+10% productive visits per agent-day** (+6% to +14%). The pilot is how we find out whether that holds.

## 2. Design

**Cluster-randomised by territory** (locality / beat), 1:1, stratified by town and by baseline not-traceable rate. Randomising addresses would contaminate arms (the same agent walks both); randomising agents would not control for territory difficulty.

| | Control | Treatment |
|---|---|---|
| What the field app shows | old geocoder pin | our pin + 50/90% circles + tier + landmark directions (offline) |
| What the planner does | current rules | `VISIT_NOW` / `VISIT_WITH_DIRECTIONS` / `VERIFY_FIRST` by tier; 5% exploration share of low-confidence addresses visited anyway |
| Everything else (scripts, incentives, schedules) | unchanged | unchanged |

**Alternative if territories are few:** agent-level *switchback*: each agent alternates control / treatment weeks in random order; each agent is their own control. Needs far fewer agents for the productivity outcome (see power below).

**Duration:** 6 weeks after a 2-week baseline (data already logged) and a 1-week agent briefing. **Blinding:** agents cannot be blinded; analysts are, using coded arms.

## 3. Sample size (from the supplied logs)

Baseline: 25.1% of visits end *address not traceable*; between-territory ICC 0.067; about 66 visits per territory in 6 weeks (design effect 5.4); 3.27 productive visits per agent-day (ICC by agent 0.021; 77 agent-days per agent in the logs). 80% power, two-sided alpha 0.05:

| Outcome | Effect to detect | Visits (unclustered) per arm | Clusters per arm needed |
|---|---|---:|---:|
| Not-traceable share | -10% relative | 4,521 | 368 territories |
| | -15% relative | 1,970 | 160 territories |
| | -20% relative | 1,085 | 88 territories |
| | -30% relative | 461 | 38 territories |
| Productive visits / agent-day | +5% | 1,309 agent-days | 45 agents |
| | **+10%** | 327 agent-days | **11 agents** |
| | +15% | 145 agent-days | 5 agents |

Reading it: the supplied sample (36 localities, 9 field agents) is too small to *prove* a not-traceable effect; a lender-scale pilot (hundreds of territories) can detect -15%. The productivity outcome is detectable with about 22 field agents in a parallel design and with far fewer in a switchback. We therefore propose: **primary = productive visits per agent-day, co-primary = not-traceable share at lender scale.**

## 4. Analysis (pre-registered)

* Intention-to-treat; cluster-robust standard errors (or a mixed model with territory random effect), covariate-adjusted for baseline rate and town.
* Report effect sizes with 95% intervals, not just p-values; no peeking before week 6.
* Pre-specified subgroups: tier (high / medium / low), old-geocoder precision tier, town, visited vs unvisited address.
* Pin accuracy is monitored throughout without surveys: every new confirmed visit gives the error of the previous pin (self-auditing), and calibration coverage of the 90% radius is tracked weekly.

## 5. Go / no-go criteria

* **Go to scale-up:** primary effect in the predicted direction with the 95% interval excluding zero; no guardrail breach; 90% radius coverage between 85% and 97% on newly confirmed visits.
* **Pause and investigate:** any third-party disclosure incident; coverage below 80%; agents report directions unusable.
* **Stop:** treatment territories show a materially worse recovery per productive visit.

## 6. Rollout after the pilot

1. Expand to all territories in the pilot towns; keep a 5% holdout for ongoing measurement.
2. Turn on write-back to address records and the PS2 feed.
3. Monthly recalibration; quarterly review of the exploration share.
4. Extend to new towns with a cold-start check (landmark + old pin only), then learn as visits arrive.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Contamination (agents share tips across arms) | Randomise by territory; separate agent pools where possible |
| Agents ignore the tier / directions | Briefing; log whether directions were opened; ITT analysis still valid |
| Low-confidence addresses delay visits | Verification task is cheap (call / local check); exploration share keeps some visits moving |
| Synthetic-to-real gap | Calibrate on pilot data in week 1-2; widen radii if coverage is low |
| Wrong coordinate orientation | Confirm CRS before launch (see `design_doc.md`) |
