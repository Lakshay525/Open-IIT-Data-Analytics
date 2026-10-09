"""Regenerate the two reproducible Task 1/2 notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


BASELINE = notebook([
    md("""# Task 1: baseline geocoder and evaluation protocol

The source data is synthetic. The full 100-address baseline is descriptive. The 15 surveyed addresses linked to the official account test split have fold 0 and must not guide later model selection. The previous review exposed some test labels, so this is not a pristine blind holdout for the team.
"""),
    code("""from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / 'eval.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from eval import baseline_inputs, score_predictions, make_fixed_folds

data = ROOT / 'clean_data'
out_dir = ROOT / 'results'
out_dir.mkdir(exist_ok=True)
predictions, truth, metadata = baseline_inputs(data)
scored, summary = score_predictions(predictions, truth, metadata)
localities = pd.read_csv(data / 'localities.csv')
folds = make_fixed_folds(truth, metadata, localities)
summary.to_csv(out_dir / 'baseline_scores.csv', index=False)
scored.to_csv(out_dir / 'baseline_address_errors.csv', index=False)
folds.to_csv(ROOT / 'folds.csv', index=False)
print(summary[['dimension','segment','n','median_error_m','p90_error_m',
               'share_within_50m','share_within_100m','share_within_250m']].to_string(index=False))
"""),
    code("""def median_plot(dimension, title, filename):
    part = summary[summary.dimension == dimension].sort_values('median_error_m')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(part))
    ax.bar(x, part.median_error_m, color='#2563eb')
    for i, row in enumerate(part.itertuples()):
        if pd.notna(row.median_ci95_low_m):
            ax.errorbar(i, row.median_error_m,
                        yerr=[[row.median_error_m-row.median_ci95_low_m],
                              [row.median_ci95_high_m-row.median_error_m]],
                        fmt='none', ecolor='#111827', capsize=3)
    ax.set_xticks(x, [f'{r.segment} (n={r.n})' for r in part.itertuples()],
                  rotation=20, ha='right')
    ax.set(ylabel='Median error (m)', title=title)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()

median_plot('precision', 'Baseline error by address precision (95% interval)',
            'baseline_by_precision.png')
median_plot('town_id', 'Baseline error by town (95% interval)',
            'baseline_by_town.png')
"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].hist(scored.error_m, bins=18, color='#0f766e', edgecolor='white')
axes[0].axvline(scored.error_m.median(), color='#b91c1c', linestyle='--',
                label=f'Median {scored.error_m.median():.1f} m')
axes[0].set(xlabel='Error (m)', ylabel='Surveyed addresses', title='Baseline errors')
axes[0].legend()
overall = summary.query("dimension == 'overall'").iloc[0]
axes[1].bar(['50 m','100 m','250 m'],
            [overall.share_within_50m, overall.share_within_100m,
             overall.share_within_250m], color=['#0f766e','#2563eb','#7c3aed'])
axes[1].set(ylim=(0, 1), ylabel='Share of surveyed addresses',
            title='Baseline pins within a radius')
for ax in axes:
    ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(out_dir / 'baseline_error_distribution.png', dpi=160)
plt.show()
"""),
    code("""print('Official holdout and grouped development folds:')
print(folds.groupby(['role','fold']).size().to_string())
print('A proxy locality never crosses two development folds:',
      (folds[folds.fold > 0].groupby('fold_group').fold.nunique() == 1).all())
print('Visited and unvisited baseline:')
print(summary[summary.dimension == 'visit_status'][
    ['segment','n','median_error_m','p90_error_m']].to_string(index=False))
"""),
    md("""## Reading the result

Median error is 376.4 m and P90 is 839.2 m on all 100 surveyed addresses. Only 5%, 9% and 35% lie within 50, 100 and 250 m. Visit status is a descriptive grouping, not an estimate of visit benefit. Confidence intervals resample addresses; intervals for fewer than 10 rows are suppressed. The proxy fold locality comes from the baseline pin's nearest named locality centroid, so spatial leakage may remain across neighboring locality boundaries.
"""),
])


STRESS = notebook([
    md("""# Task 2: visit quality and integrity review

All rules operate on outcome, GPS features, remarks, photo reuse and other visits, without using surveyed coordinates. Surveyed locations are brought in only afterwards for diagnostics on development addresses. A visit location is not automatically the current borrower's residence.
"""),
    code("""from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / 'extract_evidence.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from extract_evidence import build_evidence
from integrity_analysis import survey_diagnostics, stress_test

out_dir = ROOT / 'results'
out_dir.mkdir(exist_ok=True)
evidence, details = build_evidence(ROOT / 'clean_data', return_details=True)
evidence.to_csv(ROOT / 'evidence.csv', index=False, float_format='%.9g')
pd.DataFrame([details]).to_csv(out_dir / 'uncertainty_fit_summary.csv', index=False)
dev, quality_by_group, pins, uncertainty_quality = survey_diagnostics(evidence)
quality_by_group.to_csv(out_dir / 'visit_quality_by_group.csv', index=False)
pins.to_csv(out_dir / 'visit_pin_diagnostics.csv', index=False)
uncertainty_quality.to_csv(out_dir / 'visit_uncertainty_diagnostics.csv', index=False)
print('Evidence roles:', evidence.evidence_role.value_counts().to_dict())
print('Fit:', details)
print(uncertainty_quality.to_string(index=False))
"""),
    code("""print('Endpoint diagnostics on development surveyed addresses:')
print(quality_by_group[quality_by_group.dimension.isin(
    ['evidence_role','photo_reused_cross_address'])].to_string(index=False))
for field in ['contact_only_error_m','extended_error_m']:
    part = pins[pins[field].notna()]
    print(field, 'n=', len(part), 'median evidence error=',
          round(part[field].median(),1), 'median baseline on same subset=',
          round(part.baseline_error_m.median(),1))
"""),
    code("""# Include downweighted-but-ineligible rows to display the photo warning.
eligible = dev[dev.weight > 0].copy()
fig, ax = plt.subplots(figsize=(8, 5))
for reused, group in eligible.groupby('photo_reused_cross_address'):
    ax.scatter(group.weight.clip(lower=1e-6), group.endpoint_error_m.clip(lower=1),
               s=25, alpha=.6, label='Reused photo' if reused else 'Other visits')
    ax.set(xscale='log', yscale='log', xlabel='Trust weight (log scale)',
       ylabel='Endpoint error against survey (m, log scale)',
       title='Visit trust versus error, including low-weight photo flags')
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(out_dir / 'weight_vs_error.png', dpi=160)
plt.show()
"""),
    md("""## Real photo reuse signal

The identical photo hash occurs on different address IDs in the source data. The observed error contrast is descriptive: it does not prove intent or the causal effect of the penalty. The small surveyed reused-photo subgroup must not be treated as a fraud-detection validation set.
"""),
    code("""stress, injection, _ = stress_test(ROOT / 'clean_data')
stress.to_csv(out_dir / 'integrity_stress_results.csv', index=False)
injection.to_csv(out_dir / 'synthetic_attack_rows.csv', index=False)
summary = stress.groupby(['scenario','method']).agg(
    affected_addresses=('address_id','nunique'),
    reported_pins=('pin_shift_m','count'),
    abstentions=('abstained','sum'),
    median_pin_shift_m=('pin_shift_m','median'),
    p90_pin_shift_m=('pin_shift_m', lambda s: s.quantile(.9)),
    median_error_after_m=('error_after_m','median')).reset_index()
summary.to_csv(out_dir / 'integrity_stress_summary.csv', index=False)
print(summary.to_string(index=False))
"""),
    code("""methods = ['median, high-trust injection',
           'median, recomputed safeguards',
           'independent-agent cluster, may abstain']
scenarios = ['home_checkin','tea_stall','rogue_agent']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(3); width = .25
for j, method in enumerate(methods):
    part = summary.set_index(['scenario','method'])
    vals = [part.loc[(s,method),'p90_pin_shift_m'] for s in scenarios]
    axes[0].bar(x + (j-1)*width, vals, width, label=method)
axes[0].set_xticks(x, scenarios)
axes[0].set(ylabel='P90 pin shift among reported pins (m)',
            title='Tail shift under synthetic visits')
axes[0].legend(fontsize=7)
mix = summary[summary.method == methods[2]].set_index('scenario').loc[scenarios]
axes[1].bar(scenarios, mix.reported_pins / mix.affected_addresses, color='#0f766e')
axes[1].set(ylim=(0,1), ylabel='Share of targets receiving a pin',
            title='Independent-agent consensus coverage')
for ax in axes: ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(out_dir / 'integrity_stress_pin_shift.png', dpi=160)
plt.show()
"""),
    md("""## Interpretation

The detector sees the injected visits as ordinary input and recalculates every safeguard; it does not use a clean pin to penalise them. The clean pin and survey are used after that to measure shift and error. The independent-agent cluster method sharply reduces observed tail shifts where it emits a pin, but it abstains when independent corroboration is insufficient. The synthetic attacks are sensitivity tests, not real fraud labels or a production performance claim.
"""),
])


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for name, content in (("baseline_report.ipynb", BASELINE),
                          ("integrity_stress_test.ipynb", STRESS)):
        path = NOTEBOOKS / name
        path.write_text(json.dumps(content, indent=1), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
