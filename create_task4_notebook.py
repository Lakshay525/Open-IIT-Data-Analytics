"""Build and execute notebooks/calibration.ipynb (Task 4) from the saved results. Run after confidence_directions.py."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Task 4 - Calibration, confidence tiers and directions

**Method.** The pin model reports `raw_spread` (its own sigma). For each group of addresses we collect the normalised score
`error / raw_spread` on data the model did not see and set `radius_p = Q(p) x raw_spread` (split conformal prediction with the
finite-sample correction). Three groups behave differently and are calibrated separately:

| group | meaning | calibration data | coverage is checked on |
|---|---|---|---|
| `visit` | address has its own cleaned visit evidence | surveyed out-of-fold errors, leave-one-out | the same rows, leave-one-out |
| `novisit` | locality known, no visit of its own | 1,231-address hide-and-predict set (visit pseudo-labels) | the 51 surveyed unvisited addresses |
| `noloc` | no locality in the text | hide-and-predict, unparsed subset | 6 surveyed addresses (too few to conclude) |

So for the groups that matter most, **the data used to set the radii and the data used to report coverage are different**.
""")
code("""
import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 160)
R = Path('results')
""")
md("## 1. Does a '90%' radius contain the truth about 90% of the time?")
code("""
cov = pd.read_csv(R/'calibration_coverage.csv')
show = cov[cov.nominal.isin([0.5, 0.9])].copy()
display(show[['dimension','segment','n','nominal','coverage','ci95_low','ci95_high','median_radius_90','median_error']].round(3))
""")
code("""
rel = pd.read_csv(R/'calibration_reliability.csv')
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0,1],[0,1],'k--',lw=1,label='perfect')
for name, g in rel.groupby('curve'):
    ax.plot(g.nominal, g.observed, marker='o', ms=4, label=f'{name} (n={int(g.n.iloc[0])})')
ax.set_xlabel('nominal coverage'); ax.set_ylabel('observed coverage'); ax.grid(alpha=.3); ax.legend(fontsize=7, loc='upper left')
ax.set_title('Reliability of the confidence radii'); plt.tight_layout(); plt.savefig(R/'calibration_reliability.png', dpi=140); plt.show()
""")
md("Points above the diagonal are conservative (radius bigger than needed); below would be over-confident. Small groups wobble because of sample size, not necessarily miscalibration - the intervals above say how much.")
md("## 2. Calibration by town and by the old geocoder's precision tier (90% radius)")
code("""
sub = cov[(cov.nominal==0.9) & cov.dimension.isin(['town','old_geocoder_tier'])]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
y = np.arange(len(sub))[::-1]
ax.errorbar(sub.coverage, y, xerr=[sub.coverage-sub.ci95_low, sub.ci95_high-sub.coverage], fmt='o', color='#3b6ea5', capsize=3)
ax.axvline(0.9, color='k', ls='--', lw=1); ax.set_yticks(y); ax.set_yticklabels([f'{d}: {s} (n={n})' for d,s,n in zip(sub.dimension, sub.segment, sub.n)], fontsize=8)
ax.set_xlabel('observed coverage of the 90% radius (95% Wilson interval)'); ax.set_xlim(0.2,1.02); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(R/'calibration_by_segment.png', dpi=140); plt.show()
display(sub[['dimension','segment','n','coverage','ci95_low','ci95_high']].round(3))
""")
md("""
Intervals for small strata (rooftop n=1, pincode n=10, the `noloc` group n=6) are wide and are **not** evidence either way.
Pincode-tier addresses with no locality in the text are the hard case; the model's radius for them is already in the kilometre range and their tier is always `low`.
""")
md("## 3. Confidence tiers: do they separate easy from hard addresses?")
code("""
tq = pd.read_csv(R/'confidence_tier_quality.csv'); display(tq.round(2))
pred = pd.read_csv('predictions.csv')
display(pred.groupby('confidence_tier').agg(addresses=('address_id','size'), median_radius_90=('radius_90','median'), median_location_confidence=('location_confidence','median')).round(2))
display(pred.groupby(['confidence_tier','suggested_action']).size().rename('addresses').to_frame())
""")
md("""
Rule: **high** = 90% radius <= 60 m (visit now); **medium** = <= 300 m (visit with directions); **low** = above that (verify first, never mark 'not traceable' on this basis).
On the surveyed addresses the tiers separate cleanly (accuracy columns above).
""")
md("## 4. What drives the radius")
code("""
fig, ax = plt.subplots(figsize=(6.5, 3.4))
order = pred.groupby('method_used').radius_90.median().sort_values().index
data = [np.log10(pred[pred.method_used==m].radius_90) for m in order]
ax.boxplot(data, vert=False, showfliers=False); ax.set_yticklabels([f'{m} (n={int((pred.method_used==m).sum())})' for m in order])
ax.set_xlabel('log10 of 90% radius (m)'); plt.tight_layout(); plt.savefig(R/'radius_by_method.png', dpi=140); plt.show()
""")
md("## 5. Directions")
code("""
chk = pd.read_csv(R/'directions_checks.csv'); display(chk.round(3).T)
display(pred.sample(10, random_state=4)[['address_id','confidence_tier','radius_90','directions']])
""")
md("""
Directions are plain-text templates built from the nearest landmarks to the pin (distance rounded, 8-point compass), with the address's own landmark first when one of that type is nearby, and the locality named when the same landmark name exists nearby. They need no network.
`test_task4.py` checks that 597 of 598 sampled direction clauses reproduce the distance and bearing from the stated landmark to the pin.
**Assumption to confirm with CN:** local x points east and y north (the source gives no CRS); flip `AXES` in `confidence_directions.py` if not.

## 6. Hand-off to Problem Statement 2
""")
code("""
ps2 = pd.read_csv('ps2_location_confidence.csv'); display(ps2.head(6)); display(ps2.ps2_guidance.value_counts().to_frame())
""")
nb["cells"] = C
out = ROOT / "notebooks" / "calibration.ipynb"
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, out)
print("wrote", out)
