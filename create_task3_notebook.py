"""Build and execute notebooks/pin_model_experiments.ipynb from the saved results.

Run after run_task3.py:  python create_task3_notebook.py
"""
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
# Task 3 - Pin model experiments

Everything here is read from files written by `run_task3.py`; nothing is retrained in the notebook.

**Rules that keep the numbers honest**
- Parameters were fitted on *leave-one-out visit pseudo-labels* (`tune_pin_model.py`), never on surveyed truth.
- Surveyed truth enters only as neighbour knowledge from folds that are not being scored (see `test_task3.py`).
- Fold 0 is the official test group. It was exposed earlier in exploratory work (see the Task 1-2 README), so treat it as a sanity check, not a blind test.
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
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
R = Path('results')
""")
md("## 1. Address parsing coverage")
code("""
f = pd.read_csv(R/'address_features.csv')
cov = f[['pincode_text','loc_id','cross','main','gali','block','road','ward','building','lm_type']].notna().mean().round(3)
display(cov.rename('share of addresses with the field').to_frame())
display(f.lm_type.value_counts().rename('landmark type').to_frame().T)
""")
md("## 2. What each layer adds (ablation ladder)")
code("""
ab = pd.read_csv(R/'pin_model_ablation.csv')
display(ab.round(1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
y = np.arange(len(ab))[::-1]
ax[0].barh(y, ab.all100_median, color='#3b6ea5'); ax[0].set_yticks(y); ax[0].set_yticklabels(ab.variant, fontsize=8)
ax[0].set_xlabel('median error, 100 surveyed (m)'); ax[0].set_title('Median')
ax[1].barh(y, ab.all100_p90, color='#c26a3d'); ax[1].set_yticks([]); ax[1].set_xlabel('90th percentile error (m)'); ax[1].set_title('P90')
plt.tight_layout(); plt.savefig(R/'pin_model_ablation.png', dpi=130); plt.show()
""")
md("""
## 3. Accuracy on the 100 surveyed addresses (out-of-fold)

`dev85` = development folds 1-5, `test15` = official-test fold 0. Surveyed addresses are about 49% visited / 51% unvisited, so the overall number mixes two very different populations - see section 4.
""")
code("""
sc = pd.read_csv(R/'pin_model_scores.csv')
display(sc[['dimension','segment','n','median_error_m','median_ci95_low_m','median_ci95_high_m','p90_error_m','share_within_50m','share_within_100m','share_within_250m']].round(2))
""")
code("""
oof = pd.read_csv(R/'pin_model_oof_errors.csv')
base = pd.read_csv('clean_data/baseline_geocodes.csv').set_index('address_id'); sv = pd.read_csv('clean_data/surveyed_addresses.csv').set_index('address_id')
old = np.hypot(base.geocoder_x - sv.surveyed_x, base.geocoder_y - sv.surveyed_y).loc[oof.address_id].values
fig, ax = plt.subplots(figsize=(6.2, 4))
for e, lab, c in [(old, 'old geocoder', '#999999'), (oof.error_m.values, 'pin model (out-of-fold)', '#3b6ea5')]:
    s = np.sort(e); ax.plot(s, np.arange(1, len(s)+1)/len(s), label=lab, color=c, lw=2)
ax.set_xscale('log'); ax.set_xlabel('distance error (m, log scale)'); ax.set_ylabel('share of addresses'); ax.grid(alpha=.3); ax.legend()
ax.set_title('Error distribution, 100 surveyed addresses'); plt.tight_layout(); plt.savefig(R/'pin_model_error_cdf.png', dpi=130); plt.show()
""")
md("""
## 4. The harder question: addresses with NO visit

The hide-and-predict check hides each visited address's own visit evidence and asks the model to recover the visit pin from text, neighbours, landmarks and the old pin. It uses 1,231 addresses and **no surveyed truth**, so it is both large and independent of the 100-address survey. This is the realistic number for the ~57% of addresses that have no usable visit.
""")
code("""
hp = pd.read_csv(R/'hide_and_predict.csv')
tab = pd.DataFrame({'old geocoder': hp.err_old.describe(percentiles=[.5,.9])[['50%','90%']], 'pin model': hp.err_model.describe(percentiles=[.5,.9])[['50%','90%']]}).T
tab['within 100 m'] = [(hp.err_old<=100).mean(), (hp.err_model<=100).mean()]; display(tab.round(2))
display(hp.groupby('precision')[['err_old','err_model']].median().round(0).join(hp.groupby('precision').size().rename('n')))
display(hp.groupby('method_used').agg(n=('err_model','size'), model_median=('err_model','median'), old_median=('err_old','median')).round(0))
""")
md("## 5. Is `raw_spread` honest? (input to Task 4's calibrated radii)")
code("""
oofm = oof.merge(pd.read_csv('pins.csv')[['address_id','raw_spread','method_used']], on='address_id')
hp2 = hp.copy()
for name, e, s in [('surveyed OOF (n=%d)' % len(oofm), oofm.error_m, oofm.raw_spread), ('hide-and-predict (n=%d)' % len(hp2), hp2.err_model, hp2.raw_spread)]:
    r = e / s
    print(f'{name:30s} Spearman(raw_spread, error) = {pd.Series(s.values).corr(pd.Series(e.values), method="spearman"):.2f}; '
          f'share of errors within 1.18x / 2.15x raw_spread = {(r<=1.177).mean():.2f} / {(r<=2.146).mean():.2f} (Gaussian ideal 0.50 / 0.90)')
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(hp2.raw_spread, hp2.err_model, s=6, alpha=.35, color='#3b6ea5'); lim = [10, 4000]; ax.plot(lim, lim, 'k--', lw=1, label='error = spread')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('raw_spread (m)'); ax.set_ylabel('actual error (m)'); ax.legend(); ax.set_title('hide-and-predict'); plt.tight_layout(); plt.savefig(R/'pin_model_spread_vs_error.png', dpi=130); plt.show()
""")
md("## 6. Where it fails")
code("""
ad = pd.read_csv('clean_data/addresses.csv').set_index('address_id')
w = oofm.set_index('address_id').join(ad[['address_text']]).sort_values('error_m', ascending=False).head(10)
display(w[['precision','error_m','raw_spread','method_used','address_text']].round(0))
""")
md("""
**Reading the failures.** The worst addresses are almost all *pincode-tier addresses whose text names no locality*. The pincode leaves 3-5 candidate localities about 1.4 km apart, nothing in the text separates them (street numbers repeat across localities; only T2 wards are locality-specific), and the old pin is already the pincode centroid. No text model can fix that; the model keeps close to the old pin there and reports a wide `raw_spread`, so the visit planner should send these addresses to *verify first*. One confirmed visit resolves them (the visit becomes the pin), which is the learning loop doing its job.

## 7. Limits to state honestly
- 100 surveyed addresses is small; intervals are in `pin_model_scores.csv`. Strata below 10 are not interpretable.
- The visit-derived labels used for tuning are not survey truth (they agree with it to a median of about 8 m where both exist).
- Surveyed truth covers residences only; office/native-village pins are unvalidated.
- The official-test fold was seen during exploratory analysis; freeze the method before claiming a blind result.
""")
nb["cells"] = C
out = ROOT / "notebooks" / "pin_model_experiments.ipynb"
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, out)
print("wrote", out)
