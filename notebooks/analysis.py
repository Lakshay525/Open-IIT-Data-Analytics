# %% [markdown]
# # Analysis notebook: address geocoder that learns from field visits
#
# Reads the files written by `python run_pipeline.py`; nothing is retrained here. Sections: 1 data and baseline, 2 visit evidence and
# the fake-visit stress test, 3 pin model, 4 calibration, tiers and directions.
#
# **Rules that keep the numbers honest:** parameters were fitted on leave-one-out visit pseudo-labels (never on surveyed truth); surveyed
# truth enters only as neighbour knowledge from folds that are not being scored; fold 0 (official test) was seen in early exploration, so
# treat it as a sanity check, not a blind result.

# %%
import os
from pathlib import Path
if Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 160)
R = Path('outputs/results')

# %% [markdown]
# ## 1. Data and the baseline to beat

# %%
base = pd.read_csv(R/'baseline_scores.csv')
display(base[['dimension','segment','n','median_error_m','median_ci95_low_m','median_ci95_high_m','p90_error_m','share_within_100m']].round(2))
display(Image(str(R/'baseline_by_precision.png'), width=520))

# %% [markdown]
# ## 2. Visit evidence and the fake-visit stress test
# Each visit becomes one trusted point with a role and an uncertainty. Synthetic home check-ins, tea-stall stops and a rogue agent were
# injected and every safeguard was recomputed from the contaminated data.

# %%
ev = pd.read_csv('outputs/evidence.csv')
display(ev.evidence_role.value_counts().rename('visits').to_frame().T)
display(pd.read_csv(R/'integrity_stress_summary.csv').round(2))
display(Image(str(R/'integrity_stress_pin_shift.png'), width=620))

# %% [markdown]
# ## 3. Pin model

# %% [markdown]
# ## 1. Address parsing coverage

# %%
f = pd.read_csv(R/'address_features.csv')
cov = f[['pincode_text','loc_id','cross','main','gali','block','road','ward','building','lm_type']].notna().mean().round(3)
display(cov.rename('share of addresses with the field').to_frame())
display(f.lm_type.value_counts().rename('landmark type').to_frame().T)

# %% [markdown]
# ## 2. What each layer adds (ablation ladder)

# %%
ab = pd.read_csv(R/'pin_model_ablation.csv')
display(ab.round(1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
y = np.arange(len(ab))[::-1]
ax[0].barh(y, ab.all100_median, color='#3b6ea5'); ax[0].set_yticks(y); ax[0].set_yticklabels(ab.variant, fontsize=8)
ax[0].set_xlabel('median error, 100 surveyed (m)'); ax[0].set_title('Median')
ax[1].barh(y, ab.all100_p90, color='#c26a3d'); ax[1].set_yticks([]); ax[1].set_xlabel('90th percentile error (m)'); ax[1].set_title('P90')
plt.tight_layout(); plt.savefig(R/'pin_model_ablation.png', dpi=130); plt.show()

# %% [markdown]
# ## 3. Accuracy on the 100 surveyed addresses (out-of-fold)
#
# `dev85` = development folds 1-5, `test15` = official-test fold 0. Surveyed addresses are about 49% visited / 51% unvisited, so the overall number mixes two very different populations - see section 4.

# %%
sc = pd.read_csv(R/'pin_model_scores.csv')
display(sc[['dimension','segment','n','median_error_m','median_ci95_low_m','median_ci95_high_m','p90_error_m','share_within_50m','share_within_100m','share_within_250m']].round(2))

# %%
oof = pd.read_csv(R/'pin_model_oof_errors.csv')
base = pd.read_csv('data/baseline_geocodes.csv').set_index('address_id'); sv = pd.read_csv('data/surveyed_addresses.csv').set_index('address_id')
old = np.hypot(base.geocoder_x - sv.surveyed_x, base.geocoder_y - sv.surveyed_y).loc[oof.address_id].values
fig, ax = plt.subplots(figsize=(6.2, 4))
for e, lab, c in [(old, 'old geocoder', '#999999'), (oof.error_m.values, 'pin model (out-of-fold)', '#3b6ea5')]:
    s = np.sort(e); ax.plot(s, np.arange(1, len(s)+1)/len(s), label=lab, color=c, lw=2)
ax.set_xscale('log'); ax.set_xlabel('distance error (m, log scale)'); ax.set_ylabel('share of addresses'); ax.grid(alpha=.3); ax.legend()
ax.set_title('Error distribution, 100 surveyed addresses'); plt.tight_layout(); plt.savefig(R/'pin_model_error_cdf.png', dpi=130); plt.show()

# %% [markdown]
# ## 4. The harder question: addresses with NO visit
#
# The hide-and-predict check hides each visited address's own visit evidence and asks the model to recover the visit pin from text, neighbours, landmarks and the old pin. It uses 1,231 addresses and **no surveyed truth**, so it is both large and independent of the 100-address survey. This is the realistic number for the ~57% of addresses that have no usable visit.

# %%
hp = pd.read_csv(R/'hide_and_predict.csv')
tab = pd.DataFrame({'old geocoder': hp.err_old.describe(percentiles=[.5,.9])[['50%','90%']], 'pin model': hp.err_model.describe(percentiles=[.5,.9])[['50%','90%']]}).T
tab['within 100 m'] = [(hp.err_old<=100).mean(), (hp.err_model<=100).mean()]; display(tab.round(2))
display(hp.groupby('precision')[['err_old','err_model']].median().round(0).join(hp.groupby('precision').size().rename('n')))
display(hp.groupby('method_used').agg(n=('err_model','size'), model_median=('err_model','median'), old_median=('err_old','median')).round(0))

# %% [markdown]
# ## 5. Is `raw_spread` honest? (input to Task 4's calibrated radii)

# %%
oofm = oof.merge(pd.read_csv('outputs/pins.csv')[['address_id','raw_spread','method_used']], on='address_id')
hp2 = hp.copy()
for name, e, s in [('surveyed OOF (n=%d)' % len(oofm), oofm.error_m, oofm.raw_spread), ('hide-and-predict (n=%d)' % len(hp2), hp2.err_model, hp2.raw_spread)]:
    r = e / s
    print(f'{name:30s} Spearman(raw_spread, error) = {pd.Series(s.values).corr(pd.Series(e.values), method="spearman"):.2f}; '
          f'share of errors within 1.18x / 2.15x raw_spread = {(r<=1.177).mean():.2f} / {(r<=2.146).mean():.2f} (Gaussian ideal 0.50 / 0.90)')
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.scatter(hp2.raw_spread, hp2.err_model, s=6, alpha=.35, color='#3b6ea5'); lim = [10, 4000]; ax.plot(lim, lim, 'k--', lw=1, label='error = spread')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('raw_spread (m)'); ax.set_ylabel('actual error (m)'); ax.legend(); ax.set_title('hide-and-predict'); plt.tight_layout(); plt.savefig(R/'pin_model_spread_vs_error.png', dpi=130); plt.show()

# %% [markdown]
# ## 6. Where it fails

# %%
ad = pd.read_csv('data/addresses.csv').set_index('address_id')
w = oofm.set_index('address_id').join(ad[['address_text']]).sort_values('error_m', ascending=False).head(10)
display(w[['precision','error_m','raw_spread','method_used','address_text']].round(0))

# %% [markdown]
# **Reading the failures.** The worst addresses are almost all *pincode-tier addresses whose text names no locality*. The pincode leaves 3-5 candidate localities about 1.4 km apart, nothing in the text separates them (street numbers repeat across localities; only T2 wards are locality-specific), and the old pin is already the pincode centroid. No text model can fix that; the model keeps close to the old pin there and reports a wide `raw_spread`, so the visit planner should send these addresses to *verify first*. One confirmed visit resolves them (the visit becomes the pin), which is the learning loop doing its job.
#
# ## 7. Limits to state honestly
# - 100 surveyed addresses is small; intervals are in `pin_model_scores.csv`. Strata below 10 are not interpretable.
# - The visit-derived labels used for tuning are not survey truth (they agree with it to a median of about 8 m where both exist).
# - Surveyed truth covers residences only; office/native-village pins are unvalidated.
# - The official-test fold was seen during exploratory analysis; freeze the method before claiming a blind result.

# %% [markdown]
# ## 4. Calibration, confidence tiers and directions

# %% [markdown]
# ## 1. Does a '90%' radius contain the truth about 90% of the time?

# %%
cov = pd.read_csv(R/'calibration_coverage.csv')
show = cov[cov.nominal.isin([0.5, 0.9])].copy()
display(show[['dimension','segment','n','nominal','coverage','ci95_low','ci95_high','median_radius_90','median_error']].round(3))

# %%
rel = pd.read_csv(R/'calibration_reliability.csv')
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0,1],[0,1],'k--',lw=1,label='perfect')
for name, g in rel.groupby('curve'):
    ax.plot(g.nominal, g.observed, marker='o', ms=4, label=f'{name} (n={int(g.n.iloc[0])})')
ax.set_xlabel('nominal coverage'); ax.set_ylabel('observed coverage'); ax.grid(alpha=.3); ax.legend(fontsize=7, loc='upper left')
ax.set_title('Reliability of the confidence radii'); plt.tight_layout(); plt.savefig(R/'calibration_reliability.png', dpi=140); plt.show()

# %% [markdown]
# Points above the diagonal are conservative (radius bigger than needed); below would be over-confident. Small groups wobble because of sample size, not necessarily miscalibration - the intervals above say how much.

# %% [markdown]
# ## 2. Calibration by town and by the old geocoder's precision tier (90% radius)

# %%
sub = cov[(cov.nominal==0.9) & cov.dimension.isin(['town','old_geocoder_tier'])]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
y = np.arange(len(sub))[::-1]
ax.errorbar(sub.coverage, y, xerr=[sub.coverage-sub.ci95_low, sub.ci95_high-sub.coverage], fmt='o', color='#3b6ea5', capsize=3)
ax.axvline(0.9, color='k', ls='--', lw=1); ax.set_yticks(y); ax.set_yticklabels([f'{d}: {s} (n={n})' for d,s,n in zip(sub.dimension, sub.segment, sub.n)], fontsize=8)
ax.set_xlabel('observed coverage of the 90% radius (95% Wilson interval)'); ax.set_xlim(0.2,1.02); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(R/'calibration_by_segment.png', dpi=140); plt.show()
display(sub[['dimension','segment','n','coverage','ci95_low','ci95_high']].round(3))

# %% [markdown]
# Intervals for small strata (rooftop n=1, pincode n=10, the `noloc` group n=6) are wide and are **not** evidence either way.
# Pincode-tier addresses with no locality in the text are the hard case; the model's radius for them is already in the kilometre range and their tier is always `low`.

# %% [markdown]
# ## 3. Confidence tiers: do they separate easy from hard addresses?

# %%
tq = pd.read_csv(R/'confidence_tier_quality.csv'); display(tq.round(2))
pred = pd.read_csv('outputs/predictions.csv')
display(pred.groupby('confidence_tier').agg(addresses=('address_id','size'), median_radius_90=('radius_90','median'), median_location_confidence=('location_confidence','median')).round(2))
display(pred.groupby(['confidence_tier','suggested_action']).size().rename('addresses').to_frame())

# %% [markdown]
# Rule: **high** = 90% radius <= 60 m (visit now); **medium** = <= 300 m (visit with directions); **low** = above that (verify first, never mark 'not traceable' on this basis).
# On the surveyed addresses the tiers separate cleanly (accuracy columns above).

# %% [markdown]
# ## 4. What drives the radius

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.4))
order = pred.groupby('method_used').radius_90.median().sort_values().index
data = [np.log10(pred[pred.method_used==m].radius_90) for m in order]
ax.boxplot(data, vert=False, showfliers=False); ax.set_yticklabels([f'{m} (n={int((pred.method_used==m).sum())})' for m in order])
ax.set_xlabel('log10 of 90% radius (m)'); plt.tight_layout(); plt.savefig(R/'radius_by_method.png', dpi=140); plt.show()

# %% [markdown]
# ## 5. Directions

# %%
chk = pd.read_csv(R/'directions_checks.csv'); display(chk.round(3).T)
display(pred.sample(10, random_state=4)[['address_id','confidence_tier','radius_90','directions']])

# %% [markdown]
# Directions are plain-text templates built from the nearest landmarks to the pin (distance rounded, 8-point compass), with the address's own landmark first when one of that type is nearby, and the locality named when the same landmark name exists nearby. They need no network.
# `test_task4.py` checks that 597 of 598 sampled direction clauses reproduce the distance and bearing from the stated landmark to the pin.
# **Assumption to confirm with CN:** local x points east and y north (the source gives no CRS); flip `AXES` in `confidence_directions.py` if not.
#
# ## 6. Hand-off to Problem Statement 2

# %%
ps2 = pd.read_csv('outputs/ps2_location_confidence.csv'); display(ps2.head(6)); display(ps2.ps2_guidance.value_counts().to_frame())
