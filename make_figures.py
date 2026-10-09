"""Figures: architecture diagram and a static example map. Run after confidence_directions.py."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

_mpl = TemporaryDirectory(prefix="mpl_")
os.environ.setdefault("MPLCONFIGDIR", _mpl.name)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TC = {"high": "#2f7d5b", "medium": "#d19a2a", "low": "#c0443a"}


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axis("off"); ax.set_xlim(0, 110); ax.set_ylim(0, 43)

    def box(x, y, w, h, text, fc, ec="#33415c", fs=8.3, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.2))

    box(1, 28, 22, 12, "Inputs from CN\naddress text · old pin\nvisit GPS trails · outcomes\nagent remarks · landmarks", "#eef3fb")
    box(28, 28, 24, 12, "Task 2  Visit cleaning\none trusted point per visit\nrole · uncertainty · integrity\n(fake-visit safeguards)", "#e8f4ee")
    box(28, 6, 24, 12, "Address parser\nlocality · street numbers\nlandmark type (EN/HI/KN)", "#e8f4ee")
    box(58, 17, 25, 14, "Task 3  Pin model\nevidence fusion: own visit ·\nstreet neighbours · street-number\ngrid · landmarks · old pin", "#fdf1dc", bold=False)
    box(88, 28, 21, 12, "Task 4  Calibration\nconformal 50/90% radii\nconfidence tier · directions", "#f6e7ea")
    box(88, 6, 21, 12, "Outputs\nfield app (offline) · visit planner\naddress records · PS2 · dashboards", "#eef3fb")
    arrow(23, 34, 28, 34); arrow(52, 34, 58, 27); arrow(40, 18, 58, 22); arrow(83, 26, 88, 32); arrow(98.5, 28, 98.5, 18)
    arrow(23, 31, 28, 14)
    ax.annotate("", xy=(70, 17), xytext=(98, 6.5), arrowprops=dict(arrowstyle="-|>", color="#c26a3d", lw=1.6, ls="--", connectionstyle="arc3,rad=-0.25"))
    ax.text(79, 3.0, "every confirmed visit becomes new evidence", color="#c26a3d", fontsize=8.5, ha="center", style="italic")
    plt.tight_layout(); plt.savefig(RES / "fig_architecture.png", dpi=160); plt.close()


def example_map():
    """Six typical surveyed addresses (the 40th and 60th error percentile inside each tier - not cherry-picked)."""
    p = pd.read_csv(ROOT / "predictions.csv").set_index("address_id")
    sv = pd.read_csv(ROOT / "clean_data" / "surveyed_addresses.csv").set_index("address_id")
    lm = pd.read_csv(ROOT / "clean_data" / "landmarks_poi.csv")
    d = pd.read_csv(RES / "calibration_surveyed_detail.csv").set_index("address_id")
    picks = []
    for tier in ("high", "medium", "low"):
        g = d[p.loc[d.index, "confidence_tier"] == tier].sort_values("error_m")
        for q in (0.4, 0.6):
            if len(g):
                picks.append(g.index[int(q * (len(g) - 1))])
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for ax, a in zip(axes.T.ravel(), picks):
        r = p.loc[a]; t = sv.loc[a].to_numpy(float)
        col = TC[r.confidence_tier]
        ext = max(1.25 * r.radius_90, 1.15 * np.hypot(r.old_pin_x - t[0], r.old_pin_y - t[1]), 1.15 * np.hypot(r.pin_x - t[0], r.pin_y - t[1]), 120)
        cx, cy = (r.pin_x + t[0]) / 2, (r.pin_y + t[1]) / 2
        ax.add_patch(Circle((r.pin_x, r.pin_y), r.radius_90, fc=col, alpha=.12, ec=col, lw=1.2))
        ax.add_patch(Circle((r.pin_x, r.pin_y), r.radius_50, fc=col, alpha=.22, ec=col, lw=1.2))
        ax.plot([r.old_pin_x, r.pin_x], [r.old_pin_y, r.pin_y], color="#8a8f98", lw=1, ls=":")
        ax.scatter(r.old_pin_x, r.old_pin_y, s=70, facecolor="white", edgecolor="#8a8f98", lw=2, zorder=3)
        ax.scatter(r.pin_x, r.pin_y, s=70, color=col, edgecolor="white", lw=1.2, zorder=4)
        ax.scatter(*t, s=80, marker="D", color="black", zorder=5)
        L = lm[(lm.town_id == r.town_id)]
        L = L[np.hypot(L.x - cx, L.y - cy) < ext * 1.5]
        ax.scatter(L.x, L.y, marker="s", s=28, color="#7a6bb0", zorder=2)
        for q in L.itertuples():
            ax.annotate(q.name, (q.x, q.y), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#5b4d8f")
        ax.set_xlim(cx - ext * 1.4, cx + ext * 1.4); ax.set_ylim(cy - ext * 1.4, cy + ext * 1.4); ax.set_aspect("equal")
        eo = np.hypot(r.old_pin_x - t[0], r.old_pin_y - t[1]); en = np.hypot(r.pin_x - t[0], r.pin_y - t[1])
        ax.set_title(f"{a} - {r.confidence_tier}\nold {eo:.0f} m  ->  ours {en:.0f} m  (90% radius {r.radius_90:.0f} m)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        sb = 10 ** np.floor(np.log10(ext * .8)); x0 = cx - ext * 1.25; y0 = cy - ext * 1.25
        ax.plot([x0, x0 + sb], [y0, y0], color="k", lw=2); ax.text(x0, y0 + ext * .05, f"{sb:.0f} m", fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#8a8f98", mew=2, label="old geocoder pin"),
                        Line2D([], [], marker="o", ls="", color="#555555", label="our pin (colour = confidence tier)"),
                        Line2D([], [], marker="D", ls="", color="black", label="surveyed true location"),
                        Line2D([], [], marker="s", ls="", color="#7a6bb0", label="landmark")], loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(RES / "fig_example_map.png", dpi=150); plt.close()


if __name__ == "__main__":
    architecture()
    example_map()
    print("wrote figures")
