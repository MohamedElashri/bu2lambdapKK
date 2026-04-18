"""
Generate a slide-ready figure: FOM1 vs PID product cut for LL and DD.
Panels are stacked vertically (2 rows x 1 column).

FOM definitions (matching fit_based_scan.py):
  FOM1 = (N_jpsi + N_etac) / sqrt(N_jpsi + N_etac + N_bkg)
  FOM2 = (N_chic0 + N_chic1) / sqrt(N_chic0 + N_chic1 + N_bkg)

Legend is placed to the right of the figure to avoid blocking panel titles.

Output: figs/pid_product_fom_scan.pdf
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from _paths import SLIDES_DIR, resolve_pid_study_dir

STUDY_DIR = resolve_pid_study_dir()
FIT_DIR = STUDY_DIR / "output" / "fit_based"
FIGS_DIR = SLIDES_DIR / "figs"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)

fig, axes = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
# Leave room on the right for the legend
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.12, right=0.70, hspace=0.18)

PLATEAU_END = 0.20
line_handles = []

for i, (ax, cat) in enumerate(zip(axes, ["LL", "DD"])):
    with open(FIT_DIR / f"fit_scan_results_{cat}.json") as f:
        data = json.load(f)

    pp = data["pid_product"]
    cuts = np.array(pp["cuts"])
    nj = np.array(pp["n_jpsi"])
    ne = np.array(pp["n_etac"])
    nc0 = np.array(pp["n_chic0"])
    nc1 = np.array(pp["n_chic1"])
    nb = np.maximum(np.array(pp["n_bkg"]), 1.0)

    # FOM1 = (N_jpsi + N_etac) / sqrt(N_jpsi + N_etac + N_bkg)
    s1 = nj + ne
    fom1 = np.where(s1 + nb > 0, s1 / np.sqrt(s1 + nb), 0.0)

    # FOM2 = (N_chic0 + N_chic1) / sqrt(N_chic0 + N_chic1 + N_bkg)
    s2 = nc0 + nc1
    fom2 = np.where(s2 + nb > 0, s2 / np.sqrt(s2 + nb), 0.0)

    opt_idx = int(np.argmax(fom1))
    opt_cut = float(cuts[opt_idx])
    opt_fom = float(fom1[opt_idx])

    y_min = min(
        fom1[fom1 > 0].min() if any(fom1 > 0) else 0, fom2[fom2 > 0].min() if any(fom2 > 0) else 0
    )
    y_max = max(fom1.max(), fom2.max())
    y_pad = 0.08 * (y_max - y_min)
    y_lo = max(0, y_min - y_pad)
    y_hi = y_max + y_pad

    # Flat plateau shading
    ax.axvspan(-0.02, PLATEAU_END + 0.005, color="#dddddd", alpha=0.55, zorder=0)
    ax.text(
        PLATEAU_END / 2,
        y_lo + 0.04 * (y_hi - y_lo),
        "flat plateau",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    # FOM curves
    (l1,) = ax.plot(cuts, fom1, "o-", color="#1f77b4", lw=2, ms=7, zorder=3)
    (l2,) = ax.plot(cuts, fom2, "s--", color="#d62728", lw=1.6, ms=6, zorder=3)

    # Optimal cut line
    ax.axvline(opt_cut, color="#1f77b4", lw=1.5, ls=":", alpha=0.85, zorder=2)

    # Annotation: text in lower portion, arrow points up
    gain_pct = (opt_fom - fom1[0]) / fom1[0] * 100.0 if fom1[0] > 0 else 0.0
    y_text = y_lo + 0.18 * (y_hi - y_lo)
    ax.annotate(
        f"cut $>{opt_cut:.2f}$\nFOM1$={opt_fom:.2f}$  (+{gain_pct:.1f}\\%)",
        xy=(opt_cut, opt_fom),
        xytext=(opt_cut + 0.07, y_text),
        fontsize=8.5,
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#888888", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        zorder=6,
    )

    ax.set_ylabel("Figure of Merit", fontsize=10)
    ax.set_title(rf"$\Lambda$ {cat}", fontsize=11, pad=3)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(y_lo, y_hi)

    if i == 0:
        h3 = mlines.Line2D(
            [], [], color="#1f77b4", lw=1.5, ls=":", label=f"FOM1 optimal: cut $>{opt_cut:.2f}$"
        )
        line_handles = [l1, l2, h3]

axes[-1].set_xlabel(r"PID product cut ($p \times h_1 \times h_2 >$ value)", fontsize=10)

# Legend labels with correct S/sqrt(S+B) form
line_handles[0].set_label(
    r"FOM1 $=\dfrac{N_{J/\psi}+N_{\eta_c}}{\sqrt{N_{J/\psi}+N_{\eta_c}+N_{\rm bkg}}}$"
)
line_handles[1].set_label(
    r"FOM2 $=\dfrac{N_{\chi_{c0}}+N_{\chi_{c1}}}{\sqrt{N_{\chi_{c0}}+N_{\chi_{c1}}+N_{\rm bkg}}}$"
)

# Legend to the right, outside the axes — clear of both panel titles
fig.legend(
    handles=line_handles,
    loc="center left",
    bbox_to_anchor=(0.71, 0.50),
    ncol=1,
    fontsize=8.5,
    framealpha=0.95,
    edgecolor="#aaaaaa",
    handlelength=2.2,
    borderpad=0.7,
    labelspacing=1.2,
)

out = FIGS_DIR / "pid_product_fom_scan.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
