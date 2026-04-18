"""
Generate two slide-ready figures.

Figure 1 — pid_product_tradeoff.pdf
    2-panel (LL, DD): signal retained % and background removed % vs PID product cut.
    Focus on the one variable we actually use.  Annotates the working point 0.25 with
    exact numbers.  Shade the "net gain" region where more background is removed than
    signal is lost.

Figure 2 — variable_working_points.pdf
    Grouped bar chart: for each PID variable, signal retained (%) and background
    removed (%) at its fit-based FOM1 optimal cut.  LL and DD side by side.
    Gives the full picture without showing curves the reader has to trace.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import SLIDES_DIR, resolve_pid_study_dir

STUDY_DIR = resolve_pid_study_dir()
PROXY_DIR = STUDY_DIR / "output" / "box_proxy"
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
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)

VAR_META = {
    "pid_product": {"label": r"PID product ($p \times h_1 \times h_2$)", "color": "#333333"},
    "p_probnnp": {"label": r"$p$ ProbNNp", "color": "#1f77b4"},
    "h1_probnnk": {"label": r"$h_1$ ProbNNk ($K^+$)", "color": "#d62728"},
    "h2_probnnk": {"label": r"$h_2$ ProbNNk ($K^-$)", "color": "#2ca02c"},
}


def load(cat):
    with open(PROXY_DIR / f"proxy_scan_results_{cat}.json") as f:
        proxy = json.load(f)
    with open(FIT_DIR / f"fit_scan_results_{cat}.json") as f:
        fit = json.load(f)
    return proxy, fit


# ---------------------------------------------------------------------------
# Figure 1: PID product — signal kept vs background removed
# ---------------------------------------------------------------------------


def fig_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(
        r"PID product ($p\!\times\! h_1\!\times\! h_2$) cut — what fraction of events survive?",
        fontsize=11,
    )

    for ax, cat in zip(axes, ["LL", "DD"]):
        proxy, fit = load(cat)
        pp = proxy["pid_product"]
        fi = fit["pid_product"]
        if not pp.get("cuts"):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        cuts = np.array(pp["cuts"])
        sig_pct = np.array(pp["eps_sig"]) * 100.0  # % signal retained
        bkg_rem = (1.0 - np.array(pp["eps_bkg"])) * 100.0  # % background removed

        # Shade "net gain" region: where bkg_rem > (100 - sig_pct)
        # i.e. more background removed than signal lost
        sig_lost = 100.0 - sig_pct
        gain_mask = bkg_rem > sig_lost

        ax.fill_between(cuts, sig_pct, 100, alpha=0.10, color="#1f77b4", label="_nolegend_")
        ax.fill_between(cuts, 0, bkg_rem, alpha=0.10, color="#d62728", label="_nolegend_")

        ax.plot(cuts, sig_pct, "o-", color="#1f77b4", lw=2.2, ms=7, label="Signal retained (%)")
        ax.plot(cuts, bkg_rem, "s--", color="#d62728", lw=2.2, ms=7, label="Background removed (%)")

        # Working point at fit-optimal cut
        opt_cut = fi.get("best_cut_fom1", 0.25)
        idx = int(np.argmin(np.abs(cuts - opt_cut)))
        s_at = sig_pct[idx]
        b_at = bkg_rem[idx]

        ax.axvline(opt_cut, color="#555555", lw=1.4, ls=":")

        # Annotation box
        ax.annotate(
            f"Cut $> {opt_cut:.2f}$\n" f"Signal kept:  {s_at:.0f}%\n" f"Bkg removed: {b_at:.0f}%",
            xy=(opt_cut, (s_at + b_at) / 2),
            xytext=(opt_cut + 0.05, 40),
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="#888888", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
        )

        # Mark the working-point dots
        ax.plot(opt_cut, s_at, "o", color="#1f77b4", ms=11, zorder=5, mec="black", mew=1)
        ax.plot(opt_cut, b_at, "s", color="#d62728", ms=11, zorder=5, mec="black", mew=1)

        ax.set_xlabel(
            "PID product cut threshold (events with value $>$ threshold pass)", fontsize=10
        )
        ax.set_ylabel("Fraction of events surviving cut (%)", fontsize=10)
        ax.set_title(rf"$\Lambda$ {cat}", fontsize=11)
        ax.set_xlim(-0.01, 0.55)
        ax.set_ylim(-2, 105)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    out = FIGS_DIR / "pid_product_tradeoff.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: All variables — signal kept and background removed at optimal cut
# ---------------------------------------------------------------------------


def fig_working_points():
    """
    Horizontal grouped bar chart per variable per category.
    Two bars per variable: signal retained (blue) and background removed (red),
    evaluated at the fit-based FOM1 optimal cut for that variable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "At each variable's fit-based optimal cut: signal retained vs background removed",
        fontsize=11,
    )

    vars_ordered = list(VAR_META.keys())

    for ax, cat in zip(axes, ["LL", "DD"]):
        proxy, fit = load(cat)

        labels, sig_vals, bkg_vals, cut_vals = [], [], [], []
        for vname in vars_ordered:
            pr = proxy.get(vname, {})
            fi = fit.get(vname, {})
            if not pr.get("cuts") or not fi.get("cuts"):
                continue
            cuts = np.array(pr["cuts"])
            eps_sig = np.array(pr["eps_sig"])
            eps_bkg = np.array(pr["eps_bkg"])
            opt_cut = fi.get("best_cut_fom1", 0.0)
            idx = int(np.argmin(np.abs(cuts - opt_cut)))
            labels.append(VAR_META[vname]["label"])
            sig_vals.append(eps_sig[idx] * 100.0)
            bkg_vals.append((1.0 - eps_bkg[idx]) * 100.0)
            cut_vals.append(opt_cut)

        y = np.arange(len(labels))
        height = 0.35

        b_bars = ax.barh(
            y + height / 2,
            bkg_vals,
            height,
            color="#d62728",
            alpha=0.80,
            edgecolor="black",
            lw=0.6,
            label="Background removed",
        )
        s_bars = ax.barh(
            y - height / 2,
            sig_vals,
            height,
            color="#1f77b4",
            alpha=0.80,
            edgecolor="black",
            lw=0.6,
            label="Signal retained",
        )

        # Annotate with exact numbers and cut value
        for i, (s, b, c) in enumerate(zip(sig_vals, bkg_vals, cut_vals)):
            ax.text(s + 0.8, y[i] - height / 2, f"{s:.0f}%", va="center", fontsize=8)
            ax.text(b + 0.8, y[i] + height / 2, f"{b:.0f}%  (cut>{c:.2f})", va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 125)
        ax.set_xlabel("Fraction of events (%)", fontsize=10)
        ax.set_title(rf"$\Lambda$ {cat}", fontsize=11)
        ax.axvline(100, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    out = FIGS_DIR / "variable_working_points.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


if __name__ == "__main__":
    fig_tradeoff()
    fig_working_points()
    print("Done.")
