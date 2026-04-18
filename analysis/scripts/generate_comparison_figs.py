"""
Generate PID comparison figures

Reads JSON output from the pid_optimization_study and produces slide-ready PDFs:

  figs/eff_rejection_LL.pdf   — 2×2 panel: ε_sig & ε_bkg vs cut, all 4 PID vars, LL
  figs/eff_rejection_DD.pdf   — same for DD
  figs/roc_like_LL.pdf        — ε_sig vs background-rejection, all vars on one canvas (LL)
  figs/roc_like_DD.pdf        — same for DD
  figs/fit_fom_LL.pdf         — fit-based FOM1 vs cut, all vars on one canvas (LL)
  figs/fit_fom_DD.pdf         — same for DD
  figs/eff_summary_table.pdf  — horizontal bar chart of ε_sig at fit-optimal cut

All figures are produced in LHCb-publication style (black frame, no gridlines,
matching fonts with the beamer template).

Run using:
    uv run python scripts/generate_comparison_figs.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from _paths import SLIDES_DIR, resolve_pid_study_dir

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STUDY_DIR = resolve_pid_study_dir()
PROXY_DIR = STUDY_DIR / "output" / "box_proxy"
FIT_DIR = STUDY_DIR / "output" / "fit_based"
FIGS_DIR = SLIDES_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
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

VAR_META = {
    "pid_product": {"label": r"PID product ($p \times h_1 \times h_2$)", "color": "black"},
    "p_probnnp": {"label": r"$p$ ProbNNp", "color": "#1f77b4"},
    "h1_probnnk": {"label": r"$h_1$ ProbNNk ($K^+$)", "color": "#d62728"},
    "h2_probnnk": {"label": r"$h_2$ ProbNNk ($K^-$)", "color": "#2ca02c"},
}

OPT_COLORS = {"FOM1": "#e377c2", "FOM2": "#bcbd22"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_proxy(cat: str) -> dict:
    p = PROXY_DIR / f"proxy_scan_results_{cat}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing proxy scan results: {p}")
    with open(p) as f:
        return json.load(f)


def load_fit(cat: str) -> dict:
    p = FIT_DIR / f"fit_scan_results_{cat}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing fit scan results: {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 & 2: efficiency/rejection panels (2×2, one per variable)
# ---------------------------------------------------------------------------


def make_eff_rejection_panel(cat: str) -> None:
    proxy = load_proxy(cat)
    fit = load_fit(cat)

    vars_to_show = ["pid_product", "p_probnnp", "h1_probnnk", "h2_probnnk"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        rf"PID Signal Efficiency and Background Rejection  [$\Lambda$ {cat}]",
        fontsize=12,
        y=1.01,
    )

    for ax, vname in zip(axes.flat, vars_to_show):
        meta = VAR_META[vname]
        pr = proxy.get(vname, {})
        fi = fit.get(vname, {})

        if not pr.get("cuts"):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        cuts = np.array(pr["cuts"])
        eps_sig = np.array(pr["eps_sig"])
        eps_bkg = np.array(pr["eps_bkg"])
        bkg_rej = 1.0 - eps_bkg

        ax.plot(cuts, eps_sig, "b-o", ms=4, lw=1.5, label=r"$\varepsilon_{\rm sig}$ (MC)")
        ax.plot(cuts, bkg_rej, "r--s", ms=4, lw=1.5, label=r"Bkg rejection (sideband)")

        # Mark fit-based optimal cuts
        if fi.get("cuts"):
            for fom_key, fom_attr, oc in [
                ("FOM1", "best_cut_fom1", OPT_COLORS["FOM1"]),
                ("FOM2", "best_cut_fom2", OPT_COLORS["FOM2"]),
            ]:
                opt_cut = fi.get(fom_attr)
                if opt_cut is not None:
                    ax.axvline(
                        opt_cut,
                        color=oc,
                        lw=1.2,
                        ls=":",
                        label=f"Fit opt ({fom_key}) = {opt_cut:.2f}",
                    )

        ax.set_xlabel(f"Cut on {meta['label']} ($>$ value)", fontsize=9)
        ax.set_ylabel("Efficiency / Rejection", fontsize=9)
        ax.set_title(meta["label"], fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out = FIGS_DIR / f"eff_rejection_{cat}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 & 4: ROC-like curves (ε_sig vs bkg rejection)
# ---------------------------------------------------------------------------


def make_roc_like(cat: str) -> None:
    proxy = load_proxy(cat)
    fit = load_fit(cat)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_title(
        rf"Signal Efficiency vs Background Rejection  [$\Lambda$ {cat}]"
        "\n(with fit-based optimal working points marked)",
        fontsize=10,
    )

    for vname, meta in VAR_META.items():
        pr = proxy.get(vname, {})
        fi = fit.get(vname, {})
        if not pr.get("cuts"):
            continue

        eps_sig = np.array(pr["eps_sig"])
        eps_bkg = np.array(pr["eps_bkg"])
        bkg_rej = 1.0 - eps_bkg

        ax.plot(eps_sig, bkg_rej, color=meta["color"], lw=1.8, label=meta["label"])

        # Mark fit-based optimal FOM1 working point
        if fi.get("cuts"):
            cuts = np.array(pr["cuts"])
            opt_cut = fi.get("best_cut_fom1")
            if opt_cut is not None:
                idx = int(np.argmin(np.abs(cuts - opt_cut)))
                ax.plot(
                    eps_sig[idx],
                    bkg_rej[idx],
                    "o",
                    color=meta["color"],
                    ms=10,
                    mec="black",
                    mew=1.2,
                    zorder=5,
                )

    ax.set_xlabel(r"Signal efficiency $\varepsilon_{\rm sig}$ (from MC)", fontsize=11)
    ax.set_ylabel(r"Background rejection $1-\varepsilon_{\rm bkg}$ (from sideband)", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)

    # Add legend entry for working point markers
    ax.plot(
        [], [], "o", color="gray", ms=8, mec="black", mew=1.2, label="Fit-based FOM1 optimal cut"
    )

    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax.set_aspect("equal")

    plt.tight_layout()
    out = FIGS_DIR / f"roc_like_{cat}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 & 6: fit-based FOM vs cut (all variables on one canvas)
# ---------------------------------------------------------------------------


def make_fit_fom_overlay(cat: str) -> None:
    fit = load_fit(cat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        rf"Fit-Based FOM vs PID Cut  [$\Lambda$ {cat}]"
        "\n(each curve: apply PID cut to data, run mass fit, extract yields)",
        fontsize=10,
        y=1.03,
    )

    for ax, fom_key, ylabel, ytitle in [
        (
            axes[0],
            "fom1",
            r"$(N_{J/\psi}+N_{\eta_c})/\sqrt{N_{\rm bkg}}$",
            "FOM 1 — high-yield group",
        ),
        (axes[1], "fom2", r"$(N_{\chi_{c0}}+N_{\chi_{c1}})/\sqrt{S+B}$", "FOM 2 — low-yield group"),
    ]:
        for vname, meta in VAR_META.items():
            fi = fit.get(vname, {})
            if not fi.get("cuts"):
                continue
            cuts = np.array(fi["cuts"])
            fom = np.array(fi[fom_key])
            # Normalise to value at cut=0 (or first point) to allow comparison
            f0 = fom[0] if fom[0] > 0 else 1.0
            ax.plot(cuts, fom / f0, color=meta["color"], lw=1.8, label=meta["label"])
            # Mark optimal
            best_cut = fi.get(f"best_cut_{fom_key}")
            if best_cut is not None:
                ax.axvline(best_cut, color=meta["color"], lw=0.8, ls="--", alpha=0.7)

        ax.axhline(1.0, color="gray", lw=0.8, ls=":", label="Baseline (no cut)")
        ax.set_xlabel("PID cut value (> x)", fontsize=11)
        ax.set_ylabel("FOM / FOM(0)", fontsize=11)
        ax.set_title(ytitle, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(left=0)

    plt.tight_layout()
    out = FIGS_DIR / f"fit_fom_{cat}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 7: summary bar chart of ε_sig at fit-optimal cut
# ---------------------------------------------------------------------------


def make_eff_summary_table() -> None:
    """
    Horizontal bar chart: signal efficiency at the fit-based optimal FOM1 cut,
    for each PID variable and each category.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        r"Signal Efficiency at Fit-Based Optimal PID Cut (FOM1)"
        "\n(from MC; lower = tighter cut, more signal cost)",
        fontsize=10,
        y=1.03,
    )

    for ax, cat in zip(axes, ["LL", "DD"]):
        try:
            proxy = load_proxy(cat)
            fit = load_fit(cat)
        except FileNotFoundError:
            ax.text(
                0.5, 0.5, f"No data for {cat}", ha="center", va="center", transform=ax.transAxes
            )
            continue

        labels, effs, opt_cuts = [], [], []
        for vname, meta in VAR_META.items():
            pr = proxy.get(vname, {})
            fi = fit.get(vname, {})
            if not pr.get("cuts") or not fi.get("cuts"):
                continue
            cuts = np.array(pr["cuts"])
            eps_sig = np.array(pr["eps_sig"])
            opt_cut = fi.get("best_cut_fom1", 0.0)
            idx = int(np.argmin(np.abs(cuts - opt_cut)))
            labels.append(meta["label"])
            effs.append(float(eps_sig[idx]))
            opt_cuts.append(opt_cut)

        y_pos = np.arange(len(labels))
        colors = [list(VAR_META.values())[i]["color"] for i in range(len(labels))]
        bars = ax.barh(y_pos, effs, color=colors, edgecolor="black", lw=0.8, height=0.6)

        for bar, eff, oc in zip(bars, effs, opt_cuts):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{eff:.2f}  (cut>{oc:.2f})",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 1.35)
        ax.set_xlabel(r"$\varepsilon_{\rm sig}$ at optimal cut", fontsize=10)
        ax.set_title(rf"$\Lambda$ {cat}", fontsize=11)
        ax.axvline(1.0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    out = FIGS_DIR / "eff_summary_table.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating PID comparison figures for 20260411 slides...")
    for cat in ["LL", "DD"]:
        print(f"\n  Category: {cat}")
        try:
            make_eff_rejection_panel(cat)
            make_roc_like(cat)
            make_fit_fom_overlay(cat)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    make_eff_summary_table()
    print(f"\nAll figures written to {FIGS_DIR}")


if __name__ == "__main__":
    main()
