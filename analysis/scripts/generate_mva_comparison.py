"""
Generate MVA comparison figures.

Reads mva_pid_summary_{LL,DD}.json from pid_optimization_study and produces:

  figs/mva_auc_comparison.pdf  — grouped bar chart: AUC for 3 variants × 2 categories
  figs/mva_roc_LL.pdf          — ROC overlay (3 variants) for LL, slide-ready
  figs/mva_roc_DD.pdf          — same for DD

Run using:
    uv run python scripts/generate_mva_comparison.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import SLIDES_DIR, resolve_pid_study_dir

STUDY_DIR = resolve_pid_study_dir()
MVA_DIR = STUDY_DIR / "output" / "mva"
FIGS_DIR = SLIDES_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

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

VARIANT_META = {
    "A_baseline": {"label": "Baseline\n(no PID)", "color": "#333333", "ls": "-"},
    "B_individual": {"label": "Individual PID\n(p, h1, h2)", "color": "#1f77b4", "ls": "--"},
    "C_product": {"label": "PID product", "color": "#d62728", "ls": "-."},
}


def load_mva(cat: str) -> dict:
    p = MVA_DIR / f"mva_pid_summary_{cat}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing MVA summary: {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: AUC grouped bar chart
# ---------------------------------------------------------------------------


def make_auc_bar_chart() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    cats = ["LL", "DD"]
    variants = list(VARIANT_META.keys())
    n_vars = len(variants)
    x = np.arange(len(cats))
    width = 0.22

    for i, vid in enumerate(variants):
        meta = VARIANT_META[vid]
        aucs = []
        for cat in cats:
            try:
                data = load_mva(cat)
                aucs.append(data["roc"][vid]["auc"])
            except (FileNotFoundError, KeyError):
                aucs.append(0.0)
        offset = (i - (n_vars - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            aucs,
            width,
            label=meta["label"].replace("\n", " "),
            color=meta["color"],
            alpha=0.85,
            edgecolor="black",
            lw=0.8,
        )
        for bar, auc_val in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{auc_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    # Zoom y-axis to show differences clearly
    ax.set_ylim(0.88, 0.96)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\Lambda$ LL", r"$\Lambda$ DD"], fontsize=11)
    ax.set_ylabel("ROC AUC")
    ax.set_title(
        "MVA Classifier AUC — Baseline vs PID Variants\n"
        r"\small{(AUC difference driven by proxy bias, not analysis sensitivity)}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.axhline(0.91, color="gray", lw=0.6, ls=":", alpha=0.5)

    plt.tight_layout()
    out = FIGS_DIR / "mva_auc_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 & 3: ROC overlay per category
# ---------------------------------------------------------------------------


def make_roc_overlay(cat: str) -> None:
    try:
        data = load_mva(cat)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        return

    roc = data.get("roc", {})
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for vid, meta in VARIANT_META.items():
        if vid not in roc:
            continue
        r = roc[vid]
        fpr = np.array(r["fpr"])
        tpr = np.array(r["tpr"])
        lbl = meta["label"].replace("\n", " ") + f"  AUC={r['auc']:.4f}"
        ax.plot(fpr, tpr, color=meta["color"], ls=meta["ls"], lw=2.2, label=lbl)

    ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate (background efficiency)")
    ax.set_ylabel("True Positive Rate (signal efficiency)")
    ax.set_title(
        rf"MVA ROC Curves — $\Lambda$ {cat}"
        "\n(higher AUC for B/C not indicative of better analysis sensitivity)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = FIGS_DIR / f"mva_roc_{cat}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating MVA comparison figures for 20260411 slides...")
    make_auc_bar_chart()
    for cat in ["LL", "DD"]:
        make_roc_overlay(cat)
    print(f"\nAll figures written to {FIGS_DIR}")


if __name__ == "__main__":
    main()
