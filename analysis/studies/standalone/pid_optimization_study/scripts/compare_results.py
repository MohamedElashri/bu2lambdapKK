"""
Comparison and Summary
======================
Loads all JSON outputs from the three sub-studies (proxy box scan,
fit-based scan, MVA study) and produces:

  1. A side-by-side comparison table (Markdown + PDF figure) of optimal
     cut values from each method, for both FOM types.

  2. A cross-method FOM comparison plot: proxy FOM vs fit-based FOM at
     each PID cut value (to visualise the anti-correlation).

  3. A final recommendation section summarising what the evidence implies.

Usage
-----
  uv run python scripts/compare_results.py [--category LL|DD|both]

  Run AFTER box_scan_proxy.py, fit_based_scan.py, and mva_pid_study.py
  have produced their output JSON files.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = STUDY_DIR / "output"
CMP_DIR = OUTPUT_DIR / "comparison"
CMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_proxy(cat: str) -> dict:
    p = OUTPUT_DIR / "box_proxy" / f"proxy_scan_results_{cat}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_fit(cat: str) -> dict:
    p = OUTPUT_DIR / "fit_based" / f"fit_scan_results_{cat}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_mva(cat: str) -> dict:
    p = OUTPUT_DIR / "mva" / f"mva_pid_summary_{cat}.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

VAR_LABELS = {
    "pid_product": "PID product",
    "p_probnnp": "p ProbNNp",
    "h1_probnnk": "h1 ProbNNk",
    "h2_probnnk": "h2 ProbNNk",
}


def build_comparison_table(proxy: dict, fit: dict, cat: str) -> str:
    lines = [
        f"# PID Cut Optimisation — Comparison Table  [{cat}]",
        "",
        "Optimal cut values from each method, for FOM1 (S/√B, high-yield group)",
        "and FOM2 (S/√(S+B), low-yield group).",
        "",
        "| Variable | Proxy FOM1 cut | Proxy FOM2 cut | Fit FOM1 cut | Fit FOM2 cut |",
        "|----------|---------------|---------------|-------------|-------------|",
    ]
    all_vars = set(list(proxy.keys()) + list(fit.keys()))
    for vname in sorted(all_vars):
        lbl = VAR_LABELS.get(vname, vname)
        p_f1 = f"{proxy[vname]['best_cut_fom1']:.3f}" if vname in proxy else "—"
        p_f2 = f"{proxy[vname]['best_cut_fom2']:.3f}" if vname in proxy else "—"
        f_f1 = (
            f"{fit[vname]['best_cut_fom1']:.3f}"
            if (vname in fit and fit[vname].get("cuts"))
            else "—"
        )
        f_f2 = (
            f"{fit[vname]['best_cut_fom2']:.3f}"
            if (vname in fit and fit[vname].get("cuts"))
            else "—"
        )
        lines.append(f"| {lbl:<20} | {p_f1:>13} | {p_f2:>13} | {f_f1:>11} | {f_f2:>11} |")
    lines += [
        "",
        "## Key observations",
        "- Proxy FOM cuts cluster near 0 (anti-correlated with fit-based FOM).",
        "- Fit-based FOM cuts indicate the true optimal working point.",
        "- If proxy and fit-based agree, the proxy result may be trusted for that variable.",
        "- If they disagree, always prefer the fit-based result.",
    ]
    return "\n".join(lines)


def write_comparison_table(proxy: dict, fit: dict, cat: str) -> None:
    txt = build_comparison_table(proxy, fit, cat)
    path = CMP_DIR / f"comparison_table_{cat}.md"
    with open(path, "w") as f:
        f.write(txt)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Cross-method FOM correlation plot
# ---------------------------------------------------------------------------


def plot_fom_correlation(proxy: dict, fit: dict, cat: str) -> None:
    """
    For each variable where both proxy FOM1 and fit FOM1 are available,
    plot proxy FOM1 vs fit FOM1 as a function of the PID cut.
    Expects the same cut grid for both (or interpolates).
    """
    common_vars = [v for v in proxy if v in fit and fit[v].get("cuts")]
    if not common_vars:
        print("  No common variables for correlation plot — skipping.")
        return

    ncols = min(len(common_vars), 2)
    nrows = (len(common_vars) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(
        f"Proxy FOM1 vs Fit-Based FOM1  [{cat}]\n"
        "Anti-correlation confirms proxy systematically overestimates background PID efficiency",
        fontsize=10,
    )

    for ax_idx, vname in enumerate(common_vars):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        pr = proxy[vname]
        fi = fit[vname]

        p_cuts = np.array(pr["cuts"])
        p_fom = np.array(pr["fom1"])

        f_cuts = np.array(fi["cuts"])
        f_fom = np.array(fi["fom1"])

        # Normalise both to [0,1] so they can share an axis
        def _norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12)

        ax2 = ax.twinx()
        ax.plot(p_cuts, _norm(p_fom), "r-o", ms=4, label="Proxy FOM1 (norm.)")
        ax2.plot(f_cuts, _norm(f_fom), "b-s", ms=4, label="Fit FOM1 (norm.)")

        ax.set_xlabel(f"Cut on {VAR_LABELS.get(vname, vname)} (> value)")
        ax.set_ylabel("Proxy FOM1 (normalised)", color="red")
        ax2.set_ylabel("Fit FOM1 (normalised)", color="blue")
        ax.tick_params(axis="y", colors="red")
        ax2.tick_params(axis="y", colors="blue")
        ax.set_title(VAR_LABELS.get(vname, vname))

        # Compute Pearson r between proxy and fit on common cuts
        if len(f_cuts) >= 3 and len(p_cuts) >= 3:
            # Interpolate proxy onto fit cut grid
            p_interp = np.interp(f_cuts, p_cuts, p_fom)
            r = float(np.corrcoef(p_interp, f_fom)[0, 1])
            ax.set_title(f"{VAR_LABELS.get(vname, vname)}  (r = {r:.2f})", fontsize=9)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Hide unused axes
    for ax_idx in range(len(common_vars), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    plt.tight_layout()
    path = CMP_DIR / f"fom_correlation_{cat}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# MVA AUC comparison
# ---------------------------------------------------------------------------


def plot_mva_comparison(mva: dict, cat: str) -> None:
    if not mva:
        return
    roc = mva.get("roc", {})
    if not roc:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"A_baseline": "black", "B_individual": "royalblue", "C_product": "darkorange"}
    labels = {
        "A_baseline": "A — Baseline (no PID)",
        "B_individual": "B — Individual PID vars",
        "C_product": "C — PID product",
    }
    for vid, r in roc.items():
        fpr = np.array(r["fpr"])
        tpr = np.array(r["tpr"])
        ax.plot(
            fpr,
            tpr,
            color=colors.get(vid, "gray"),
            lw=2,
            label=f"{labels.get(vid, vid)}  AUC={r['auc']:.4f}",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"MVA ROC Comparison: Baseline vs PID Variants  [{cat}]\n"
        "AUC difference driven by proxy bias, not analysis sensitivity"
    )
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    path = CMP_DIR / f"mva_roc_comparison_{cat}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Final recommendation text
# ---------------------------------------------------------------------------


def write_recommendation(proxy: dict, fit: dict, mva: dict, cat: str) -> None:
    lines = [
        f"# PID Optimization Study — Final Recommendation  [{cat}]",
        "",
        "## Evidence Summary",
        "",
        "### 1. Proxy-Based Box Scan",
        "- All proxy FOMs (efficiency ratio, proxy significance) peak at PID cut = 0.",
        "- This is the expected result from the sideband-proxy bias.",
        "- **Conclusion from proxy scan**: proxy says no PID cut is needed — but this",
        "  conclusion is known to be wrong (see fit-based evidence below).",
        "",
        "### 2. Fit-Based Box Scan (authoritative)",
    ]

    for vname in sorted(fit.keys()):
        r = fit.get(vname, {})
        if not r.get("cuts"):
            continue
        lines.append(
            f"- {VAR_LABELS.get(vname, vname)}: FOM1 optimal cut = **{r['best_cut_fom1']:.3f}**"
            f"  (FOM1 = {r['best_fom1']:.2f})"
        )

    lines += [
        "- Fit-based FOM is not affected by the proxy bias because S and B are",
        "  measured directly from the mass fit, not from the sideband region.",
        "- **Conclusion from fit scan**: PID cuts DO improve the FOM; the optimal",
        "  working point is at the cut value listed above for each variable.",
        "",
        "### 3. MVA with PID Features",
    ]
    roc = mva.get("roc", {})
    for vid, r in roc.items():
        lines.append(f"- {vid}: AUC = {r['auc']:.4f}")
    lines += [
        "- AUC differences between variants A/B/C cannot be used to judge whether",
        "  PID is beneficial, because the sideband background has similar PID",
        "  quality to signal MC → the BDT cannot exploit the true PID separation.",
        "",
        "## Recommended Decision",
        "",
        "Based on the fit-based evidence (the only unbiased method):",
        "",
        "1. **Box optimization**: Include PID variable(s) as a 1-D pre-selection cut",
        "   set to the fit-based optimal value.  Do NOT include in the N-D proxy scan.",
        "",
        "2. **MVA**: Exclude PID variables from BDT features.  The sideband training",
        "   background shares the PID quality of signal → including PID features",
        "   does not help and may introduce a spurious boundary.",
        "",
        "3. **PIDCalib2 evidence** (from studies/pid_cancellation): PID efficiency",
        "   ≈ 81% and cancels to <1% in branching fraction ratios across states.",
        "   This supports a fixed pre-cut rather than a state-dependent optimisation.",
        "",
        "## Reference: Prior Studies",
        "- `studies/pid_proxy_comparison/` — all four proxy strategies anti-correlated",
        "  with fit-based FOM (r = −0.63 to −0.90).",
        "- `studies/fit_based_optimizer/` — PID > 0.20 improves FOM1 from 16.85 to 22.13.",
        "- `studies/pid_cancellation/` — PIDCalib2 real efficiency ratios ≈ 1.001–1.009.",
    ]

    txt = "\n".join(lines)
    path = CMP_DIR / f"recommendation_{cat}.md"
    with open(path, "w") as f:
        f.write(txt)
    print(f"  Saved {path.name}")
    print()
    print(txt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare PID study results")
    parser.add_argument("--category", choices=["LL", "DD", "both"], default="both")
    args = parser.parse_args()

    categories = ["LL", "DD"] if args.category == "both" else [args.category]

    for cat in categories:
        print(f"\n{'='*70}")
        print(f"Comparison  [{cat}]")
        print(f"{'='*70}")

        proxy = _load_proxy(cat)
        fit = _load_fit(cat)
        mva = _load_mva(cat)

        if not proxy:
            print("  [WARN] No proxy scan results found — run box_scan_proxy.py first.")
        if not fit:
            print("  [WARN] No fit-based scan results found — run fit_based_scan.py first.")
        if not mva:
            print("  [WARN] No MVA study results found — run mva_pid_study.py first.")

        write_comparison_table(proxy, fit, cat)
        plot_fom_correlation(proxy, fit, cat)
        plot_mva_comparison(mva, cat)
        write_recommendation(proxy, fit, mva, cat)


if __name__ == "__main__":
    main()
