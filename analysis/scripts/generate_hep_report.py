"""
Generate HEP-style final results report for B+ → Λ̄pK⁻K⁺ charmonium analysis.

Produces:
  generated/output/reports/final/hep_results.tex  — LaTeX table in LHCb/PDG style
  generated/output/reports/final/hep_results.txt  — Plain-text summary for terminal reading

Measured quantities:
  B(B+ → X K+) × B(X → Λ̄pK⁻)
  for X ∈ {χc0(1P), χc1(1P), ηc(1S)}
  Normalised to:  B(B+ → J/ψ K+) × B(J/ψ → Λ̄pK⁻K⁺)   [PDG 2024]

Usage (standalone):
  uv run python scripts/generate_hep_report.py \\
      --results-dir generated/output/reports/collected \\
      --output-dir  generated/output/reports/final
"""

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDG 2024 normalization constants
# ---------------------------------------------------------------------------
BR_BU_JPSI_K = 1.020e-3  # B(B+ → J/ψ K+)
BR_BU_JPSI_K_ERR = 0.019e-3
BR_JPSI_LPKK = 8.6e-4  # B(J/ψ → p K⁻ Λ̄ + c.c.)
BR_JPSI_LPKK_ERR = 1.1e-4
NORM_FACTOR = BR_BU_JPSI_K * BR_JPSI_LPKK
NORM_FACTOR_ERR = NORM_FACTOR * math.sqrt(
    (BR_BU_JPSI_K_ERR / BR_BU_JPSI_K) ** 2 + (BR_JPSI_LPKK_ERR / BR_JPSI_LPKK) ** 2
)

# ---------------------------------------------------------------------------
# State metadata
# ---------------------------------------------------------------------------
STATES = ["chic0", "chic1", "etac"]  # etac_2s excluded (MC pending)

STATE_META = {
    "chic0": {
        "tex": r"\chi_{c0}(1P)",
        "unicode": "χc0(1P)",
        "pdg_mass": 3414.71,
        "preferred_branch": "low_yield",  # optimised FOM for low-yield states
    },
    "chic1": {
        "tex": r"\chi_{c1}(1P)",
        "unicode": "χc1(1P)",
        "pdg_mass": 3510.67,
        "preferred_branch": "low_yield",
    },
    "etac": {
        "tex": r"\eta_c(1S)",
        "unicode": "ηc(1S)",
        "pdg_mass": 2984.1,
        "preferred_branch": "high_yield",  # optimised FOM for high-yield states
    },
}

# Full decay-mode strings for table rows
DECAY_TEX = {
    "chic0": r"B^+ \to \chi_{c0}(1P)\,K^+,\ \chi_{c0}(1P) \to \bar{\Lambda}\,p\,K^-",
    "chic1": r"B^+ \to \chi_{c1}(1P)\,K^+,\ \chi_{c1}(1P) \to \bar{\Lambda}\,p\,K^-",
    "etac": r"B^+ \to \eta_c(1S)\,K^+,\ \eta_c(1S) \to \bar{\Lambda}\,p\,K^-",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return None
    return pd.read_csv(path)


def load_syst(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def syst_bf(state: str, syst: dict, bf: float) -> float:
    """Convert yield-scale total_syst_abs → BF-scale absolute uncertainty."""
    entry = syst.get(state, {})
    n_nom = float(entry.get("nominal_yield", 0.0))
    syst_abs = float(entry.get("total_syst_abs", 0.0))
    return (syst_abs / n_nom * bf) if (n_nom > 0 and bf > 0) else 0.0


def significance(val: float, stat: float, syst: float) -> float:
    """Naive Gaussian significance: value / total_uncertainty."""
    total = math.sqrt(stat**2 + syst**2)
    return val / total if total > 0 else 0.0


def sci_tex(val: float, stat: float, syst: float) -> str:
    """
    Format (val ± stat ± syst) × 10^n for LaTeX.
    If syst == 0, format as (val ± stat) × 10^n.
    """
    if val == 0:
        return "0"
    exp = int(f"{val:.2e}".split("e")[1])
    scale = 10**exp
    sv, ss, sy = val / scale, stat / scale, syst / scale
    if syst > 0:
        body = rf"{sv:.1f} \pm {ss:.1f} \pm {sy:.1f}"
    else:
        body = rf"{sv:.1f} \pm {ss:.1f}"
    return rf"({body}) \times 10^{{{exp}}}"


def sci_txt(val: float, stat: float, syst: float) -> str:
    """Plain-text version."""
    if val == 0:
        return "0"
    exp = int(f"{val:.2e}".split("e")[1])
    scale = 10**exp
    sv, ss, sy = val / scale, stat / scale, syst / scale
    if syst > 0:
        return f"({sv:.1f} ± {ss:.1f}(stat) ± {sy:.1f}(syst)) × 10^{exp}"
    return f"({sv:.1f} ± {ss:.1f}(stat)) × 10^{exp}"


def weighted_average(
    val_h: float,
    stat_h: float,
    syst_h: float,
    val_l: float,
    stat_l: float,
    syst_l: float,
) -> tuple[float, float, float]:
    """
    Combine two measurements (high/low yield) via inverse-total-variance weighting.
    Returns (combined_val, combined_stat, combined_syst).
    Systematics are treated as uncorrelated between methods.
    """
    tot_h = math.sqrt(stat_h**2 + syst_h**2)
    tot_l = math.sqrt(stat_l**2 + syst_l**2)
    if tot_h == 0 and tot_l == 0:
        return val_h, stat_h, syst_h
    w_h = 1 / tot_h**2 if tot_h > 0 else 0
    w_l = 1 / tot_l**2 if tot_l > 0 else 0
    w_tot = w_h + w_l
    val = (w_h * val_h + w_l * val_l) / w_tot
    stat = math.sqrt(w_h**2 * stat_h**2 + w_l**2 * stat_l**2) / w_tot
    syst = math.sqrt(w_h**2 * syst_h**2 + w_l**2 * syst_l**2) / w_tot
    return val, stat, syst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_results(results_dir: Path) -> list[dict]:
    """Load CSVs + systematics, compute per-state measurements."""
    tables = results_dir / "tables"
    rows = []
    for branch in ("high_yield", "low_yield"):
        df = load_csv(tables / f"branching_fraction_ratios_{branch}.csv")
        syst = load_syst(tables / f"systematics_{branch}.json")
        if df is None:
            continue
        for state in STATES:
            row = df[df["state"] == state]
            if row.empty:
                continue
            bf = float(row["bf_product"].values[0])
            stat = float(row["bf_product_err"].values[0])
            sy = syst_bf(state, syst, bf)
            rows.append({"state": state, "branch": branch, "bf": bf, "stat": stat, "syst": sy})
    return rows


def generate(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = build_results(results_dir)
    if not raw:
        logger.error("No results found — run make collect-results first.")
        return

    # Organise into per-state dicts
    data = {}
    for r in raw:
        data.setdefault(r["state"], {})[r["branch"]] = r

    # Build per-state summary
    summary = {}
    for state in STATES:
        if state not in data:
            continue
        m = STATE_META[state]
        preferred = m["preferred_branch"]
        alt = "high_yield" if preferred == "low_yield" else "low_yield"

        d_pref = data[state].get(preferred, {})
        d_alt = data[state].get(alt, {})

        val_p = d_pref.get("bf", 0)
        stat_p = d_pref.get("stat", 0)
        syst_p = d_pref.get("syst", 0)
        val_a = d_alt.get("bf", 0)
        stat_a = d_alt.get("stat", 0)
        syst_a = d_alt.get("syst", 0)

        # Weighted combination of the two methods
        val_c, stat_c, syst_c = weighted_average(val_p, stat_p, syst_p, val_a, stat_a, syst_a)

        summary[state] = {
            "preferred_branch": preferred,
            "val_pref": val_p,
            "stat_pref": stat_p,
            "syst_pref": syst_p,
            "val_alt": val_a,
            "stat_alt": stat_a,
            "syst_alt": syst_a,
            "val_comb": val_c,
            "stat_comb": stat_c,
            "syst_comb": syst_c,
            "sig_comb": significance(val_c, stat_c, syst_c),
        }

    _write_latex(summary, output_dir / "hep_results.tex")
    _write_text(summary, output_dir / "hep_results.txt")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def _write_latex(summary: dict, path: Path):
    norm_tex = (
        rf"\mathcal{{B}}(B^+ \to J/\psi\,K^+) \times "
        rf"\mathcal{{B}}(J/\psi \to \bar{{\Lambda}}\,p\,K^-) = "
        rf"({NORM_FACTOR * 1e7:.2f} \pm {NORM_FACTOR_ERR * 1e7:.2f}) \times 10^{{-7}}"
    )

    lines = []
    lines += [
        r"% ================================================================",
        r"% HEP Results Table — B+ → Λ̄pK⁻K⁺ Charmonium Analysis",
        r"% Auto-generated by generate_hep_report.py",
        r"% ================================================================",
        r"",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \renewcommand{\arraystretch}{1.4}",
        (
            r"  \caption{Measurements of $\mathcal{B}(B^+ \to X\,K^+) \times "
            r"\mathcal{B}(X \to \bar{\Lambda}\,p\,K^-)$ for charmonium states $X$. "
            r"The first uncertainty is statistical and the second is systematic. "
            r"Results are obtained from a weighted combination of two independent "
            r"selection optimisations (high-yield and low-yield). "
            r"Significances are computed assuming Gaussian errors.}"
        ),
        r"  \begin{tabular}{lcc}",
        r"    \hline\hline",
        r"    Decay mode & $\mathcal{B} \times \mathcal{B}$ & Significance \\",
        r"    \hline",
    ]

    for state in STATES:
        if state not in summary:
            continue
        s = summary[state]
        decay = DECAY_TEX[state]
        bf_str = sci_tex(s["val_comb"], s["stat_comb"], s["syst_comb"])
        sig = s["sig_comb"]
        lines.append(rf"    ${decay}$ & ${bf_str}$ & ${sig:.1f}\sigma$ \\")

    lines += [
        r"    \hline\hline",
        r"  \end{tabular}",
        (
            rf"  \caption*{{\footnotesize Normalisation: ${norm_tex}$ (PDG~2024). "
            r"Systematic uncertainties include fit model variation, "
            r"selection threshold, PID efficiency, and kinematic reweighting.}}"
        ),
        r"  \label{tab:hep_results}",
        r"\end{table}",
        r"",
    ]

    # --- Systematics breakdown table ---
    lines += [
        r"% ----------------------------------------------------------------",
        r"% Systematic uncertainty breakdown",
        r"% ----------------------------------------------------------------",
        r"",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \renewcommand{\arraystretch}{1.3}",
        r"  \caption{Breakdown of relative systematic uncertainties (\%).}",
        r"  \begin{tabular}{lccc}",
        r"    \hline\hline",
        r"    Source & $\chi_{c0}(1P)$ & $\chi_{c1}(1P)$ & $\eta_c(1S)$ \\",
        r"    \hline",
    ]

    # Load raw systematics for breakdown
    syst_h = _load_syst_from_tables(Path("results") / "tables" / "systematics_high_yield.json")
    syst_l = _load_syst_from_tables(Path("results") / "tables" / "systematics_low_yield.json")

    def _rel(state, key, syst_dict):
        e = syst_dict.get(state, {})
        n = float(e.get("nominal_yield", 1.0) or 1.0)
        if "rel" in key:
            return float(e.get(key, 0.0)) * 100
        return float(e.get(key, 0.0)) / n * 100

    syst_map = {
        "chic0": syst_l,
        "chic1": syst_l,
        "etac": syst_h,
    }

    breakdown_rows = [
        ("Fit model", "fit_syst_abs", False),
        ("Selection threshold", "sel_syst_abs", False),
        ("PID efficiency", "pid_syst_rel", True),
        ("Kinematic reweighting", "kin_syst_rel", True),
        ("Total", "total_syst_abs", False),
    ]

    for label, key, is_rel in breakdown_rows:
        vals = []
        for st in STATES:
            sd = syst_map.get(st, {})
            e = sd.get(st, {})
            if is_rel:
                v = float(e.get(key, 0.0)) * 100
            else:
                n = float(e.get("nominal_yield", 1.0) or 1.0)
                v = float(e.get(key, 0.0)) / n * 100
            vals.append(f"{v:.1f}")
        row = " & ".join(vals)
        sep = r"    \hline" if label == "Total" else ""
        if sep:
            lines.append(sep)
        lines.append(rf"    {label} & {row} \\")

    lines += [
        r"    \hline\hline",
        r"  \end{tabular}",
        r"  \label{tab:systematics}",
        r"\end{table}",
        r"",
    ]

    path.write_text("\n".join(lines))
    logger.info(f"LaTeX report → {path}")


def _load_syst_from_tables(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plain-text output
# ---------------------------------------------------------------------------


def _write_text(summary: dict, path: Path):
    norm_txt = (
        f"B(B+ → J/ψ K+) × B(J/ψ → Λ̄pK⁻K⁺) = "
        f"({NORM_FACTOR * 1e7:.2f} ± {NORM_FACTOR_ERR * 1e7:.2f}) × 10⁻⁷  [PDG 2024]"
    )

    w = 72
    lines = [
        "=" * w,
        "  B⁺ → Λ̄pK⁻K⁺ Charmonium Analysis — Final Results",
        "=" * w,
        "",
        "  Measured quantity:",
        "    B(B+ → X K+) × B(X → Λ̄ p K⁻)",
        "    for X ∈ {χc0(1P), χc1(1P), ηc(1S)}",
        "",
        f"  Normalisation: {norm_txt}",
        "",
        "-" * w,
        f"  {'Decay channel':<32}  {'B × B':>26}  {'Sig.':>6}",
        "-" * w,
    ]

    for state in STATES:
        if state not in summary:
            continue
        s = summary[state]
        meta = STATE_META[state]
        channel = f"B+ → {meta['unicode']} K+, {meta['unicode']} → Λ̄ p K⁻"
        bf_str = sci_txt(s["val_comb"], s["stat_comb"], s["syst_comb"])
        sig = s["sig_comb"]
        lines.append(f"  {channel:<32}  {bf_str:>26}  {sig:.1f}σ")

    lines += [
        "-" * w,
        "",
        "  Individual method comparison:",
        f"  {'State':<10}  {'Preferred method':>18}  {'Cross-check':>18}",
        f"  {'-'*10}  {'-'*18}  {'-'*18}",
    ]

    for state in STATES:
        if state not in summary:
            continue
        s = summary[state]
        meta = STATE_META[state]
        pref = s["preferred_branch"].replace("_", " ")
        alt = "low yield" if pref == "high yield" else "high yield"

        def _fmt(val, stat, syst):
            if val == 0:
                return "N/A"
            exp = int(f"{val:.2e}".split("e")[1])
            sc = 10**exp
            sy = f"±{syst/sc:.1f}" if syst > 0 else ""
            return f"({val/sc:.1f}±{stat/sc:.1f}{sy})×10^{exp}"

        pref_str = _fmt(s["val_pref"], s["stat_pref"], s["syst_pref"])
        alt_str = _fmt(s["val_alt"], s["stat_alt"], s["syst_alt"])
        lines.append(f"  {meta['unicode']:<10}  {pref_str:>18}  {alt_str:>18}")

    lines += [
        "",
        "  Notes:",
        "    • Preferred method for χc0/χc1: low-yield optimisation",
        "    • Preferred method for ηc(1S):  high-yield optimisation",
        "    • Combined result: inverse-variance weighted average",
        "    • ηc(2S): MC in LHCb production pipeline; result pending",
        "    • Systematics: fit model, selection threshold, PID, reweighting",
        "=" * w,
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Text report  → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HEP-style results report")
    parser.add_argument(
        "--results-dir",
        default="generated/output/reports/collected",
        help="Collected results directory",
    )
    parser.add_argument(
        "--output-dir",
        default="generated/output/reports/final",
        help="Where to write hep_results.*",
    )
    args = parser.parse_args()

    generate(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
    )
