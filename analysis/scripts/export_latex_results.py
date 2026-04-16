import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# States whose MC simulation is in the LHCb production pipeline.
# They are excluded from the primary LaTeX table and noted in a footnote.
MC_PENDING_STATES = {"etac_2s"}


def load_csv(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def load_syst_json(path: Path) -> dict:
    """Load a required systematics.json artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Required systematics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _syst_err_for_state(syst_data: dict, state: str, bf_product: float, n_sig: float) -> float:
    """Convert total_syst_abs (yield units) → BF-scale absolute uncertainty."""
    entry = syst_data.get(state, {})
    syst_abs_yield = float(entry.get("total_syst_abs", 0.0))
    if n_sig > 0 and bf_product > 0 and syst_abs_yield > 0:
        return syst_abs_yield / n_sig * bf_product
    return 0.0


def format_latex_scientific(val, err):
    """
    Format a value and error into scientific notation for LaTeX.
    Example: 3.27e-06 ± 0.77e-06 -> (3.3 \\pm 0.8) \\times 10^{-6}
    """
    if val == 0 and err == 0:
        return "0"

    # Convert to standard scientific string to find exponent
    val_str = f"{val:e}"
    err_str = f"{err:e}"

    # Extract exponent from the value (assuming error has similar magnitude)
    try:
        exponent = int(val_str.split("e")[1])
    except IndexError:
        exponent = 0

    # Scale values to the exponent
    scaled_val = val / (10**exponent)
    scaled_err = err / (10**exponent)

    # Format conditionally
    if exponent == 0:
        return f"{scaled_val:.2f} \\pm {scaled_err:.2f}"
    else:
        return f"({scaled_val:.1f} \\pm {scaled_err:.1f}) \\times 10^{{{exponent}}}"


def _format_stat_syst(val, stat, syst):
    """Format as (val ± stat ± syst) × 10^n for LaTeX."""
    if val == 0 and stat == 0 and syst == 0:
        return "0"
    try:
        exponent = int(f"{val:e}".split("e")[1])
    except (IndexError, ValueError):
        exponent = 0
    scale = 10**exponent
    sv, ss, sy = val / scale, stat / scale, syst / scale
    if exponent == 0:
        return f"{sv:.2f} \\pm {ss:.2f} \\pm {sy:.2f}"
    return f"({sv:.1f} \\pm {ss:.1f} \\pm {sy:.1f}) \\times 10^{{{exponent}}}"


def generate_latex_table(high_path, low_path, output_file, syst_high_path=None, syst_low_path=None):
    high_df = load_csv(high_path)
    low_df = load_csv(low_path)

    if high_df is None or low_df is None:
        logger.error("Missing branching fraction ratio tables. Cannot generate LaTeX.")
        sys.exit(1)

    # Load yield-scale systematics and convert them to BF-scale uncertainties.
    if not syst_high_path or not syst_low_path:
        logger.error("Systematics JSON paths are required to export the final LaTeX table.")
        sys.exit(1)
    syst_high = load_syst_json(Path(syst_high_path))
    syst_low = load_syst_json(Path(syst_low_path))

    # Drop the placeholder syst_err=0 column before merging to avoid name collisions
    high_df = high_df.drop(columns=["syst_err"], errors="ignore")
    low_df = low_df.drop(columns=["syst_err"], errors="ignore")

    # Combine data
    merged = pd.merge(high_df, low_df, on="state", suffixes=("_high", "_low"))

    # Compute BF-scale systematic uncertainties from yield-scale syst dicts
    def _syst_bf(row, syst: dict, bf_col: str) -> float:
        state = row["state"]
        entry = syst.get(state, {})
        n_nom = float(entry.get("nominal_yield", 0.0))
        syst_abs = float(entry.get("total_syst_abs", 0.0))
        bf = row.get(bf_col, 0)
        return (syst_abs / n_nom * bf) if (n_nom > 0 and bf > 0) else 0.0

    merged["syst_err_high"] = merged.apply(
        _syst_bf, axis=1, syst=syst_high, bf_col="bf_product_high"
    )
    merged["syst_err_low"] = merged.apply(_syst_bf, axis=1, syst=syst_low, bf_col="bf_product_low")

    # Determine if systematic errors are available
    has_syst = (
        merged.get("syst_err_high", pd.Series([0])).abs().max() > 0
        or merged.get("syst_err_low", pd.Series([0])).abs().max() > 0
    )
    caption_note = (
        "First uncertainty is statistical, second is systematic."
        if has_syst
        else "Statistical uncertainties only; systematic studies ongoing."
    )

    # LaTeX mapping for states
    state_tex = {
        "jpsi": "$J/\\psi$",
        "etac": "$\\eta_c(1S)$",
        "chic0": "$\\chi_{c0}$",
        "chic1": "$\\chi_{c1}$",
        "etac_2s": "$\\eta_c(2S)$",
    }

    # Build LaTeX Document
    lines = []
    lines.append("% Auto-generated Branching Fraction Product Results")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(
        f"  \\caption{{Branching Fraction Products for High and Low Yield Methods. {caption_note}}}"
    )
    lines.append("  \\renewcommand{\\arraystretch}{1.3} % Add some padding")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\hline\\hline")
    lines.append(
        "    State & $\\mathcal{B}$ Product (High Yield) & $\\mathcal{B}$ Product (Low Yield) \\\\"
    )
    lines.append("    \\hline")

    pending_states_in_table = []
    for _, row in merged.iterrows():
        state = row["state"]
        if state in MC_PENDING_STATES:
            pending_states_in_table.append(state)
            continue  # exclude from primary table

        latex_name = state_tex.get(state, state)

        bf_high = row.get("bf_product_high", 0)
        err_high = row.get("bf_product_err_high", 0)
        syst_high = row.get("syst_err_high", 0)

        bf_low = row.get("bf_product_low", 0)
        err_low = row.get("bf_product_err_low", 0)
        syst_low = row.get("syst_err_low", 0)

        if has_syst and (syst_high > 0 or syst_low > 0):
            # Format as (val ± stat ± syst) × 10^n
            str_high = _format_stat_syst(bf_high, err_high, syst_high)
            str_low = _format_stat_syst(bf_low, err_low, syst_low)
        else:
            str_high = format_latex_scientific(bf_high, err_high)
            str_low = format_latex_scientific(bf_low, err_low)

        lines.append(f"    {latex_name} & ${str_high}$ & ${str_low}$ \\\\")

    lines.append("    \\hline\\hline")
    lines.append("  \\end{tabular}")
    if pending_states_in_table:
        pending_tex = ", ".join(state_tex.get(s, s) for s in pending_states_in_table)
        lines.append(
            f"  \\caption*{{\\footnotesize "
            f"{pending_tex}: MC simulation in LHCb production pipeline; "
            f"results will be added when samples are available.}}"
        )
    lines.append("  \\label{tab:bf_products}")
    lines.append("\\end{table}")
    lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"LaTeX BF Product table successfully saved to {output_file}")


if __name__ == "__main__":
    if "snakemake" in globals():
        h_path = Path(snakemake.input.high_yield)
        l_path = Path(snakemake.input.low_yield)
        out_file = Path(snakemake.output.report)
        syst_h = Path(snakemake.input.syst_high)
        syst_l = Path(snakemake.input.syst_low)
        generate_latex_table(h_path, l_path, out_file, syst_high_path=syst_h, syst_low_path=syst_l)
    else:
        logger.error("This script must be run via Snakemake.")
        sys.exit(1)
