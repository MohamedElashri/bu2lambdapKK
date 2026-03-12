import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


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


def generate_latex_table(high_path, low_path, output_file):
    high_df = load_csv(high_path)
    low_df = load_csv(low_path)

    if high_df is None or low_df is None:
        logger.error("Missing branching fraction ratio tables. Cannot generate LaTeX.")
        sys.exit(1)

    # Combine data
    merged = pd.merge(high_df, low_df, on="state", suffixes=("_high", "_low"))

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
    lines.append("  \\caption{Branching Fraction Products for High and Low Yield Methods}")
    lines.append("  \\renewcommand{\\arraystretch}{1.3} % Add some padding")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\hline\\hline")
    lines.append(
        "    State & $\\mathcal{B}$ Product (High Yield) & $\\mathcal{B}$ Product (Low Yield) \\\\"
    )
    lines.append("    \\hline")

    for _, row in merged.iterrows():
        state = row["state"]
        latex_name = state_tex.get(state, state)

        # High Yield Format
        bf_high = row.get("bf_product_high", 0)
        err_high = row.get("bf_product_err_high", 0)
        str_high = format_latex_scientific(bf_high, err_high)

        # Low Yield Format
        bf_low = row.get("bf_product_low", 0)
        err_low = row.get("bf_product_err_low", 0)
        str_low = format_latex_scientific(bf_low, err_low)

        lines.append(f"    {latex_name} & ${str_high}$ & ${str_low}$ \\\\")

    lines.append("    \\hline\\hline")
    lines.append("  \\end{tabular}")
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
        generate_latex_table(h_path, l_path, out_file)
    else:
        logger.error("This script must be run via Snakemake.")
        sys.exit(1)
