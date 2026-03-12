import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def load_json(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def generate_comparison():
    output_dir = Path("analysis_output")
    mva_dir = output_dir / "mva"
    box_dir = output_dir / "box"
    report_file = output_dir / "method_comparison.md"

    # 1. Extraction
    mva_cuts = load_json(mva_dir / "tables" / "optimized_cuts.json")
    thr_high = mva_cuts.get("mva_threshold_high", "N/A")
    thr_low = mva_cuts.get("mva_threshold_low", "N/A")

    report = [
        "# Optimization Method Comparison: MVA vs Box\n",
        f"**MVA BDT Threshold (High-Yield):** `{thr_high}`\n",
        f"**MVA BDT Threshold (Low-Yield):** `{thr_low}`\n",
        "This report compares the performance of the MVA-based selection and the two-step Box Grid Search.\n",
    ]

    branches = ["high_yield", "low_yield"]

    for branch in branches:
        report.append(f"## Branch: {branch.replace('_', ' ').title()}\n")

        # Load results
        mva_br = load_csv(mva_dir / branch / "tables" / "branching_fraction_ratios.csv")
        box_br = load_csv(box_dir / branch / "tables" / "branching_fraction_ratios.csv")

        if mva_br is not None and box_br is not None:
            merged = pd.merge(mva_br, box_br, on="state", suffixes=("_mva", "_box"))

            # BF Product Table
            report.append("### BF Product Comparison\n")
            cols = ["state", "bf_product_mva", "bf_product_box"]
            merged["ratio (mva/box)"] = merged["bf_product_mva"] / merged["bf_product_box"]
            report.append(merged[cols + ["ratio (mva/box)"]].to_markdown(index=False))
            report.append("\n")

            # Load Yields
            mva_yields = load_csv(mva_dir / branch / "tables" / "fitted_yields.csv")
            box_yields = load_csv(box_dir / branch / "tables" / "fitted_yields.csv")

            if mva_yields is not None and box_yields is not None:
                report.append("### Fitted Yield Comparison (Combined All Years)\n")
                # Filter for combined results if present, or just use state
                y_merged = pd.merge(
                    mva_yields, box_yields, on=["year", "state"], suffixes=("_mva", "_box")
                )
                # Filter for combined
                comb_merged = y_merged[y_merged["year"] == "combined"].copy()
                if not comb_merged.empty:
                    comb_merged["gain_percent"] = (
                        (comb_merged["yield_mva"] - comb_merged["yield_box"])
                        / comb_merged["yield_box"]
                        * 100
                    )
                    y_cols = ["state", "yield_mva", "yield_box", "gain_percent"]
                    report.append(comb_merged[y_cols].to_markdown(index=False))
                    report.append("\n")

    # 2. Save Report
    with open(report_file, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Comparison report saved to {report_file}")


if __name__ == "__main__":
    generate_comparison()
