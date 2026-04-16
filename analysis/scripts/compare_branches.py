import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def compare_branches(high_path_prefix, low_path_prefix, output_file):
    report = ["# Branch Comparison Report\n"]

    # 1. Compare Cuts
    logger.info("Comparing selection cuts...")
    high_cuts = load_json(high_path_prefix / "tables" / "cut_summary.json")
    low_cuts = load_json(low_path_prefix / "tables" / "cut_summary.json")

    if high_cuts and low_cuts:
        report.append("## Selection Cut Comparison\n")
        report.append("| Dataset | High Yield N_pass | Low Yield N_pass | Ratio (Low/High) |")
        report.append("|---------|-------------------|------------------|------------------|")

        # Merge keys
        all_keys = sorted(
            set(high_cuts["data_yields"].keys()) | set(low_cuts["data_yields"].keys())
        )
        for key in all_keys:
            h_n = high_cuts["data_yields"].get(key, 0)
            l_n = low_cuts["data_yields"].get(key, 0)
            ratio = l_n / h_n if h_n > 0 else 0
            report.append(f"| Data {key} | {h_n} | {l_n} | {ratio:.3f} |")

        all_mc = sorted(set(high_cuts["mc_yields"].keys()) | set(low_cuts["mc_yields"].keys()))
        for key in all_mc:
            h_n = high_cuts["mc_yields"].get(key, 0)
            l_n = low_cuts["mc_yields"].get(key, 0)
            ratio = l_n / h_n if h_n > 0 else 0
            report.append(f"| MC {key} | {h_n} | {l_n} | {ratio:.3f} |")
        report.append("\n")

    # 2. Compare Yields
    logger.info("Comparing fitted yields...")
    high_yields = load_csv(high_path_prefix / "tables" / "fitted_yields.csv")
    low_yields = load_csv(low_path_prefix / "tables" / "fitted_yields.csv")

    if high_yields is not None and low_yields is not None:
        report.append("## Fitted Yield Comparison\n")
        merged_yields = pd.merge(
            high_yields, low_yields, on=["year", "state"], suffixes=("_high", "_low")
        )
        merged_yields["ratio"] = merged_yields["yield_low"] / merged_yields["yield_high"]

        table_cols = ["year", "state", "yield_high", "yield_low", "ratio"]
        report.append(merged_yields[table_cols].to_markdown(index=False))
        report.append("\n")

    # 3. Compare Efficiencies
    logger.info("Comparing efficiencies...")
    high_eff = load_csv(high_path_prefix / "tables" / "efficiencies.csv")
    low_eff = load_csv(low_path_prefix / "tables" / "efficiencies.csv")

    if high_eff is not None and low_eff is not None:
        report.append("## Efficiency Comparison\n")
        merged_eff = pd.merge(high_eff, low_eff, on="state", suffixes=("_high", "_low"))
        merged_eff["ratio"] = merged_eff["efficiency_low"] / merged_eff["efficiency_high"]

        table_cols = ["state", "efficiency_high", "efficiency_low", "ratio"]
        report.append(merged_eff[table_cols].to_markdown(index=False))
        report.append("\n")

    # 4. Compare Branching Fraction Ratios
    logger.info("Comparing final branching ratios...")
    high_br = load_csv(high_path_prefix / "tables" / "branching_fraction_ratios.csv")
    low_br = load_csv(low_path_prefix / "tables" / "branching_fraction_ratios.csv")

    if high_br is not None and low_br is not None:
        report.append("## Branching Fraction Ratio Comparison\n")
        merged_br = pd.merge(high_br, low_br, on="state", suffixes=("_high", "_low"))

        # Support both legacy and current branching-ratio column names.
        col_high = (
            "ratio_to_ref_high"
            if "ratio_to_ref_high" in merged_br.columns
            else "ratio_to_jpsi_high"
        )
        col_low = (
            "ratio_to_ref_low" if "ratio_to_ref_low" in merged_br.columns else "ratio_to_jpsi_low"
        )

        merged_br["diff_percent"] = (
            (merged_br[col_low] - merged_br[col_high]) / merged_br[col_high] * 100
        )

        table_cols = [
            "state",
            col_high,
            col_low,
            "bf_product_high",
            "bf_product_low",
            "diff_percent",
        ]
        report.append(merged_br[table_cols].to_markdown(index=False))
        report.append("\n")

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(report))
    logger.info(f"Comparison report saved to {output_file}")


if __name__ == "__main__":
    if "snakemake" in globals():
        high_path = Path(snakemake.input.high_yield).parent.parent
        low_path = Path(snakemake.input.low_yield).parent.parent
        out_file = snakemake.output.report
        compare_branches(high_path, low_path, out_file)
    else:
        # Manual execution for testing
        h_path = Path("analysis_output/mva/high_yield")
        l_path = Path("analysis_output/mva/low_yield")
        o_file = Path("analysis_output/mva/comparison/branch_comparison.md")
        o_file.parent.mkdir(parents=True, exist_ok=True)
        compare_branches(h_path, l_path, o_file)
