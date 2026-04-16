"""
Passthrough selection — skips optimisation, applies no additional cuts beyond
those already applied in load_data (trigger + stripping + Lambda pre-selection
+ fixed PID > 0.25 + B+ mass window).

Writes a trivial optimized_cuts.json that apply_cuts.py recognises as
opt_type == "passthrough", so it simply copies the pre-selected arrays without
further filtering.  Every downstream step (mass fitting, efficiency, BR ratios,
systematics) runs normally on this larger, unoptimised sample.

Usage (via Snakemake):
    snakemake --config opt_method=passthrough -j4
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if "snakemake" in globals():
    output_file = snakemake.output[0]
    branch = snakemake.params.branch
    category = snakemake.params.category
else:
    output_file = "analysis_output/passthrough/high_yield/LL/models/optimized_cuts.json"
    branch = "high_yield"
    category = "LL"

out_path = Path(output_file)
out_path.parent.mkdir(parents=True, exist_ok=True)

cuts = {
    "opt_type": "passthrough",
    "branch": branch,
    "category": category,
    "note": (
        "No selection optimisation applied. "
        "Sample contains all events passing trigger + stripping + "
        "Lambda pre-selection + PID > 0.25 + B+ mass window. "
        "Results are a first-draft order-of-magnitude estimate."
    ),
}

with open(out_path, "w") as f:
    json.dump(cuts, f, indent=2)

logger.info(f"Passthrough cuts file written to {out_path}")
