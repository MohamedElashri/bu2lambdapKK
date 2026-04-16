"""
Passthrough selection systematic — writes zero systematic for all states.

In passthrough mode there is no MVA threshold (or box-cut grid) to vary,
so the selection systematic is identically zero.  This script produces the
same JSON schema that compute_systematics.py expects, with all values set
to 0.0, so the aggregation step runs without modification.

Output schema (per state):
    {
      "sel_syst_abs": 0.0,
      "note": "..."
    }
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATES = ["jpsi", "etac", "chic0", "chic1"]

if "snakemake" in globals():
    output_file = snakemake.output[0]
    branch = snakemake.wildcards.branch
    category = snakemake.wildcards.category
else:
    output_file = (
        "generated/output/studies/selection_systematic/" "selection_systematics_high_yield_LL.json"
    )
    branch = "high_yield"
    category = "LL"

out_path = Path(output_file)
out_path.parent.mkdir(parents=True, exist_ok=True)

result = {
    state: {
        "sel_syst_abs": 0.0,
        "note": (
            "Passthrough mode — no selection optimisation applied; "
            "selection systematic set to zero."
        ),
    }
    for state in STATES
}

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

logger.info(
    f"Zero selection systematic written to {out_path} "
    f"(passthrough mode, branch={branch}, category={category})"
)
