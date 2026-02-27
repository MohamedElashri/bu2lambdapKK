"""
Cross-Validation Optimizer Study

Purpose:
  Quantify optimization bias by running the 252-point fit-based grid scan
  independently on odd-indexed and even-indexed events, then cross-validating:
    - Train on odd  → validate on even  (odd→even direction)
    - Train on even → validate on odd   (even→odd direction)

  If the optimal cuts agree between halves and FOM_val/FOM_train ≈ 1,
  the selection optimization is robust (low bias).

Split criterion:
  Events are divided by their position index in the concatenated all-years
  array (after PID>0.20 + B+ mass window). Index parity (% 2 == 0/1) gives
  a reproducible, unbiased 50/50 split.

Grid (same as fit_based_optimizer):
  Bu_DTF_chi2     < {10, 20, 30}
  Bu_FDCHI2_OWNPV > {100, 150, 200, 250}
  Bu_IPCHI2_OWNPV < {4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0}
  Bu_PT           > {3000, 3200, 3400}
  PID_product     > 0.20  [fixed]
  Total: 252 grid points × 2 halves = 504 scan fits

FOM Definitions:
  FOM1 = (N_jpsi + N_etac) / sqrt(N_bkg)         [S/sqrt(B), high-yield]
  FOM2 = S2 / sqrt(S2 + N_bkg)                    [S/sqrt(S+B), Punzi FOM]
  where S2 = N_chic0 + N_chic1 + N_etac_2s

Outputs:
  output/scan_odd.csv              — 252 scan results on odd half
  output/scan_even.csv             — 252 scan results on even half
  output/crossval_summary.csv      — cross-validation comparison table
  output/crossval_report.pdf       — 4-panel comparison plot + cut tables
  output/fits/val_set1_odd2even.pdf  — Set1: odd-optimal cuts on even data
  output/fits/val_set1_even2odd.pdf  — Set1: even-optimal cuts on odd data
  output/fits/val_set2_odd2even.pdf  — Set2: odd-optimal cuts on even data
  output/fits/val_set2_even2odd.pdf  — Set2: even-optimal cuts on odd data
"""

import itertools
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")

from modules.cache_manager import CacheManager
from modules.data_handler import TOMLConfig
from modules.exceptions import AnalysisError
from modules.mass_fitter import MassFitter

# ---------------------------------------------------------------------------
# Snakemake params
# ---------------------------------------------------------------------------
config_dir = snakemake.params.config_dir
cache_dir = snakemake.params.cache_dir
output_dir = snakemake.params.output_dir

csv_odd_output = snakemake.output.csv_odd
csv_even_output = snakemake.output.csv_even
summary_output = snakemake.output.summary
report_output = snakemake.output.report
val_s1_o2e_output = snakemake.output.val_s1_o2e
val_s1_e2o_output = snakemake.output.val_s1_e2o
val_s2_o2e_output = snakemake.output.val_s2_o2e
val_s2_e2o_output = snakemake.output.val_s2_e2o

ref_cuts_csv = snakemake.input.ref_cuts

years = snakemake.config.get("years", ["2016", "2017", "2018"])
track_types = snakemake.config.get("track_types", ["LL", "DD"])

Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(val_s1_o2e_output).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PID_CUT = 0.20
BU_MIN = 5255.0
BU_MAX = 5305.0

GRID = {
    "chi2": [10.0, 20.0, 30.0],
    "fdchi2": [100.0, 150.0, 200.0, 250.0],
    "ipchi2": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0],
    "pt": [3000.0, 3200.0, 3400.0],
}
TOTAL_GRID = 1
for v in GRID.values():
    TOTAL_GRID *= len(v)  # = 252

# Grid steps for stability check (one full step allowed)
GRID_STEPS = {"chi2": 10.0, "fdchi2": 50.0, "ipchi2": 0.5, "pt": 200.0}

# ---------------------------------------------------------------------------
# Configuration and cache
# ---------------------------------------------------------------------------
config = TOMLConfig(config_dir)
config.paths["output"]["plots_dir"] = output_dir

cache = CacheManager(cache_dir)

config_files = list(Path(config_dir).glob("*.toml"))
code_files = [
    project_root / "modules" / "data_handler.py",
    project_root / "modules" / "lambda_selector.py",
]
step2_deps = cache.compute_dependencies(
    config_files=config_files,
    code_files=code_files,
    extra_params={"years": years, "track_types": track_types},
)

print("Loading Step 2 cached data...")
data_dict = cache.load("step2_data_after_lambda", dependencies=step2_deps)

if not data_dict:
    raise AnalysisError("Step 2 cached data not found! Run load_data first.")

# ---------------------------------------------------------------------------
# Build concatenated array
# ---------------------------------------------------------------------------
data_combined = {}
for year in data_dict:
    arrays = [
        data_dict[year][tt] for tt in data_dict[year] if hasattr(data_dict[year][tt], "layout")
    ]
    if arrays:
        data_combined[year] = ak.concatenate(arrays, axis=0)

all_data = ak.concatenate(list(data_combined.values()), axis=0)
print(f"Total events loaded: {len(all_data):,}")

bu_branch = "Bu_MM_corrected" if "Bu_MM_corrected" in all_data.fields else "Bu_M"


def flat(arr, branch):
    """Extract branch as flat 1D numpy array."""
    a = arr[branch]
    if "var" in str(ak.type(a)):
        a = ak.firsts(a)
    return ak.to_numpy(a)


chi2_all = flat(all_data, "Bu_DTF_chi2")
fdchi2_all = flat(all_data, "Bu_FDCHI2_OWNPV")
ipchi2_all = flat(all_data, "Bu_IPCHI2_OWNPV")
pt_all = flat(all_data, "Bu_PT")
pid_all = flat(all_data, "PID_product")
bu_all = flat(all_data, bu_branch)

# Apply fixed pre-cuts (PID + B+ mass window)
base_mask = (pid_all > PID_CUT) & (bu_all >= BU_MIN) & (bu_all <= BU_MAX)
all_data_base = all_data[base_mask]
chi2_base = chi2_all[base_mask]
fdchi2_base = fdchi2_all[base_mask]
ipchi2_base = ipchi2_all[base_mask]
pt_base = pt_all[base_mask]

n_base = len(all_data_base)
print(f"Events after PID>{PID_CUT} + B+ window: {n_base:,}")

# ---------------------------------------------------------------------------
# Odd / Even split by array-index parity
# ---------------------------------------------------------------------------
idx = np.arange(n_base)
odd_mask_split = idx % 2 == 1
even_mask_split = idx % 2 == 0

all_data_odd = all_data_base[odd_mask_split]
chi2_odd = chi2_base[odd_mask_split]
fdchi2_odd = fdchi2_base[odd_mask_split]
ipchi2_odd = ipchi2_base[odd_mask_split]
pt_odd = pt_base[odd_mask_split]

all_data_even = all_data_base[even_mask_split]
chi2_even = chi2_base[even_mask_split]
fdchi2_even = fdchi2_base[even_mask_split]
ipchi2_even = ipchi2_base[even_mask_split]
pt_even = pt_base[even_mask_split]

print(f"  Odd  half: {len(all_data_odd):,} events")
print(f"  Even half: {len(all_data_even):,} events")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_yields(results, key):
    """Extract yields and compute FOM1/FOM2 from a MassFitter results dict."""
    ylds = results["yields"][key]
    n_jpsi = max(ylds.get("jpsi", (0, 0))[0], 0.0)
    n_etac = max(ylds.get("etac", (0, 0))[0], 0.0)
    n_chic0 = max(ylds.get("chic0", (0, 0))[0], 0.0)
    n_chic1 = max(ylds.get("chic1", (0, 0))[0], 0.0)
    n_etac_2s = max(ylds.get("etac_2s", (0, 0))[0], 0.0)
    n_bkg = max(ylds.get("background", (0, 0))[0], 1.0)

    s1 = n_jpsi + n_etac
    fom1 = s1 / np.sqrt(n_bkg)

    s2 = n_chic0 + n_chic1 + n_etac_2s
    fom2 = s2 / np.sqrt(max(s2 + n_bkg, 1.0))

    return {
        "fom1": fom1,
        "fom2": fom2,
        "n_jpsi": n_jpsi,
        "n_etac": n_etac,
        "n_chic0": n_chic0,
        "n_chic1": n_chic1,
        "n_etac_2s": n_etac_2s,
        "n_bkg": n_bkg,
    }


def run_scan(data_h, chi2_h, fdchi2_h, ipchi2_h, pt_h, label):
    """Run 252-point grid scan on one half of the data.  Returns (DataFrame, n_failed)."""
    records = []
    n_failed = 0

    with tqdm(total=TOTAL_GRID, desc=f"Scan ({label})", unit="fit") as pbar:
        for c, f, i, p in itertools.product(
            GRID["chi2"], GRID["fdchi2"], GRID["ipchi2"], GRID["pt"]
        ):
            mask = (chi2_h < c) & (fdchi2_h > f) & (ipchi2_h < i) & (pt_h > p)
            n_events = int(np.sum(mask))

            rec = {
                "chi2": c,
                "fdchi2": f,
                "ipchi2": i,
                "pt": p,
                "n_events": n_events,
                "N_jpsi": np.nan,
                "N_etac": np.nan,
                "N_chic0": np.nan,
                "N_chic1": np.nan,
                "N_etac_2s": np.nan,
                "N_bkg": np.nan,
                "FOM1": np.nan,
                "FOM2": np.nan,
            }

            if n_events >= 50:
                with tempfile.TemporaryDirectory() as tmpdir:
                    config.paths["output"]["plots_dir"] = tmpdir
                    fitter = MassFitter(config)
                    try:
                        results = fitter.perform_fit({"scan": data_h[mask]}, fit_combined=True)
                        vals = extract_yields(results, "scan")
                        rec.update(
                            {
                                "N_jpsi": vals["n_jpsi"],
                                "N_etac": vals["n_etac"],
                                "N_chic0": vals["n_chic0"],
                                "N_chic1": vals["n_chic1"],
                                "N_etac_2s": vals["n_etac_2s"],
                                "N_bkg": vals["n_bkg"],
                                "FOM1": vals["fom1"],
                                "FOM2": vals["fom2"],
                            }
                        )
                    except Exception:
                        n_failed += 1
                        tqdm.write(
                            f"  WARN [{label}]: fit failed at " f"chi2<{c} FD>{f} IP<{i} PT>{p}"
                        )
            else:
                n_failed += 1

            records.append(rec)
            pbar.update(1)

    config.paths["output"]["plots_dir"] = output_dir
    return pd.DataFrame(records), n_failed


def run_val_fit(data_v, chi2_v, fdchi2_v, ipchi2_v, pt_v, cuts, out_pdf):
    """Apply a set of optimal cuts to the validation half and run a single fit.

    Returns dict with fom1, fom2, n_events, and yield values.
    Copies the mass fit PDF to out_pdf.
    """
    mask = (
        (chi2_v < cuts["chi2"])
        & (fdchi2_v > cuts["fdchi2"])
        & (ipchi2_v < cuts["ipchi2"])
        & (pt_v > cuts["pt"])
    )
    n_val = int(np.sum(mask))

    result = {
        "fom1": np.nan,
        "fom2": np.nan,
        "n_events": n_val,
        "n_jpsi": np.nan,
        "n_etac": np.nan,
        "n_chic0": np.nan,
        "n_chic1": np.nan,
        "n_etac_2s": np.nan,
        "n_bkg": np.nan,
    }

    dst = Path(out_pdf)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if n_val < 50:
        print(f"  WARNING: only {n_val} validation events — skipping fit")
        dst.touch()
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        config.paths["output"]["plots_dir"] = tmpdir
        fitter = MassFitter(config)
        try:
            fit_results = fitter.perform_fit({"val": data_v[mask]}, fit_combined=True)
            vals = extract_yields(fit_results, "val")
            result.update(vals)
        except Exception:
            tb = traceback.format_exc()
            print(f"  WARNING: validation fit failed: {tb.splitlines()[-1]}")

        # Copy fit PDF to output
        src = Path(tmpdir) / "fits" / "mass_fit_val.pdf"
        if src.exists():
            shutil.copy(src, dst)
        else:
            fits_dir = Path(tmpdir) / "fits"
            pdfs = sorted(fits_dir.glob("*.pdf")) if fits_dir.exists() else []
            if pdfs:
                shutil.copy(pdfs[-1], dst)
                print(f"  Saved fallback PDF: {dst.name}")
            else:
                dst.touch()
                print(f"  WARNING: no fit PDF found for {dst.name}")

    config.paths["output"]["plots_dir"] = output_dir
    result["n_events"] = n_val
    return result


def cuts_stable(a, b):
    """Return True if all cut values agree between the two optimal rows."""
    return all(abs(float(a[k]) - float(b[k])) < 1e-6 for k in ["chi2", "fdchi2", "ipchi2", "pt"])


def cuts_within_one_step(a, b):
    """Return True if cuts agree within one grid step."""
    return all(
        abs(float(a[k]) - float(b[k])) <= GRID_STEPS[k] + 1e-6
        for k in ["chi2", "fdchi2", "ipchi2", "pt"]
    )


def safe_ratio(val, train):
    if np.isnan(val) or np.isnan(train) or train == 0:
        return np.nan
    return val / train


# ---------------------------------------------------------------------------
# Phase 1A: Grid scan on odd half
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print(f"Phase 1A: Grid scan — ODD half ({len(all_data_odd):,} events)")
print(f"{'=' * 65}")

df_odd, n_fail_odd = run_scan(all_data_odd, chi2_odd, fdchi2_odd, ipchi2_odd, pt_odd, "odd")
df_odd.to_csv(csv_odd_output, index=False)
print(f"Odd scan: {n_fail_odd}/{TOTAL_GRID} failures. Saved: {csv_odd_output}")

# ---------------------------------------------------------------------------
# Phase 1B: Grid scan on even half
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print(f"Phase 1B: Grid scan — EVEN half ({len(all_data_even):,} events)")
print(f"{'=' * 65}")

df_even, n_fail_even = run_scan(all_data_even, chi2_even, fdchi2_even, ipchi2_even, pt_even, "even")
df_even.to_csv(csv_even_output, index=False)
print(f"Even scan: {n_fail_even}/{TOTAL_GRID} failures. Saved: {csv_even_output}")

# ---------------------------------------------------------------------------
# Find optimal cuts from each half
# ---------------------------------------------------------------------------
df_odd_v = df_odd.dropna(subset=["FOM1", "FOM2"])
df_even_v = df_even.dropna(subset=["FOM1", "FOM2"])

if df_odd_v.empty or df_even_v.empty:
    raise AnalysisError("All fits failed in one or both halves — cannot determine optimal cuts.")

best1_odd = df_odd_v.loc[df_odd_v["FOM1"].idxmax()]
best2_odd = df_odd_v.loc[df_odd_v["FOM2"].idxmax()]
best1_even = df_even_v.loc[df_even_v["FOM1"].idxmax()]
best2_even = df_even_v.loc[df_even_v["FOM2"].idxmax()]

print(
    f"\nOdd  half → Set1: chi2<{best1_odd['chi2']:.0f} FD>{best1_odd['fdchi2']:.0f} "
    f"IP<{best1_odd['ipchi2']:.1f} PT>{best1_odd['pt']:.0f}  FOM1={best1_odd['FOM1']:.3f}"
)
print(
    f"Even half → Set1: chi2<{best1_even['chi2']:.0f} FD>{best1_even['fdchi2']:.0f} "
    f"IP<{best1_even['ipchi2']:.1f} PT>{best1_even['pt']:.0f}  FOM1={best1_even['FOM1']:.3f}"
)
print(
    f"Odd  half → Set2: chi2<{best2_odd['chi2']:.0f} FD>{best2_odd['fdchi2']:.0f} "
    f"IP<{best2_odd['ipchi2']:.1f} PT>{best2_odd['pt']:.0f}  FOM2={best2_odd['FOM2']:.3f}"
)
print(
    f"Even half → Set2: chi2<{best2_even['chi2']:.0f} FD>{best2_even['fdchi2']:.0f} "
    f"IP<{best2_even['ipchi2']:.1f} PT>{best2_even['pt']:.0f}  FOM2={best2_even['FOM2']:.3f}"
)

# Load full-dataset reference
df_ref = pd.read_csv(ref_cuts_csv)
ref1 = df_ref[df_ref["Set"] == "Set1_HighYield"].iloc[0]
ref2 = df_ref[df_ref["Set"] == "Set2_LowYield"].iloc[0]

# ---------------------------------------------------------------------------
# Phase 2: Validation fits (4 fits total)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("Phase 2: Validation fits (4 fits)")
print(f"{'=' * 65}")

print("\n[1/4] Set1: odd-optimal cuts → even data")
val_s1_o2e = run_val_fit(
    all_data_even, chi2_even, fdchi2_even, ipchi2_even, pt_even, best1_odd, val_s1_o2e_output
)
print(f"      FOM1={val_s1_o2e['fom1']:.3f}  n_events={val_s1_o2e['n_events']:,}")

print("\n[2/4] Set1: even-optimal cuts → odd data")
val_s1_e2o = run_val_fit(
    all_data_odd, chi2_odd, fdchi2_odd, ipchi2_odd, pt_odd, best1_even, val_s1_e2o_output
)
print(f"      FOM1={val_s1_e2o['fom1']:.3f}  n_events={val_s1_e2o['n_events']:,}")

print("\n[3/4] Set2: odd-optimal cuts → even data")
val_s2_o2e = run_val_fit(
    all_data_even, chi2_even, fdchi2_even, ipchi2_even, pt_even, best2_odd, val_s2_o2e_output
)
print(f"      FOM2={val_s2_o2e['fom2']:.3f}  n_events={val_s2_o2e['n_events']:,}")

print("\n[4/4] Set2: even-optimal cuts → odd data")
val_s2_e2o = run_val_fit(
    all_data_odd, chi2_odd, fdchi2_odd, ipchi2_odd, pt_odd, best2_even, val_s2_e2o_output
)
print(f"      FOM2={val_s2_e2o['fom2']:.3f}  n_events={val_s2_e2o['n_events']:,}")

# ---------------------------------------------------------------------------
# Compute stability flags and ratios
# ---------------------------------------------------------------------------
s1_exact = cuts_stable(best1_odd, best1_even)
s2_exact = cuts_stable(best2_odd, best2_even)
s1_close = cuts_within_one_step(best1_odd, best1_even)
s2_close = cuts_within_one_step(best2_odd, best2_even)

s1_label = "EXACT MATCH" if s1_exact else ("WITHIN 1 STEP" if s1_close else "DIFFERS >1 STEP")
s2_label = "EXACT MATCH" if s2_exact else ("WITHIN 1 STEP" if s2_close else "DIFFERS >1 STEP")

r1_o2e = safe_ratio(val_s1_o2e["fom1"], best1_odd["FOM1"])
r1_e2o = safe_ratio(val_s1_e2o["fom1"], best1_even["FOM1"])
r2_o2e = safe_ratio(val_s2_o2e["fom2"], best2_odd["FOM2"])
r2_e2o = safe_ratio(val_s2_e2o["fom2"], best2_even["FOM2"])

# ---------------------------------------------------------------------------
# Phase 3: Save summary CSV
# ---------------------------------------------------------------------------
summary_rows = [
    {
        "Set": "Set1_HighYield",
        "FOM": "FOM1=S/sqrt(B)",
        "direction": "odd_train",
        "chi2": best1_odd["chi2"],
        "fdchi2": best1_odd["fdchi2"],
        "ipchi2": best1_odd["ipchi2"],
        "pt": best1_odd["pt"],
        "pid": PID_CUT,
        "train_fom": best1_odd["FOM1"],
        "val_fom": val_s1_o2e["fom1"],
        "fom_ratio": r1_o2e,
        "n_train": df_odd_v.loc[df_odd_v["FOM1"].idxmax(), "n_events"],
        "n_val": val_s1_o2e["n_events"],
        "cut_stability": s1_label,
    },
    {
        "Set": "Set1_HighYield",
        "FOM": "FOM1=S/sqrt(B)",
        "direction": "even_train",
        "chi2": best1_even["chi2"],
        "fdchi2": best1_even["fdchi2"],
        "ipchi2": best1_even["ipchi2"],
        "pt": best1_even["pt"],
        "pid": PID_CUT,
        "train_fom": best1_even["FOM1"],
        "val_fom": val_s1_e2o["fom1"],
        "fom_ratio": r1_e2o,
        "n_train": df_even_v.loc[df_even_v["FOM1"].idxmax(), "n_events"],
        "n_val": val_s1_e2o["n_events"],
        "cut_stability": s1_label,
    },
    {
        "Set": "Set1_HighYield",
        "FOM": "FOM1=S/sqrt(B)",
        "direction": "full_reference",
        "chi2": ref1["bu_dtf_chi2"],
        "fdchi2": ref1["bu_fdchi2"],
        "ipchi2": ref1["bu_ipchi2"],
        "pt": ref1["bu_pt"],
        "pid": PID_CUT,
        "train_fom": ref1["FOM_value"],
        "val_fom": np.nan,
        "fom_ratio": np.nan,
        "n_train": ref1.get("n_events", np.nan),
        "n_val": np.nan,
        "cut_stability": "REFERENCE",
    },
    {
        "Set": "Set2_LowYield",
        "FOM": "FOM2=S/sqrt(S+B)",
        "direction": "odd_train",
        "chi2": best2_odd["chi2"],
        "fdchi2": best2_odd["fdchi2"],
        "ipchi2": best2_odd["ipchi2"],
        "pt": best2_odd["pt"],
        "pid": PID_CUT,
        "train_fom": best2_odd["FOM2"],
        "val_fom": val_s2_o2e["fom2"],
        "fom_ratio": r2_o2e,
        "n_train": df_odd_v.loc[df_odd_v["FOM2"].idxmax(), "n_events"],
        "n_val": val_s2_o2e["n_events"],
        "cut_stability": s2_label,
    },
    {
        "Set": "Set2_LowYield",
        "FOM": "FOM2=S/sqrt(S+B)",
        "direction": "even_train",
        "chi2": best2_even["chi2"],
        "fdchi2": best2_even["fdchi2"],
        "ipchi2": best2_even["ipchi2"],
        "pt": best2_even["pt"],
        "pid": PID_CUT,
        "train_fom": best2_even["FOM2"],
        "val_fom": val_s2_e2o["fom2"],
        "fom_ratio": r2_e2o,
        "n_train": df_even_v.loc[df_even_v["FOM2"].idxmax(), "n_events"],
        "n_val": val_s2_e2o["n_events"],
        "cut_stability": s2_label,
    },
    {
        "Set": "Set2_LowYield",
        "FOM": "FOM2=S/sqrt(S+B)",
        "direction": "full_reference",
        "chi2": ref2["bu_dtf_chi2"],
        "fdchi2": ref2["bu_fdchi2"],
        "ipchi2": ref2["bu_ipchi2"],
        "pt": ref2["bu_pt"],
        "pid": PID_CUT,
        "train_fom": ref2["FOM_value"],
        "val_fom": np.nan,
        "fom_ratio": np.nan,
        "n_train": ref2.get("n_events", np.nan),
        "n_val": np.nan,
        "cut_stability": "REFERENCE",
    },
]
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(summary_output, index=False)
print(f"\nSaved: {summary_output}")

# ---------------------------------------------------------------------------
# Phase 4: Comparison report plot
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("Phase 4: Comparison report plot")
print(f"{'=' * 65}")

fig = plt.figure(figsize=(17, 11))
fig.suptitle(
    "Cross-Validation Study: Fit-Based Optimizer Bias Check\n"
    f"(PID > {PID_CUT}  |  {n_base:,} total events  →  "
    f"{len(all_data_odd):,} odd + {len(all_data_even):,} even)",
    fontsize=13,
    fontweight="bold",
)

gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.38)

BAR_COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#8c8c8c"]
BAR_LABELS = [
    "Odd (train)",
    "Even (val,\nodd-cuts)",
    "Even (train)",
    "Odd (val,\neven-cuts)",
    "Full (ref)",
]


def fom_bar(ax, values, title, ylabel):
    """Draw a FOM bar chart with value labels above each bar."""
    bars = ax.bar(
        BAR_LABELS,
        values,
        color=BAR_COLORS,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    max_v = max((v for v in values if not np.isnan(v)), default=1.0)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, max_v * 1.30)
    ax.grid(axis="y", alpha=0.3)


def ratio_annotation(ax, ro2e, re2o):
    """Annotate the bar chart with val/train FOM ratios."""
    ro2e_str = f"{ro2e:.3f}" if not np.isnan(ro2e) else "N/A"
    re2o_str = f"{re2o:.3f}" if not np.isnan(re2o) else "N/A"
    ax.text(
        0.5,
        0.02,
        f"FOM_val/FOM_train:  odd→even = {ro2e_str},   even→odd = {re2o_str}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="#222222",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.75),
    )


def cut_table(ax, odd_row, even_row, ref_row, stability_label, stability_ok):
    """Draw cut comparison table as a matplotlib table."""
    ax.axis("off")
    col_labels = ["Variable", "Odd-train", "Even-train", "Full (ref)"]
    var_names = ["chi2 <", "FDCHI2 >", "IPCHI2 <", "PT > [MeV/c]", "PID >"]
    keys = ["chi2", "fdchi2", "ipchi2", "pt"]
    fmts = [".0f", ".0f", ".1f", ".0f"]

    table_vals = []
    cell_clrs = []
    for key, fmt, var in zip(keys, fmts, var_names[:4]):
        o_val = format(float(odd_row[key]), fmt)
        e_val = format(float(even_row[key]), fmt)
        r_val = format(float(ref_row[key]), fmt)
        table_vals.append([var, o_val, e_val, r_val])

        # Colour coding
        def col(v):
            if v == r_val:
                return "#d4edda"  # green: matches ref
            if o_val == e_val:
                return "#fff3cd"  # yellow: odd=even but diff from ref
            return "#f8d7da"  # red: odd≠even

        cell_clrs.append(["#e8e8e8", col(o_val), col(e_val), "#e8f4f8"])

    # PID row (always the same)
    table_vals.append(["PID >", f"{PID_CUT}", f"{PID_CUT}", f"{PID_CUT}"])
    cell_clrs.append(["#e8e8e8", "#d4edda", "#d4edda", "#e8f4f8"])

    tbl = ax.table(
        cellText=table_vals,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_clrs,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    s_color = "darkgreen" if stability_ok else "firebrick"
    ax.set_title(
        f"Optimal Cuts Comparison  [{stability_label}]",
        fontsize=10,
        color=s_color,
    )


# --- Set1 bar chart ---
ax0 = fig.add_subplot(gs[0, 0])
vals_fom1 = [
    best1_odd["FOM1"],
    val_s1_o2e["fom1"],
    best1_even["FOM1"],
    val_s1_e2o["fom1"],
    ref1["FOM_value"],
]
fom_bar(
    ax0,
    vals_fom1,
    "Set1 (High-yield: J/psi + eta_c)\nFOM1 = (N_jpsi + N_etac) / sqrt(N_bkg)",
    "FOM1",
)
ratio_annotation(ax0, r1_o2e, r1_e2o)

# --- Set1 cut table ---
ax1 = fig.add_subplot(gs[0, 1])
ref1_dict = {
    "chi2": ref1["bu_dtf_chi2"],
    "fdchi2": ref1["bu_fdchi2"],
    "ipchi2": ref1["bu_ipchi2"],
    "pt": ref1["bu_pt"],
}
cut_table(ax1, best1_odd, best1_even, ref1_dict, s1_label, s1_exact or s1_close)

# --- Set2 bar chart ---
ax2 = fig.add_subplot(gs[1, 0])
vals_fom2 = [
    best2_odd["FOM2"],
    val_s2_o2e["fom2"],
    best2_even["FOM2"],
    val_s2_e2o["fom2"],
    ref2["FOM_value"],
]
fom_bar(
    ax2,
    vals_fom2,
    "Set2 (Low-yield: chi_c0/1 + eta_c(2S))\nFOM2 = S2 / sqrt(S2 + N_bkg)",
    "FOM2",
)
ratio_annotation(ax2, r2_o2e, r2_e2o)

# --- Set2 cut table ---
ax3 = fig.add_subplot(gs[1, 1])
ref2_dict = {
    "chi2": ref2["bu_dtf_chi2"],
    "fdchi2": ref2["bu_fdchi2"],
    "ipchi2": ref2["bu_ipchi2"],
    "pt": ref2["bu_pt"],
}
cut_table(ax3, best2_odd, best2_even, ref2_dict, s2_label, s2_exact or s2_close)

# Legend for bar colours
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=BAR_COLORS[0], label="Training (odd half)"),
    Patch(facecolor=BAR_COLORS[1], label="Validation (even half, odd-trained cuts)"),
    Patch(facecolor=BAR_COLORS[2], label="Training (even half)"),
    Patch(facecolor=BAR_COLORS[3], label="Validation (odd half, even-trained cuts)"),
    Patch(facecolor=BAR_COLORS[4], label="Full dataset reference"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=8.5,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.04),
)

fig.savefig(report_output, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {report_output}")

# ---------------------------------------------------------------------------
# Final summary print
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'=' * 65}")
print(
    f"Split:  {len(all_data_odd):,} odd  +  {len(all_data_even):,} even  events  (total {n_base:,})"
)
print(f"Scan failures:  {n_fail_odd} (odd),  {n_fail_even} (even)  out of {TOTAL_GRID} each")

print("\n--- Set1 (FOM1 = S/sqrt(B), J/psi + eta_c) ---")
print(
    f"  Full  ref:    chi2<{ref1['bu_dtf_chi2']:.0f}  FD>{ref1['bu_fdchi2']:.0f}  "
    f"IP<{ref1['bu_ipchi2']:.1f}  PT>{ref1['bu_pt']:.0f}   FOM1={ref1['FOM_value']:.3f}"
)
print(
    f"  Odd  train:   chi2<{best1_odd['chi2']:.0f}  FD>{best1_odd['fdchi2']:.0f}  "
    f"IP<{best1_odd['ipchi2']:.1f}  PT>{best1_odd['pt']:.0f}   FOM1={best1_odd['FOM1']:.3f}"
)
print(
    f"  Even train:   chi2<{best1_even['chi2']:.0f}  FD>{best1_even['fdchi2']:.0f}  "
    f"IP<{best1_even['ipchi2']:.1f}  PT>{best1_even['pt']:.0f}   FOM1={best1_even['FOM1']:.3f}"
)
print(f"  Val odd→even: FOM1={val_s1_o2e['fom1']:.3f}   ratio={r1_o2e:.3f}")
print(f"  Val even→odd: FOM1={val_s1_e2o['fom1']:.3f}   ratio={r1_e2o:.3f}")
print(f"  Cut stability: {s1_label}")

print("\n--- Set2 (FOM2 = S2/sqrt(S2+B), chi_c0/1 + eta_c(2S)) ---")
print(
    f"  Full  ref:    chi2<{ref2['bu_dtf_chi2']:.0f}  FD>{ref2['bu_fdchi2']:.0f}  "
    f"IP<{ref2['bu_ipchi2']:.1f}  PT>{ref2['bu_pt']:.0f}   FOM2={ref2['FOM_value']:.3f}"
)
print(
    f"  Odd  train:   chi2<{best2_odd['chi2']:.0f}  FD>{best2_odd['fdchi2']:.0f}  "
    f"IP<{best2_odd['ipchi2']:.1f}  PT>{best2_odd['pt']:.0f}   FOM2={best2_odd['FOM2']:.3f}"
)
print(
    f"  Even train:   chi2<{best2_even['chi2']:.0f}  FD>{best2_even['fdchi2']:.0f}  "
    f"IP<{best2_even['ipchi2']:.1f}  PT>{best2_even['pt']:.0f}   FOM2={best2_even['FOM2']:.3f}"
)
print(f"  Val odd→even: FOM2={val_s2_o2e['fom2']:.3f}   ratio={r2_o2e:.3f}")
print(f"  Val even→odd: FOM2={val_s2_e2o['fom2']:.3f}   ratio={r2_e2o:.3f}")
print(f"  Cut stability: {s2_label}")

print("\nOutput files:")
for p in [
    csv_odd_output,
    csv_even_output,
    summary_output,
    report_output,
    val_s1_o2e_output,
    val_s1_e2o_output,
    val_s2_o2e_output,
    val_s2_e2o_output,
]:
    print(f"  {p}")
