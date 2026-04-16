"""
Data mass fit plots for the B+ → Λ̄pK⁻K⁺ analysis note.

Produces:
  figs/LambdaLL/fit/fit_todata_signal.pdf
  figs/LambdaDD/fit/fit_todata_signal.pdf

Strategy: use existing pipeline mass fit output (generated/output/pipeline/mva/high_yield/)
and copy to the analysis note figure directory with the reference-matching filename.

The pipeline already runs full RooFit mass fits; we just route the output correctly.
Re-running a fresh fit with the exact same MassFitter + data is also supported via
the --refit flag.

Run from analysis/ directory:
    uv run python presentation/ana_note_plots/scripts/plot_datafit.py
    uv run python presentation/ana_note_plots/scripts/plot_datafit.py --refit   (re-run fits)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPTS_DIR.resolve().parents[3]  # bu2lambdapKK root
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.resolve().parents[2]))  # analysis/ for modules.*

from modules.plot_utils import figs_path
from modules.presentation_config import get_presentation_config

PRESENTATION = get_presentation_config()
PIPELINE_OUT = PRESENTATION.pipeline_output_dir / "high_yield"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def copy_pipeline_fit(cat: str, year_key: str = "combined"):
    """
    Copy pipeline fit PDF to analysis note figure directory.

    Parameters
    ----------
    cat      : "LL" or "DD"
    year_key : "combined" (Run 2) or "2016"/"2017"/"2018"
    """
    src = PIPELINE_OUT / cat / "plots" / "fits" / f"mass_fit_{year_key}.pdf"
    if not src.exists():
        log.warning(f"  Pipeline output not found: {src}")
        log.warning("  Run 'uv run snakemake mass_fitting -j4' first")
        return False

    if year_key == "combined":
        dst_name = "fit_todata_signal.pdf"
    else:
        dst_name = f"fit_todata_signal_{year_key}.pdf"

    dst = figs_path(cat, "fit", dst_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log.info(f"  Copied: {src.name} → {dst}")
    return True


def refit_and_plot(cat: str):
    """
    Re-run mass fit using MassFitter and save to note figure directory.
    Uses the existing pipeline configuration and cached data.
    """
    try:
        from modules.cache_manager import CacheManager
        from modules.config_loader import StudyConfig
        from modules.mass_fitter import MassFitter
    except ImportError:
        sys.path.insert(0, str(SCRIPTS_DIR.parents[3]))
        from modules.cache_manager import CacheManager
        from modules.config_loader import StudyConfig
        from modules.mass_fitter import MassFitter

    output_dir = str(PRESENTATION.pipeline_output_dir)
    cache_dir = str(PRESENTATION.pipeline_output_dir / "cache")

    config = StudyConfig.from_dir(PRESENTATION.analysis_dir / "config", output_dir=output_dir)
    cache = CacheManager(cache_dir=cache_dir)

    # Try to load cached data (high_yield branch)
    cut_deps = cache.compute_dependencies(
        config_files=config.config_paths(),
        code_files=[PRESENTATION.analysis_dir / "scripts" / "apply_cuts.py"],
    )
    data_dict = cache.load(f"high_yield_{cat}_final_data", dependencies=cut_deps)

    if data_dict is None:
        log.error(f"  Cached data not found for high_yield_{cat}. Run pipeline first.")
        return

    # Plot directory: note figure location
    plot_dir = figs_path(cat, "fit").parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  Running MassFitter for {cat}...")
    fitter = MassFitter(config=config)
    fitter.perform_fit(
        data_dict,
        fit_combined=True,
        plot_dir=plot_dir,
        fit_label=f"Run 2 / {cat}",
    )

    # Rename output to note convention
    combined_pdf = plot_dir / "mass_fit_combined.pdf"
    if combined_pdf.exists():
        dst = plot_dir / "fit_todata_signal.pdf"
        shutil.copy2(combined_pdf, dst)
        log.info(f"  Saved: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refit", action="store_true", help="Re-run mass fits instead of copying pipeline output"
    )
    args = parser.parse_args()

    for cat in ("LL", "DD"):
        log.info(f"=== Category: Lambda{cat} ===")
        if args.refit:
            refit_and_plot(cat)
        else:
            # Copy Run2 combined fit
            ok = copy_pipeline_fit(cat, "combined")
            if ok:
                # Also copy per-year fits for appendix
                for yr in ("2016", "2017", "2018"):
                    copy_pipeline_fit(cat, yr)

    log.info("=== Done. ===")


if __name__ == "__main__":
    main()
