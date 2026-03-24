"""
TMVA Implementation for BDT Optimization Study

This script mirrors the Python-based MVA optimization but uses ROOT's TMVA.
It leverages the exact same data preparation steps for a fair comparison.
"""

import logging
import os
import sys
from pathlib import Path

import uproot

# Ensure script is run from tmva directory so dataset is created inside tmva folder
script_dir = Path(__file__).resolve().parent
os.chdir(str(script_dir))

# Add project root to sys.path to allow correct module resolution
project_root = script_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# The Python MVA modules are now in the 'mva' directory, next to 'tmva'
mva_opt_dir = script_dir.parent / "mva"
if str(mva_opt_dir) not in sys.path:
    sys.path.insert(0, str(mva_opt_dir))

from config_loader import StudyConfig
from data_preparation import load_and_prepare_data
from tmva_utils.presentation_utils import extract_summary, save_tmva_plots_with_root

# Setup dual output to file and stdout
output_dir = script_dir.parent / "output" / "tmva"
output_dir.mkdir(parents=True, exist_ok=True)
report_file = output_dir / "tmva_optimization_report.txt"


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.file.close()


sys.stdout = DualOutput(report_file)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.FileHandler(report_file, mode="a"), logging.StreamHandler(sys.stdout.stdout)],
)
logger = logging.getLogger(__name__)


def export_data_to_root(ml_data, out_file_path):
    logger.info(f"Exporting data to {out_file_path} for TMVA...")

    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]
    w_train = ml_data["w_train"]

    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]
    w_test = ml_data["w_test"]

    features = ml_data["features"]

    sig_train = {f: X_train[y_train == 1][:, i] for i, f in enumerate(features)}
    sig_train["weight"] = w_train[y_train == 1]

    bkg_train = {f: X_train[y_train == 0][:, i] for i, f in enumerate(features)}
    bkg_train["weight"] = w_train[y_train == 0]

    sig_test = {f: X_test[y_test == 1][:, i] for i, f in enumerate(features)}
    sig_test["weight"] = w_test[y_test == 1]

    bkg_test = {f: X_test[y_test == 0][:, i] for i, f in enumerate(features)}
    bkg_test["weight"] = w_test[y_test == 0]

    with uproot.recreate(out_file_path) as f:
        f["TrainTree_Signal"] = sig_train
        f["TrainTree_Background"] = bkg_train
        f["TestTree_Signal"] = sig_test
        f["TestTree_Background"] = bkg_test

    logger.info("Export complete.")


def run_tmva(root_file_path, output_dir, features):
    import ROOT

    ROOT.TMVA.Tools.Instance()

    raw_log_path = output_dir / "tmva_raw.log"
    logger.info(f"Running TMVA... Standard output redirected to {raw_log_path}")

    # Redirect ROOT's C++ stdout to capture TMVA evaluation logs
    ROOT.gSystem.RedirectOutput(str(raw_log_path), "w")

    out_file = ROOT.TFile(str(output_dir / "TMVA_Output.root"), "RECREATE")
    factory = ROOT.TMVA.Factory(
        "TMVAClassification",
        out_file,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification",
    )

    dataloader = ROOT.TMVA.DataLoader("dataset")

    for f in features:
        dataloader.AddVariable(f, "F")

    data_file = ROOT.TFile(str(root_file_path))
    t_sig_train = data_file.Get("TrainTree_Signal")
    t_bkg_train = data_file.Get("TrainTree_Background")
    t_sig_test = data_file.Get("TestTree_Signal")
    t_bkg_test = data_file.Get("TestTree_Background")

    dataloader.AddSignalTree(t_sig_train, 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddSignalTree(t_sig_test, 1.0, ROOT.TMVA.Types.kTesting)
    dataloader.AddBackgroundTree(t_bkg_train, 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddBackgroundTree(t_bkg_test, 1.0, ROOT.TMVA.Types.kTesting)

    dataloader.SetSignalWeightExpression("weight")
    dataloader.SetBackgroundWeightExpression("weight")

    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), ROOT.TCut(""), "SplitMode=Random:NormMode=NumEvents:!V"
    )

    bdt_options = (
        "!H:!V:NTrees=350:MinNodeSize=2.5%:MaxDepth=6:"
        "BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:"
        "BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20"
    )
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", bdt_options)

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    out_file.Close()
    data_file.Close()

    # Restore C++ stdout
    ROOT.gSystem.RedirectOutput(0)

    logger.info("TMVA evaluation complete.")


def main():
    logger.info("Starting TMVA Optimization Study...")

    # Change working directory temporarily to load the config correctly
    os.chdir(str(mva_opt_dir))
    try:
        config = StudyConfig()
        logger.info("Loading and preparing data...")
        ml_data = load_and_prepare_data(config)
    finally:
        os.chdir(str(script_dir))

    root_file_path = output_dir / "tmva_input.root"
    export_data_to_root(ml_data, root_file_path)

    try:
        import ROOT

        ROOT.gSystem.Load("libTMVA")
        if not hasattr(ROOT, "TMVA"):
            raise ImportError("TMVA not found in current ROOT installation.")

        run_tmva(root_file_path, output_dir, ml_data["features"])

        logger.info("Step: Generating Presentation Artifacts (Phase 2, 3, 4) for TMVA...")
        save_tmva_plots_with_root(output_dir / "TMVA_Output.root", output_dir / "plots")
        extract_summary(output_dir / "tmva_raw.log", output_dir / "tmva_summary.md")

        with open(script_dir.parent / "tmva_optimization_completed.txt", "w") as f:
            f.write("Completed\n")

    except ImportError as e:
        logger.error(f"Cannot run TMVA: {e}")
        logger.warning(
            "Please run this script in an environment with full ROOT + TMVA (e.g., conda bphysics)."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during TMVA execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
