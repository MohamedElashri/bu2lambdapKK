import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def extract_tmva_correlations():
    import ROOT

    script_dir = Path(__file__).resolve().parent
    tmva_out_path = script_dir / "analysis_output" / "TMVA_Output.root"
    out_dir = script_dir.parent / "comparison" / "raw_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tmva_out_path.exists():
        print(f"File {tmva_out_path} not found.")
        return

    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile(str(tmva_out_path))

    dataset_dir = f.Get("dataset")
    if not dataset_dir:
        print("dataset dir not found")
        return

    corr_sig_hist = dataset_dir.Get("CorrelationMatrixS")
    corr_bkg_hist = dataset_dir.Get("CorrelationMatrixB")

    if not corr_sig_hist or not corr_bkg_hist:
        print("Correlation matrices not found in TMVA output")
        return

    n_vars = corr_sig_hist.GetNbinsX()

    features = []
    for i in range(1, n_vars + 1):
        features.append(corr_sig_hist.GetXaxis().GetBinLabel(i))

    corr_sig = np.zeros((n_vars, n_vars))
    corr_bkg = np.zeros((n_vars, n_vars))

    for i in range(1, n_vars + 1):
        for j in range(1, n_vars + 1):
            corr_sig[i - 1, j - 1] = corr_sig_hist.GetBinContent(i, j)
            # TMVA stores it transposed relative to standard matrix indexing usually,
            # but let's just dump it exactly as ROOT returns it
            corr_bkg[i - 1, j - 1] = corr_bkg_hist.GetBinContent(i, j)

    np.savez(
        out_dir / "tmva_correlations.npz", corr_sig=corr_sig, corr_bkg=corr_bkg, features=features
    )

    print(f"Saved TMVA correlation data to {out_dir / 'tmva_correlations.npz'}")
    f.Close()


if __name__ == "__main__":
    extract_tmva_correlations()
