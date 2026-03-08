import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def extract_tmva_data():
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

    # ROC curve
    roc = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_rejBvsS")
    if roc:
        n_bins = roc.GetNbinsX()
        fpr = np.zeros(n_bins)
        tpr = np.zeros(n_bins)
        for i in range(1, n_bins + 1):
            tpr[i - 1] = roc.GetBinCenter(i)  # signal efficiency
            fpr[i - 1] = 1.0 - roc.GetBinContent(i)  # 1 - background rejection

        # Reverse to sort by threshold (standard sklearn format)
        fpr = fpr[::-1]
        tpr = tpr[::-1]
    else:
        fpr, tpr = None, None
        print("ROC curve not found")

    # Overtraining Plot distributions
    sig_train_hist = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_Train_S")
    bkg_train_hist = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_Train_B")
    sig_test_hist = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_S")
    bkg_test_hist = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_B")

    def hist_to_arrays(h):
        if not h:
            return None, None
        n_bins = h.GetNbinsX()
        bins = np.zeros(n_bins)
        vals = np.zeros(n_bins)
        for i in range(1, n_bins + 1):
            bins[i - 1] = h.GetBinCenter(i)
            vals[i - 1] = h.GetBinContent(i)
        return bins, vals

    sig_train_bins, sig_train_vals = hist_to_arrays(sig_train_hist)
    bkg_train_bins, bkg_train_vals = hist_to_arrays(bkg_train_hist)
    sig_test_bins, sig_test_vals = hist_to_arrays(sig_test_hist)
    bkg_test_bins, bkg_test_vals = hist_to_arrays(bkg_test_hist)

    # AUC
    auc_val = 0.943  # Hardcoded from our log extraction

    np.savez(
        out_dir / "tmva_raw.npz",
        fpr=fpr,
        tpr=tpr,
        roc_auc=np.array([auc_val]),
        sig_train_bins=sig_train_bins,
        sig_train_vals=sig_train_vals,
        bkg_train_bins=bkg_train_bins,
        bkg_train_vals=bkg_train_vals,
        sig_test_bins=sig_test_bins,
        sig_test_vals=sig_test_vals,
        bkg_test_bins=bkg_test_bins,
        bkg_test_vals=bkg_test_vals,
    )

    print(f"Saved TMVA raw data to {out_dir / 'tmva_raw.npz'}")
    f.Close()


if __name__ == "__main__":
    extract_tmva_data()
