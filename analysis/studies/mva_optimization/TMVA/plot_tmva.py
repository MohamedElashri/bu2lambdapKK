import os
import re
import sys
from pathlib import Path

# Add project root to sys.path to allow correct module resolution
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def extract_summary(raw_log_path, md_out_path):
    if not os.path.exists(raw_log_path):
        print(f"Log file not found: {raw_log_path}")
        return

    with open(raw_log_path, "r") as f:
        content = f.read()

    # Extract variable ranking (make regex more robust for different outputs)
    ranking_match = re.search(
        r"Ranking result \(top variable is best ranked\)\s*\n\s*[-]+\n(.*?)\n\s*[-]+",
        content,
        re.DOTALL,
    )

    # Extract ROC integral
    roc_match = re.search(r"dataset\s+BDT\s+:\s+([0-9.]+)", content)

    # Extract overtraining
    # Look for the exact line after the header
    overtrain_match = re.search(
        r"dataset\s+BDT\s+:\s+([0-9.]+\s+\([0-9.]+\)\s+[0-9.]+\s+\([0-9.]+\)\s+[0-9.]+\s+\([0-9.]+\))",
        content,
    )

    with open(md_out_path, "w") as f:
        f.write("# TMVA Optimization Summary\n\n")

        if roc_match:
            f.write(f"## Performance\n- **ROC Integral (AUC):** {roc_match.group(1)}\n\n")

        if overtrain_match:
            vals = overtrain_match.group(1).replace("(", "").replace(")", "").split()
            if len(vals) >= 6:
                f.write("## Overtraining Check (Signal Efficiency)\n")
                f.write("| Background Rejection | Test Sample | Train Sample |\n")
                f.write("|---|---|---|\n")
                f.write(f"| B=0.01 | {vals[0]} | {vals[1]} |\n")
                f.write(f"| B=0.10 | {vals[2]} | {vals[3]} |\n")
                f.write(f"| B=0.30 | {vals[4]} | {vals[5]} |\n\n")

        if ranking_match:
            # Clean up the ranking table format
            ranking_text = ranking_match.group(1)
            lines = [
                line.strip().replace(":", "|").split("|")
                for line in ranking_text.split("\n")
                if line.strip()
            ]

            f.write("## Feature Importance Ranking\n")
            f.write("| Rank | Variable | Importance |\n")
            f.write("|---|---|---|\n")
            for line in lines:
                if len(line) >= 3:
                    f.write(f"| {line[0].strip()} | {line[1].strip()} | {line[2].strip()} |\n")
            f.write("\n")

        f.write("## Generated Plots\n\n")
        f.write("![ROC Curve](plots/tmva_roc_curve.png)\n\n")
        f.write("![Overtraining Plot](plots/tmva_overtraining_plot.png)\n")


def save_tmva_plots_with_root(tmva_out_path, plot_dir):
    import ROOT

    tmva_out_path = Path(tmva_out_path)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not tmva_out_path.exists():
        print(f"File {tmva_out_path} not found.")
        return

    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile(str(tmva_out_path))

    dataset_dir = f.Get("dataset")
    if not dataset_dir:
        return

    bdt_dir = dataset_dir.Get("Method_BDT/BDT")
    if not bdt_dir:
        return

    # 1. ROC Curve
    roc = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_rejBvsS")
    if roc:
        c1 = ROOT.TCanvas("c1", "ROC Curve", 800, 600)
        c1.SetGrid()
        roc.SetLineColor(ROOT.kBlue)
        roc.SetLineWidth(2)
        roc.SetTitle("TMVA BDT ROC Curve")
        roc.GetXaxis().SetTitle("Signal Efficiency")
        roc.GetYaxis().SetTitle("Background Rejection (1 - effB)")
        roc.Draw("L")
        c1.SaveAs(str(plot_dir / "tmva_roc_curve.pdf"))
        c1.SaveAs(str(plot_dir / "tmva_roc_curve.png"))

    # 2. Overtraining Plot
    sig_train = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_Train_S")
    bkg_train = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_Train_B")
    sig_test = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_S")
    bkg_test = dataset_dir.Get("Method_BDT/BDT/MVA_BDT_B")

    if sig_train and bkg_train and sig_test and bkg_test:
        c2 = ROOT.TCanvas("c2", "Overtraining Plot", 800, 600)

        sig_train = sig_train.Clone("sig_train")
        bkg_train = bkg_train.Clone("bkg_train")
        sig_test = sig_test.Clone("sig_test")
        bkg_test = bkg_test.Clone("bkg_test")

        if sig_train.Integral() > 0:
            sig_train.Scale(1.0 / sig_train.Integral())
        if bkg_train.Integral() > 0:
            bkg_train.Scale(1.0 / bkg_train.Integral())
        if sig_test.Integral() > 0:
            sig_test.Scale(1.0 / sig_test.Integral())
        if bkg_test.Integral() > 0:
            bkg_test.Scale(1.0 / bkg_test.Integral())

        max_y = (
            max(
                sig_train.GetMaximum(),
                bkg_train.GetMaximum(),
                sig_test.GetMaximum(),
                bkg_test.GetMaximum(),
            )
            * 1.2
        )

        sig_test.SetMaximum(max_y)
        sig_test.SetTitle("TMVA BDT Output: Test vs Train")
        sig_test.GetXaxis().SetTitle("BDT Response")
        sig_test.GetYaxis().SetTitle("Normalized Units")

        bkg_train.SetLineColor(ROOT.kRed)
        bkg_train.SetFillColor(ROOT.kRed)
        bkg_train.SetFillStyle(3005)

        sig_train.SetLineColor(ROOT.kBlue)
        sig_train.SetFillColor(ROOT.kBlue)
        sig_train.SetFillStyle(3004)

        sig_test.SetMarkerColor(ROOT.kBlue)
        sig_test.SetMarkerStyle(20)
        sig_test.SetLineColor(ROOT.kBlue)
        sig_test.SetLineWidth(2)

        bkg_test.SetMarkerColor(ROOT.kRed)
        bkg_test.SetMarkerStyle(20)
        bkg_test.SetLineColor(ROOT.kRed)
        bkg_test.SetLineWidth(2)

        sig_test.Draw("EP")
        bkg_test.Draw("EP SAME")
        sig_train.Draw("HIST SAME")
        bkg_train.Draw("HIST SAME")
        sig_test.Draw("EP SAME")
        bkg_test.Draw("EP SAME")

        leg = ROOT.TLegend(0.65, 0.7, 0.9, 0.9)
        leg.AddEntry(sig_test, "Signal (Test)", "p")
        leg.AddEntry(bkg_test, "Background (Test)", "p")
        leg.AddEntry(sig_train, "Signal (Train)", "f")
        leg.AddEntry(bkg_train, "Background (Train)", "f")
        leg.Draw()

        ks_sig = sig_test.KolmogorovTest(sig_train)
        ks_bkg = bkg_test.KolmogorovTest(bkg_train)

        t = ROOT.TLatex()
        t.SetNDC()
        t.SetTextSize(0.035)
        t.DrawLatex(0.2, 0.85, f"KS p-value (Sig): {ks_sig:.3f}")
        t.DrawLatex(0.2, 0.80, f"KS p-value (Bkg): {ks_bkg:.3f}")

        c2.SaveAs(str(plot_dir / "tmva_overtraining_plot.pdf"))
        c2.SaveAs(str(plot_dir / "tmva_overtraining_plot.png"))

    f.Close()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "analysis_output"

    save_tmva_plots_with_root(out_dir / "TMVA_Output.root", out_dir / "plots")
    extract_summary(out_dir / "tmva_raw.log", out_dir / "tmva_summary.md")
