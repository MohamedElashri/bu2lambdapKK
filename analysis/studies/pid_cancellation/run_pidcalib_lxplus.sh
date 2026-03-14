#!/bin/bash
# run_pidcalib_lxplus.sh
# Run this script on LXPLUS to generate PID efficiency histograms using PIDCalib2

# Stop on errors
set -e

echo "Setting up LHCb environment..."
source /cvmfs/lhcb.cern.ch/lib/LbEnv.sh

echo "Running PIDCalib2 make_eff_hists inside lb-conda environment..."
lb-conda pidcalib bash -c "
mkdir -p pidcalib_output

# Note: For 2018 (Run 2) data, PIDCalib2 uses the MC15TuneV1_ProbNNx alias instead of just ProbNNx.

echo 'Processing Protons (Magnet UP)...'
pidcalib2.make_eff_hists -s Turbo18 -m up -p P -i 'MC15TuneV1_ProbNNp > 0.05' -b P -b ETA -o pidcalib_output

echo 'Processing Protons (Magnet DOWN)...'
pidcalib2.make_eff_hists -s Turbo18 -m down -p P -i 'MC15TuneV1_ProbNNp > 0.05' -b P -b ETA -o pidcalib_output

echo 'Processing Kaons (Magnet UP)...'
# Since the analysis cut is a product (h1_ProbNNk * h2_ProbNNk > 0.05), we use sqrt(0.05) ~ 0.224 as a representative single-track cut for the cancellation study
pidcalib2.make_eff_hists -s Turbo18 -m up -p K -i 'MC15TuneV1_ProbNNk > 0.224' -b P -b ETA -o pidcalib_output

echo 'Processing Kaons (Magnet DOWN)...'
pidcalib2.make_eff_hists -s Turbo18 -m down -p K -i 'MC15TuneV1_ProbNNk > 0.224' -b P -b ETA -o pidcalib_output

echo 'Creating a tarball of the output...'
tar -czvf pidcalib_results.tar.gz pidcalib_output/

echo 'Done! The histograms are saved in pidcalib_output/ and packaged into pidcalib_results.tar.gz'
"
