#!/usr/bin/env bash
# source this once per shell
echo "To run the analysis, use the command: br_run"
echo "Setting up the environment for the analysis..."
## activate bphysics from conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bphysics
echo "Conda environment 'bphysics' activated."
## set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/../
echo "PYTHONPATH set to include the current directory."
## set up the alias
alias br_run='python fit_yields.py && python efficiency.py && python br_estimate.py'
echo "Environment ready – PYTHONPATH set, alias 'br_run' defined."
