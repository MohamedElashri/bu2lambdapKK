#!/usr/bin/env bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bphysics
echo "Conda environment 'bphysics' activated."
export PYTHONPATH=$PYTHONPATH:$(pwd)/../
echo "PYTHONPATH set to include the current directory."
alias br_run='python fit.py && python calculate_eff.py && python br.py && python generate_result.py'
echo "Environment ready – PYTHONPATH set, alias 'br_run' defined."
