#!/usr/bin/env bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bphysics
echo "Conda environment 'bphysics' activated."

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/../
echo "PYTHONPATH set to include the current directory."

# Define alias for running the analysis
alias br_run='python fit.py && python calculate_eff.py && python br.py && python generate_result.py'
echo "Environment ready – PYTHONPATH set, alias 'br_run' defined."

# Echo environment setup instructions
# if bphysics environment is not activated or installed, print activation and installation instructions
if [ ! -z "$(conda info --envs | grep bphysics | grep active)" ]; then
    echo "Conda environment 'bphysics' is not activated."
    echo "Please activate it by running:"
    echo "conda activate bphysics"
    echo "Or create it by running:"
    echo "conda create -n bphysics python=3.10 -y"
    echo "conda activate bphysics"
    echo "conda install -c conda-forge root numpy awkward uproot pyyaml matplotlib tqdm -y"
    echo "pip install -r requirements.txt"

    exit 1
fi

echo "Run 'br_run' to execute the entire analysis."
echo "Or run each script separately: fit.py, calculate_eff.py, br.py, generate_result.py"
echo "I'm dedicated to a reproducible analysis. Although I code badly"
echo "Good luck! Mohamed Elashri"
