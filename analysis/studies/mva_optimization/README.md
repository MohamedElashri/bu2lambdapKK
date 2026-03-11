# TMVA Optimization Study

This directory contains two ways to perform the MVA (Multivariate Analysis) optimization for the B⁺ → Λ̄pK⁻K⁺ decay.

## 1. Modern Python Approach (Default)
The standard Snakefile and `main.py` run an optimization using XGBoost, LightGBM, and CatBoost. It utilizes Python-native libraries (`scikit-learn`, `catboost`, etc.) and Awkward Array for data handling.

To run:
```bash
uv run snakemake -j1
```
Or directly:
```bash
uv run python main.py
```

## 2. Classic TMVA Approach (Particle Physics Standard)
We have also implemented the exact same study using ROOT's TMVA (Toolkit for Multivariate Data Analysis), which is the traditional tool used in high energy physics.

The script `run_tmva.py` uses the exact same data preparation pipeline (`data_preparation.py` and `clean_data_loader.py`) to apply the baseline cuts and sideband selection. It then exports the data to a `.root` file and runs TMVA on it.

### How to run the TMVA study
Because the current `uv` environment relies on the PyPI `root` package which is currently missing the `libTMVA.so` compiled library, you need to run `run_tmva.py` in an environment where a full ROOT installation is available (e.g., your conda `bphysics` environment).

**Step 1: Export data, run TMVA, and generate plots**
Activate an environment with full ROOT + TMVA support (The wheels published on PyPi for `ROOT` does not have `TMVA`), then run:
```bash
cd tmva
conda activate bphysics
python run_tmva.py
```
*(Note: Do not use `uv run` here. If you are missing `scikit-learn` in your bphysics environment, run `pip install scikit-learn` first.)*

**Step 2: View the TMVA Plots**
The script will automatically generate PNG and PDF plots (ROC curve, Overtraining check) in `analysis_output/plots/tmva/`.

If you prefer the classic interactive TMVA GUI, you can run:
```bash
root -l -e 'TMVA::TMVAGui("analysis_output/tmva/TMVA_Output.root")'
```

### Comparison
By maintaining both pipelines, we ensure that:
1. We can validate the new Python-based Machine Learning models against the field's standard TMVA results.
2. The exact same training data, testing data, weights, and features are passed to both frameworks, allowing for an apples-to-apples comparison of AUC, ROC curves, and feature importance.
