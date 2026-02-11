# Standalone Studies

Self-contained studies that are **not** part of the main analysis pipeline DAG.

Each study lives in its own subdirectory with its own `Snakefile` and scripts.
Studies may depend on cached outputs from the main pipeline (e.g. Step 2 data/MC)
but are never triggered by `snakemake -j1` in the main `analysis/` directory.

## Directory structure

```
analysis/studies/
├── README.md
├── <study_name>/
│   ├── Snakefile          # standalone Snakefile for this study
│   ├── <study_name>.py    # main study script
│   ├── README.md          # study-specific documentation / results summary
│   └── output/            # study outputs (gitignored)
└── <another_study>/
    └── ...
```

## Running a study

```bash
cd analysis/studies/<study_name>
uv run snakemake -j1
```

Each study's `Snakefile` references the main analysis `config/` and `cache/`
directories via relative paths, so it must be run from within its own directory.

## Adding a new study

1. Create `analysis/studies/<study_name>/`
2. Add a `Snakefile` with at least one rule (see `fom_comparison/` for a template)
3. Add your study script(s)
4. Add a `README.md` documenting motivation, methodology, and results

## Current studies

| Study | Description |
|-------|-------------|
| `fom_comparison` | Compare S/√B vs S/(√S+√B) FoM formulas across charmonium states |
| `cumulative_cut_efficiency` | Cumulative cut efficiency on M(Λ̄pK⁻) distributions (3 categories: all/pass/fail) |
| `ccbar_background_search` | Search for cc̄ resonances in M(Λ̄pK⁻) separated by B⁺ mass region (signal vs sidebands) |
| `signal_efficiency_fits` | Signal efficiency from M(B⁺) fits (Crystal Ball + ARGUS) for MC and Data |
