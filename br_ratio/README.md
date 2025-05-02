## Branching Ratio Analysis
**WORK IN PROGRESS**


### Setup

you can setup the environment by running:

```bash
source run_br.sh
```

### Running the analysis

you can run the analysis by running:

```bash
br_run
```


### Config

Everthing should be configured in the `config.yml` file.


### Setup environment

We will use conda to setup the environment. This can be done using `venv` but I prefer conda.

```bash
conda create -n bphysics python=3.9 -y
```

Then activate the environment:
```bash
conda activate bphysics
```

And install ROOT from `conda-forge` channel: or if you have ROOT installed already, you can skip this step.

```bash
conda install -c conda-forge root -y
```

And install Python dependencies:

```bash
conda install -c conda-forge numpy awkward uproot pyyaml matplotlib tqdm -y
```

And install the remaining dependencies:

```bash
pip install -r requirements.txt
```

now you can run the analysis by running:
