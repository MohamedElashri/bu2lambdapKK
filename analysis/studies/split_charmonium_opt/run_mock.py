class MockSnakemake:
    def __init__(self):
        self.params = type(
            "Params",
            (),
            {"config_dir": "../../config", "cache_dir": "../../cache", "output_dir": "output"},
        )()
        self.output = type("Output", (), {"csv": "output/split_opt_cut_table.csv"})()
        self.config = {"years": ["2016", "2017", "2018"], "track_types": ["LL", "DD"]}


import builtins

builtins.snakemake = MockSnakemake()

with open("study_split_opt.py") as f:
    exec(f.read())
