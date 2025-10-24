import yaml
from loaders import load_data
from branches import canonical
import awkward as ak

# Load config
CFG = yaml.safe_load(open("config.yml"))

def compute_trigger_efficiencies(sample):
    # Get the correct data path and decay mode
    if sample == "signal":
        data_path = CFG["signal_data_dir"]
        decay_mode = "L0barPKpKm"
    else:
        data_path = CFG["norm_data_dir"]
        decay_mode = CFG.get("norm_mode_data", "KSKmKpPip")

    # Get trigger keys and corresponding branch names
    trigger_keys = list(CFG["triggers"].keys())
    trigger_branches = canonical(sample, trigger_keys)

    # Load all data with just the trigger branches
    from loaders import load_data
    data = load_data(
        data_path=data_path,
        decay_mode=decay_mode,
        years=CFG["years"],
        tracks=CFG["tracks"],
        sample=sample,
        additional_branches=trigger_branches
    )
    if data is None or len(data) == 0:
        print(f"No data found for {sample}")
        return {}

    eff = {}
    total = len(data)
    for key, branch in zip(trigger_keys, trigger_branches):
        n_pass = ak.sum(data[branch])
        eff[key] = (n_pass, total, n_pass / total if total > 0 else 0)
    return eff

def print_efficiency_table():
    header = (
        "***" +
        f"{'Trigger':^14}***{'Signal Eff (%)':^20}***{'Norm Eff (%)':^20}***"
    )
    border = "*" * len(header)
    # Collect table rows as strings
    table_rows = []
    sig_eff = compute_trigger_efficiencies("signal")
    norm_eff = compute_trigger_efficiencies("norm")
    for key in CFG["triggers"].keys():
        sig_val = sig_eff.get(key, (0, 0, 0))
        norm_val = norm_eff.get(key, (0, 0, 0))
        table_rows.append(f"***{key:^14}***{sig_val[2]*100:^20.2f}***{norm_val[2]*100:^20.2f}***")
        table_rows.append(border)
    # Print the table after all other output
    print(border)
    for row in table_rows:
        print(row)
    print(f"{'':^3}{'Trigger':^14}{'':^3}***{'Signal (data)':^20}***{'Norm (data)':^20}***")
    print(border)

if __name__ == "__main__":
    print_efficiency_table()