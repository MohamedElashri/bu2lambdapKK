import yaml, awkward as ak
import numpy as np
from collections import OrderedDict # Keep if step-by-step needed later

CFG = yaml.safe_load(open("config.yml"))

def trigger_mask(evt, sample):
    """
    Boolean mask: event passes *all* trigger lines defined in config.yml.
    """
    cname = lambda key: CFG["triggers"][key][sample]
    # Use the specific triggers defined in config.yml for the mask
    trigger_keys = ["L0TIS", "L0H_TOS", "H12_TOS", "Topo2_TOS"] # We add others if needed
    cols = [cname(k) for k in trigger_keys if k in CFG["triggers"]] # Ensure key exists

    if not cols:
        print(f"Warning: No trigger columns found for sample '{sample}' in config.yml")
        return ak.Array(np.ones(len(evt), dtype=bool)) # Return all pass if no triggers defined

    # Initialize mask based on the first trigger, ensuring it's boolean
    try:
        mask = ak.values_astype(evt[cols[0]], bool)
        # Combine remaining triggers, ensuring each is cast to boolean
        for c in cols[1:]:
            mask = mask & ak.values_astype(evt[c], bool)
    except Exception as e:
        print(f"Error applying trigger mask for sample {sample}: {e}")
        print(f"Available branches: {evt.fields}")
        print(f"Attempted trigger columns: {cols}")
        # Return a mask of all False in case of error to avoid proceeding with bad data
        return ak.Array(np.zeros(len(evt), dtype=bool))
    return mask

def create_selection_mask(evt, sample):
    """
    Applies selection cuts defined in config.yml.

    Args:
        evt (awkward.Array): The event data.
        sample (str): 'signal' or 'norm'.

    Returns:
        awkward.Array: A boolean mask where True indicates the event passes all cuts.
    """
    cuts = CFG['selections']
    v = CFG['selection_vars']
    pdg = CFG['pdg_masses']
    is_sig = sample == "signal"

    # Initialize mask to all True
    mask = ak.Array(np.ones(len(evt), dtype=bool))

    try:
        # --- Common Cuts ---
        # Delta Z cut
        if 'delta_z_min' in cuts['common'] and v['l0_endvertex_z'] in evt.fields and v['l0_ownpv_z'] in evt.fields:
            mask = mask & ((evt[v['l0_endvertex_z']] - evt[v['l0_ownpv_z']]) > cuts['common']['delta_z_min'])
        # L0 FD chi2 cut
        if 'l0_fdchi2_min' in cuts['common'] and v['l0_fdchi2'] in evt.fields:
             mask = mask & (evt[v['l0_fdchi2']] > cuts['common']['l0_fdchi2_min'])
        # Kaon PID product cut
        if 'kaon_pid_prod_min' in cuts['common'] and v['h1_probNNK'] in evt.fields and v['h2_probNNK'] in evt.fields:
             mask = mask & ((evt[v['h1_probNNK']] * evt[v['h2_probNNK']]) > cuts['common']['kaon_pid_prod_min'])
        # B PT cut
        if 'bu_pt_min' in cuts['common'] and v['bu_pt'] in evt.fields:
             mask = mask & (evt[v['bu_pt']] > cuts['common']['bu_pt_min'])
        # B DTF chi2 cut
        if 'bu_dtf_chi2_max' in cuts['common'] and v['bu_dtf_chi2'] in evt.fields:
             mask = mask & (evt[v['bu_dtf_chi2']] < cuts['common']['bu_dtf_chi2_max'])
        # B IP chi2 cut
        if 'bu_ipchi2_max' in cuts['common'] and v['bu_ipchi2'] in evt.fields:
             mask = mask & (evt[v['bu_ipchi2']] < cuts['common']['bu_ipchi2_max'])
        # B FD chi2 cut
        if 'bu_fdchi2_min' in cuts['common'] and v['bu_fdchi2'] in evt.fields:
             mask = mask & (evt[v['bu_fdchi2']] > cuts['common']['bu_fdchi2_min'])

        # --- Sample-Specific Cuts ---
        l0_branch = v['l0_m']
        if l0_branch in evt.fields:
            if is_sig:
                # Signal specific cuts (Lambda)
                pdg_mass = pdg['lambda']
                window = cuts['signal']['l0_mass_window']
                mask = mask & (np.abs(evt[l0_branch] - pdg_mass) < window)
                # Proton PID (assuming bachelor proton for Lambda channel)
                if 'proton_pid_min' in cuts['signal'] and v['proton_pid'] in evt.fields:
                    mask = mask & (evt[v['proton_pid']] > cuts['signal']['proton_pid_min'])
                 # L0 Proton PID (proton from Lambda decay)
                if 'l0_proton_pid_min' in cuts['signal'] and v['l0_proton_pid'] in evt.fields:
                     mask = mask & (evt[v['l0_proton_pid']] > cuts['signal']['l0_proton_pid_min'])
            else:
                # Normalization specific cuts (Ks)
                pdg_mass = pdg['ks']
                window = cuts['norm']['l0_mass_window']
                mask = mask & (np.abs(evt[l0_branch] - pdg_mass) < window)
                # L0 Pion PID (pions from Ks decay) - Apply cut on *both* pions if needed
                if 'l0_pion_pid_min' in cuts['norm'] and v['l0_pionplus_pid'] in evt.fields and v['l0_pionminus_pid'] in evt.fields:
                    pion_cut = cuts['norm']['l0_pion_pid_min']
                    # Example: require *both* pions to pass PID cut
                    mask = mask & (evt[v['l0_pionplus_pid']] > pion_cut) & (evt[v['l0_pionminus_pid']] > pion_cut)
                    # Or require *at least one* pion to pass:
                    # mask = mask & ((evt[v['l0_pionplus_pid']] > pion_cut) | (evt[v['l0_pionminus_pid']] > pion_cut))

        # Add checks for missing branches/cuts and print warnings if necessary

    except KeyError as e:
        print(f"Error applying selection mask for sample '{sample}': Missing key {e}")
        print(f"Check if variable name exists in config.yml ['selection_vars'] and in TTree branches.")
        print(f"Available branches: {evt.fields}")
        raise # Re-raise the exception to halt execution
    except Exception as e:
        print(f"Unexpected error applying selection mask for sample '{sample}': {e}")
        # Return a mask of all False in case of error
        return ak.Array(np.zeros(len(evt), dtype=bool))

    return mask
