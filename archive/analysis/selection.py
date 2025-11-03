"""
Module for applying selection criteria to the data
"""

import logging
import numpy as np
import awkward as ak
import tomli
from pathlib import Path
from collections import OrderedDict

class SelectionProcessor:
    """Class for applying selection criteria to B+ → pK⁻Λ̄ K+ data"""
    
    def __init__(self, config_path=None):
        """
        Initialize the selection processor
        
        Parameters:
        - config_path: Path to selection configuration TOML file
                      If None, uses default selection.toml in the same directory
        """
        self.logger = logging.getLogger("Bu2LambdaPKK.SelectionProcessor")
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "selection.toml"
        
        with open(config_path, 'rb') as f:
            self.config = tomli.load(f)
        
        self.logger.info(f"Loaded selection configuration from: {config_path}")
        self.logger.info(f"Configuration version: {self.config.get('metadata', {}).get('version', 'unknown')}")
        
        # Store derived branches definitions
        self.derived_branches = self.config.get('derived_branches', {})
        
        # Determine which cut set to use
        self.active_cut_set = self.config.get('cut_set', {}).get('active', 'tight')
        self.logger.info(f"Using '{self.active_cut_set}' cut set")
        
        # Load the active cut values
        self._load_cut_values()
    
    def _load_cut_values(self):
        """
        Load cut values from the active cut set and update the cut definitions
        """
        cut_set_name = self.active_cut_set
        
        if cut_set_name not in self.config:
            raise ValueError(f"Cut set '{cut_set_name}' not found in configuration")
        
        cut_set = self.config[cut_set_name]
        self.logger.info(f"Loading cut values from '{cut_set_name}' set: {cut_set.get('description', '')}")
        
        # Update cut values in the main cut definitions
        for category in ['proton', 'lambda', 'kaon', 'bplus']:
            if category not in self.config:
                continue
            
            # Get the cut values for this category from the active set
            if category not in cut_set:
                continue
            
            set_values = cut_set[category]
            
            # Update each cut with values from the active set
            for cut_info in self.config[category].get('cuts', []):
                cut_name = cut_info['name']
                branch = cut_info['branch']
                
                # Create a mapping of possible keys to check in the cut set
                # Priority: specific name parts that match the cut set keys
                possible_keys = []
                
                # For proton cuts: p_ProbNNp -> p_ProbNNp
                if category == 'proton':
                    if 'ProbNNp' in branch:
                        possible_keys.append('p_ProbNNp')
                
                # For lambda cuts: extract meaningful parts
                elif category == 'lambda':
                    if 'delta_z' in cut_name or 'delta_z' in branch:
                        possible_keys.append('delta_z')
                    elif 'FD_CHISQ' in branch or 'FD_CHISQ' in cut_name:
                        possible_keys.append('FD_CHISQ')
                    elif 'FDCHI2' in branch:
                        possible_keys.append('FDCHI2')
                    elif 'mass_window_lower' in cut_name:
                        possible_keys.append('mass_window_lower')
                    elif 'mass_window_upper' in cut_name:
                        possible_keys.append('mass_window_upper')
                    elif 'mass' in cut_name:
                        possible_keys.append('mass_window')
                    elif 'Lp_' in branch and 'ProbNNp' in branch:
                        possible_keys.append('proton_ProbNNp')
                
                # For kaon cuts
                elif category == 'kaon':
                    if 'kk_product' in cut_name or 'kk_product' in branch:
                        possible_keys.append('kk_product')
                
                # For B+ cuts: Bu_PT -> PT, Bu_DTF_chi2 -> DTF_chi2, etc.
                elif category == 'bplus':
                    # Handle mass window cuts specially
                    if 'mass_window_lower' in cut_name:
                        possible_keys.append('mass_window_lower')
                    elif 'mass_window_upper' in cut_name:
                        possible_keys.append('mass_window_upper')
                    # Check if mass_window is enabled
                    if 'mass_window' in cut_name:
                        # Check for mass_window_enabled flag
                        if 'mass_window_enabled' in set_values:
                            cut_info['enabled'] = set_values['mass_window_enabled']
                    
                    # Remove Bu_ prefix and use the rest
                    if branch.startswith('Bu_'):
                        key = branch.replace('Bu_', '')
                        possible_keys.append(key)
                    # Also try the last part of the cut name
                    simple_name = cut_name.split('_')[-1]
                    if simple_name not in possible_keys:
                        possible_keys.append(simple_name)
                
                # Try to find a matching key in the cut set
                updated = False
                for key in possible_keys:
                    if key in set_values:
                        old_value = cut_info['value']
                        cut_info['value'] = set_values[key]
                        self.logger.debug(f"Updated {cut_name}: {old_value} -> {cut_info['value']}")
                        updated = True
                        break
                
                if not updated:
                    self.logger.debug(f"No update for {cut_name} (tried: {possible_keys})")
    
    def _compute_derived_branches(self, events):
        """
        Compute derived branches from existing branches
        
        Parameters:
        - events: awkward array with event data
        
        Returns:
        - Dictionary of derived branch names and their computed values
        """
        derived = {}
        
        for branch_name, expression in self.derived_branches.items():
            try:
                # Build a safe namespace for evaluation
                namespace = {
                    'events': events,
                    'np': np,
                    'ak': ak,
                    'abs': np.abs,
                    'sqrt': np.sqrt,
                    'square': np.square
                }
                
                # Add all event branches to namespace
                for field in events.fields:
                    namespace[field] = ak.to_numpy(events[field])
                
                # Evaluate the expression directly
                result = eval(expression, {"__builtins__": {}}, namespace)
                
                # Convert to numpy array if needed
                if hasattr(result, '__array__'):
                    derived[branch_name] = np.array(result)
                else:
                    derived[branch_name] = np.array(result)
                    
                self.logger.debug(f"Computed derived branch: {branch_name}")
                
            except Exception as e:
                self.logger.error(f"Error computing derived branch {branch_name}: {e}")
                raise
        
        # Compute L0_FD_CHISQ (special case - more complex calculation)
        # Based on old notebook: L0_FD_CHISQ = (Delta_X/Delta_X_ERR)^2 + (Delta_Y/Delta_Y_ERR)^2 + (Delta_Z/Delta_Z_ERR)^2
        try:
            Delta_X = ak.to_numpy(events.L0_ENDVERTEX_X) - ak.to_numpy(events.Bu_ENDVERTEX_X)
            Delta_Y = ak.to_numpy(events.L0_ENDVERTEX_Y) - ak.to_numpy(events.Bu_ENDVERTEX_Y)
            Delta_Z = ak.to_numpy(events.L0_ENDVERTEX_Z) - ak.to_numpy(events.Bu_ENDVERTEX_Z)
            
            Delta_X_ERR = np.sqrt(np.square(ak.to_numpy(events.Bu_ENDVERTEX_XERR)) + 
                                 np.square(ak.to_numpy(events.L0_ENDVERTEX_XERR)))
            Delta_Y_ERR = np.sqrt(np.square(ak.to_numpy(events.Bu_ENDVERTEX_YERR)) + 
                                 np.square(ak.to_numpy(events.L0_ENDVERTEX_YERR)))
            Delta_Z_ERR = np.sqrt(np.square(ak.to_numpy(events.Bu_ENDVERTEX_ZERR)) + 
                                 np.square(ak.to_numpy(events.L0_ENDVERTEX_ZERR)))
            
            delta_x = np.divide(Delta_X, Delta_X_ERR)
            delta_y = np.divide(Delta_Y, Delta_Y_ERR)
            delta_z = np.divide(Delta_Z, Delta_Z_ERR)
            
            L0_FD_CHISQ = np.square(delta_x) + np.square(delta_y) + np.square(delta_z)
            derived['L0_FD_CHISQ'] = L0_FD_CHISQ
            
            self.logger.debug(f"Computed L0_FD_CHISQ derived branch")
            
        except Exception as e:
            self.logger.error(f"Error computing L0_FD_CHISQ: {e}")
            raise
        
        return derived
    
    def _evaluate_cut(self, events, derived_branches, cut_info):
        """
        Evaluate a single cut
        
        Parameters:
        - events: awkward array with event data
        - derived_branches: dictionary of derived branches
        - cut_info: dictionary with cut configuration
        
        Returns:
        - Boolean mask for this cut
        """
        branch_name = cut_info['branch']
        operator = cut_info['operator']
        value = cut_info['value']
        
        # Get the branch data (either from events or derived)
        if branch_name in derived_branches:
            branch_data = derived_branches[branch_name]
        else:
            branch_data = ak.to_numpy(events[branch_name])
        
        # Apply the operator
        if operator == '>':
            mask = branch_data > value
        elif operator == '<':
            mask = branch_data < value
        elif operator == '>=':
            mask = branch_data >= value
        elif operator == '<=':
            mask = branch_data <= value
        elif operator == '==':
            mask = branch_data == value
        elif operator == '!=':
            mask = branch_data != value
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return mask
    
    def apply_trigger_selection(self, data):
        """
        Apply trigger requirements from configuration
        
        Parameters:
        - data: Dictionary with data arrays
        
        Returns:
        - Dictionary with filtered data arrays
        """
        if not self.config.get('trigger', {}).get('enabled', True):
            self.logger.info("Trigger selection is disabled in config")
            return data
        
        selected_data = {}
        trigger_config = self.config['trigger']
        
        for key, events in data.items():
            # L0 trigger requirement
            l0_expr = trigger_config['L0']['expression']
            l0_selection = eval(l0_expr.replace('Bu_L0Global_TIS', 'events.Bu_L0Global_TIS')
                                      .replace('Bu_L0HadronDecision_TOS', 'events.Bu_L0HadronDecision_TOS'))
            
            # L1 trigger requirement
            l1_expr = trigger_config['L1']['expression']
            l1_selection = eval(l1_expr.replace('Bu_Hlt1TrackMVADecision_TOS', 'events.Bu_Hlt1TrackMVADecision_TOS')
                                      .replace('Bu_Hlt1TwoTrackMVADecision_TOS', 'events.Bu_Hlt1TwoTrackMVADecision_TOS'))
            
            # L2 trigger requirement
            l2_expr = trigger_config['L2']['expression']
            l2_selection = eval(l2_expr.replace('Bu_Hlt2Topo2BodyDecision_TOS', 'events.Bu_Hlt2Topo2BodyDecision_TOS')
                                      .replace('Bu_Hlt2Topo3BodyDecision_TOS', 'events.Bu_Hlt2Topo3BodyDecision_TOS')
                                      .replace('Bu_Hlt2Topo4BodyDecision_TOS', 'events.Bu_Hlt2Topo4BodyDecision_TOS'))
            
            # Combined trigger selection
            trigger_mask = l0_selection & l1_selection & l2_selection
            
            # Apply mask
            selected_data[key] = events[trigger_mask]
            self.logger.info(f"Trigger selection for {key}: {len(selected_data[key])}/{len(events)} events passed")
        
        return selected_data
    
    def apply_physics_selection(self, data, return_summary=False):
        """
        Apply physics selection criteria from configuration
        
        Parameters:
        - data: Dictionary with data arrays
        - return_summary: If True, also return detailed cut summary
        
        Returns:
        - Dictionary with filtered data arrays
        - (Optional) Dictionary with cut summaries
        """
        selected_data = {}
        all_summaries = {}
        
        for key, events in data.items():
            # Compute derived branches once for all cuts
            derived_branches = self._compute_derived_branches(events)
            
            # Initialize mask with all True
            combined_mask = np.ones(len(events), dtype=bool)
            
            # Track cuts for summary
            cuts_summary = OrderedDict()
            cuts_summary['initial_events'] = len(events)
            
            # Apply cuts from each category
            for category in ['proton', 'lambda', 'kaon', 'bplus']:
                if category not in self.config:
                    continue
                
                category_config = self.config[category]
                
                for cut_info in category_config.get('cuts', []):
                    # Check if cut is enabled (default True)
                    if not cut_info.get('enabled', True):
                        self.logger.debug(f"Skipping disabled cut: {cut_info['name']}")
                        continue
                    
                    # Evaluate the cut
                    cut_mask = self._evaluate_cut(events, derived_branches, cut_info)
                    
                    # Track statistics
                    n_passed = np.sum(cut_mask)
                    efficiency = n_passed / len(events) * 100
                    
                    cuts_summary[cut_info['name']] = {
                        'passed': n_passed,
                        'total': len(events),
                        'efficiency': efficiency,
                        'description': cut_info.get('description', '')
                    }
                    
                    self.logger.debug(f"Cut {cut_info['name']}: {n_passed}/{len(events)} "
                                    f"({efficiency:.2f}%) - {cut_info.get('description', '')}")
                    
                    # Combine with overall mask
                    combined_mask = combined_mask & cut_mask
            
            # Track final statistics
            n_selected = np.sum(combined_mask)
            total_efficiency = n_selected / len(events) * 100
            cuts_summary['final_selected'] = {
                'passed': n_selected,
                'total': len(events),
                'efficiency': total_efficiency
            }
            
            # Apply mask
            selected_data[key] = events[combined_mask]
            self.logger.info(f"Physics selection for {key}: {n_selected}/{len(events)} "
                           f"events passed ({total_efficiency:.2f}%)")
            
            if return_summary:
                all_summaries[key] = cuts_summary
        
        if return_summary:
            return selected_data, all_summaries
        return selected_data
    
    def apply_basic_selection(self, data, return_summary=False):
        """
        Apply both trigger and physics selections
        
        Parameters:
        - data: Dictionary with data arrays
        - return_summary: If True, also return detailed cut summary
        
        Returns:
        - Dictionary with filtered data arrays
        - (Optional) Dictionary with cut summaries
        """
        # First apply trigger selection
        trigger_selected = self.apply_trigger_selection(data)
        
        # Then apply physics selection
        if return_summary:
            physics_selected, summaries = self.apply_physics_selection(trigger_selected, return_summary=True)
            return physics_selected, summaries
        else:
            physics_selected = self.apply_physics_selection(trigger_selected, return_summary=False)
            return physics_selected
    
    def print_cut_summary(self, summaries):
        """
        Print a formatted summary of all cuts
        
        Parameters:
        - summaries: Dictionary of cut summaries from apply_physics_selection
        """
        for dataset_key, cuts_summary in summaries.items():
            print(f"\n{'='*70}")
            print(f"Selection Summary for {dataset_key}")
            print(f"{'='*70}")
            
            initial = cuts_summary['initial_events']
            print(f"Initial events: {initial}")
            print(f"\n{'Cut Name':<30} {'Passed':<12} {'Efficiency':<12} {'Description':<30}")
            print(f"{'-'*70}")
            
            for cut_name, cut_info in cuts_summary.items():
                if cut_name in ['initial_events', 'final_selected']:
                    continue
                    
                passed = cut_info['passed']
                efficiency = cut_info['efficiency']
                description = cut_info['description'][:30]
                
                print(f"{cut_name:<30} {passed:<12} {efficiency:>6.2f}%     {description}")
            
            final = cuts_summary['final_selected']
            print(f"\n{'Final selected':<30} {final['passed']:<12} {final['efficiency']:>6.2f}%")
            print(f"{'='*70}\n")