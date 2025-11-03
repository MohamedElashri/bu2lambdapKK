#!/usr/bin/env python3
"""
Efficiency Calculator Module for Selection Study

Calculate selection efficiencies for single and combined cuts,
perform efficiency scans, and generate cutflow tables.

Author: Mohamed Elashri
Date: October 28, 2025
"""

import logging
import numpy as np
import awkward as ak
from typing import Dict, List, Tuple


class EfficiencyCalculator:
    """
    Calculate selection efficiencies for single and combined cuts
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._jagged_branches_warned = set()  # Track branches we've already warned about
    
    def calculate_single_cut(self, data: ak.Array, branch: str, 
                            cut_value: float, operator: str) -> Tuple[float, int, int]:
        """
        Calculate efficiency for a single cut
        
        Parameters:
        - data: awkward array with event data
        - branch: branch name to cut on
        - cut_value: cut threshold value
        - operator: comparison operator ('>', '<', '>=', '<=', '==', '!=')
        
        Returns:
        - efficiency, n_passed, n_total
        """
        if branch not in data.fields:
            self.logger.warning(f"Branch {branch} not found in data")
            return 0.0, 0, 0
            
        branch_data = data[branch]
        # Check if jagged by comparing flattened length to original
        try:
            if len(ak.flatten(branch_data)) != len(branch_data):
                # Only warn once per branch
                if branch not in self._jagged_branches_warned:
                    self.logger.warning(f"Branch {branch} is jagged, using first element")
                    self._jagged_branches_warned.add(branch)
                branch_data = branch_data[:, 0]
        except:
            pass
        
        branch_data = ak.to_numpy(branch_data)
        n_total = len(branch_data)
        
        # Apply operator
        if operator == '>':
            mask = branch_data > cut_value
        elif operator == '<':
            mask = branch_data < cut_value
        elif operator == '>=':
            mask = branch_data >= cut_value
        elif operator == '<=':
            mask = branch_data <= cut_value
        elif operator == '==':
            mask = branch_data == cut_value
        elif operator == '!=':
            mask = branch_data != cut_value
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        n_passed = np.sum(mask)
        efficiency = n_passed / n_total if n_total > 0 else 0.0
        
        return efficiency, n_passed, n_total
    
    def scan_efficiency(self, data: ak.Array, branch: str, 
                       cut_values: np.ndarray, operator: str) -> List[Tuple]:
        """
        Scan efficiency over range of cut values
        
        Returns:
        - List of (cut_value, efficiency, n_passed, n_total) tuples
        """
        results = []
        for cut_val in cut_values:
            eff, n_pass, n_tot = self.calculate_single_cut(data, branch, cut_val, operator)
            results.append((cut_val, eff, n_pass, n_tot))
        
        return results
    
    def find_optimal_cut(self, scan_results: List[Tuple], 
                        min_efficiency: float = 0.70) -> float:
        """
        Find optimal cut value based on criteria
        
        Parameters:
        - scan_results: List of (cut_value, efficiency, n_passed, n_total)
        - min_efficiency: Minimum acceptable efficiency
        
        Returns:
        - Optimal cut value
        """
        valid_cuts = [(cut, eff) for cut, eff, _, _ in scan_results if eff >= min_efficiency]
        
        if not valid_cuts:
            self.logger.warning(f"No cuts satisfy minimum efficiency {min_efficiency}")
            return scan_results[0][0]  # Return first cut value as fallback
        
        # Return cut with highest efficiency among valid cuts
        return max(valid_cuts, key=lambda x: x[1])[0]
    
    def generate_cutflow(self, data_dict: Dict[str, ak.Array], 
                        cuts: List[Tuple[str, callable]]) -> Dict:
        """
        Generate sequential cutflow table
        
        Parameters:
        - data_dict: Dictionary of datasets {'name': data}
        - cuts: List of (cut_name, cut_function) tuples
        
        Returns:
        - Dictionary with cutflow information
        """
        cutflow = {}
        
        for dataset_name, initial_data in data_dict.items():
            cutflow[dataset_name] = {}
            current_data = initial_data
            n_initial = len(initial_data)
            
            for cut_name, cut_func in cuts:
                # Apply cut
                current_data = cut_func(current_data)
                n_passed = len(current_data)
                
                # Calculate efficiencies
                abs_eff = n_passed / n_initial if n_initial > 0 else 0.0
                
                cutflow[dataset_name][cut_name] = {
                    'n_passed': n_passed,
                    'abs_efficiency': abs_eff
                }
        
        return cutflow
