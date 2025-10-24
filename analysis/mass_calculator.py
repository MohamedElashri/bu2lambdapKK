"""
Module for calculating invariant masses
"""

import logging
import numpy as np
import awkward as ak

class MassCalculator:
    """Class for calculating invariant masses for B+ → pK⁻Λ̄ K+ analysis"""
    
    def __init__(self):
        """Initialize the mass calculator"""
        self.logger = logging.getLogger("Bu2LambdaPKK.MassCalculator")
    
    def calculate_jpsi_candidates(self, data):
        """
        Calculate invariant mass of pK⁻Λ̄ to identify J/ψ candidates
        
        Parameters:
        - data: Dictionary with data arrays
        
        Returns:
        - Dictionary with updated data arrays containing M_pKLambdabar
        """
        data_with_masses = {}
        
        for key, events in data.items():
            self.logger.info(f"Calculating pK⁻Λ̄ mass for {key}...")
            
            # Extract four-momenta of proton (p)
            p_p = {
                "px": events.p_PX,
                "py": events.p_PY,
                "pz": events.p_PZ,
                "e": events.p_PE
            }
            
            # For Kaon (K-), we need to identify which hadron is the K-
            # Based on ID and ProbNNk, h1 or h2 could be the K-
            # Let's assume h1 is K- if its ID is negative (K- has negative charge)
            # Otherwise, h2 is K-
            k_minus_mask = events.h1_ID < 0
            
            # Initialize arrays for K- four-momentum
            k_minus_px = ak.zeros_like(events.h1_PX)
            k_minus_py = ak.zeros_like(events.h1_PY)
            k_minus_pz = ak.zeros_like(events.h1_PZ)
            k_minus_e = ak.zeros_like(events.h1_PE)
            
            # Fill K- arrays based on which hadron is K-
            k_minus_px = ak.where(k_minus_mask, events.h1_PX, events.h2_PX)
            k_minus_py = ak.where(k_minus_mask, events.h1_PY, events.h2_PY)
            k_minus_pz = ak.where(k_minus_mask, events.h1_PZ, events.h2_PZ)
            k_minus_e = ak.where(k_minus_mask, events.h1_PE, events.h2_PE)
            
            p_k_minus = {
                "px": k_minus_px,
                "py": k_minus_py,
                "pz": k_minus_pz,
                "e": k_minus_e
            }
            
            # For Lambda (anti-Lambda), we use its four-momentum
            p_lambda = {
                "px": events.L0_PX,
                "py": events.L0_PY,
                "pz": events.L0_PZ,
                "e": events.L0_PE
            }
            
            # Calculate the invariant mass of pK⁻Λ̄
            m_pklambda = self._invariant_mass(p_p, p_k_minus, p_lambda)
            
            # Add invariant mass to data
            data_with_masses[key] = ak.with_field(events, m_pklambda, "M_pKLambdabar")
            
            self.logger.info(f"Added pK⁻Λ̄ mass to {len(events)} events")
        
        return data_with_masses
    
    def _invariant_mass(self, p1, p2, p3):
        """
        Calculate invariant mass of three particles
        
        Parameters:
        - p1, p2, p3: Dictionaries with four-momentum components (px, py, pz, e)
        
        Returns:
        - Invariant mass
        """
        # Add four-momenta
        px_tot = p1["px"] + p2["px"] + p3["px"]
        py_tot = p1["py"] + p2["py"] + p3["py"]
        pz_tot = p1["pz"] + p2["pz"] + p3["pz"]
        e_tot = p1["e"] + p2["e"] + p3["e"]
        
        # Calculate invariant mass
        m_squared = e_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        
        # Handle negative values due to floating-point precision
        m_squared_positive = ak.where(m_squared < 0, 0, m_squared)
        
        return np.sqrt(m_squared_positive)