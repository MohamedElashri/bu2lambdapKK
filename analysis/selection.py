"""
Module for applying selection criteria to the data
"""

import logging
import numpy as np
import awkward as ak

class SelectionProcessor:
    """Class for applying selection criteria to B+ → pK⁻Λ̄ K+ data"""
    
    def __init__(self):
        """Initialize the selection processor"""
        self.logger = logging.getLogger("Bu2LambdaPKK.SelectionProcessor")
    
    def apply_trigger_selection(self, data):
        """
        Apply trigger requirements
        
        Parameters:
        - data: Dictionary with data arrays
        
        Returns:
        - Dictionary with filtered data arrays
        """
        selected_data = {}
        
        for key, events in data.items():
            # L0 trigger requirement
            l0_selection = (events.Bu_L0Global_TIS) | (events.Bu_L0HadronDecision_TOS)
            
            # L1 trigger requirement
            l1_selection = (events.Bu_Hlt1TrackMVADecision_TOS) | (events.Bu_Hlt1TwoTrackMVADecision_TOS)
            
            # L2 trigger requirement
            l2_selection = (events.Bu_Hlt2Topo2BodyDecision_TOS) | (events.Bu_Hlt2Topo3BodyDecision_TOS) | (events.Bu_Hlt2Topo4BodyDecision_TOS)
            
            # Combined trigger selection
            trigger_mask = l0_selection & l1_selection & l2_selection
            
            # Apply mask
            selected_data[key] = events[trigger_mask]
            self.logger.info(f"Trigger selection for {key}: {len(selected_data[key])}/{len(events)} events passed")
        
        return selected_data
    
    def apply_physics_selection(self, data):
        """
        Apply physics selection criteria
        
        Parameters:
        - data: Dictionary with data arrays
        
        Returns:
        - Dictionary with filtered data arrays
        """
        selected_data = {}
        
        for key, events in data.items():
            # 1. Proton (p) selection
            p_selection = events.p_MC15TuneV1_ProbNNp > 0.05
            
            # 2. Lambda selection
            lambda_dz = events.L0_ENDVERTEX_Z - events.L0_OWNPV_Z
            lambda_selection = (lambda_dz > 20.0) & (events.L0_FDCHI2_OWNPV < 6.0) & (events.Lp_MC15TuneV1_ProbNNp > 0.2)
            
            # 3. B+ selection
            b_selection = (events.Bu_PT > 3000.0) & (events.Bu_DTF_chi2 < 30.0) & (events.Bu_DTF_status == 0) & (events.Bu_IPCHI2_OWNPV < 10.0) & (events.Bu_FDCHI2_OWNPV > 175.0)
            
            # Combined physics selection
            physics_mask = p_selection & lambda_selection & b_selection
            
            # Apply mask
            selected_data[key] = events[physics_mask]
            self.logger.info(f"Physics selection for {key}: {len(selected_data[key])}/{len(events)} events passed")
        
        return selected_data
    
    def apply_basic_selection(self, data):
        """
        Apply both trigger and physics selections
        
        Parameters:
        - data: Dictionary with data arrays
        
        Returns:
        - Dictionary with filtered data arrays
        """
        # First apply trigger selection
        trigger_selected = self.apply_trigger_selection(data)
        
        # Then apply physics selection
        physics_selected = self.apply_physics_selection(trigger_selected)
        
        return physics_selected