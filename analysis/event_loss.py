#!/usr/bin/env python3
"""
Diagnostic script to analyze event losses at each selection stage
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import awkward as ak

from data_loader import DataLoader
from selection import SelectionProcessor

def setup_logging(verbose=False):
    """Configure logging level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Bu2LambdaPKK_Diagnostic")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Diagnose event losses in B+ → pK⁻Λ̄ K+ analysis")
    parser.add_argument("--data-dir", default="/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced", 
                        help="Directory containing data files")
    parser.add_argument("--years", default="16,17,18", help="Years to process (comma-separated)")
    parser.add_argument("--polarity", default="MD,MU", help="Magnet polarities to process")
    parser.add_argument("--track-types", default="LL,DD", help="Track types to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def analyze_selection_stages(data_dir, years, polarities, track_types):
    """Analyze event losses at each selection stage"""
    logger = setup_logging(True)
    
    # Load data
    loader = DataLoader(data_dir)
    data = loader.load_data(years, polarities, track_types, "B2L0barPKpKm")
    
    # Print initial event counts
    total_raw_events = 0
    for key, events in data.items():
        logger.info(f"Sample {key}: {len(events)} raw events")
        total_raw_events += len(events)
    
    logger.info(f"Total raw events: {total_raw_events}")
    
    # Create selection processor
    selector = SelectionProcessor()
    
    # Analyze trigger selection
    trigger_selected = selector.apply_trigger_selection(data)
    
    total_trigger_events = 0
    for key, events in trigger_selected.items():
        trigger_eff = 100.0 * len(events) / len(data[key]) if len(data[key]) > 0 else 0
        logger.info(f"Sample {key}: {len(events)} events after trigger selection ({trigger_eff:.1f}% efficiency)")
        total_trigger_events += len(events)
    
    trigger_total_eff = 100.0 * total_trigger_events / total_raw_events if total_raw_events > 0 else 0
    logger.info(f"Total events after trigger: {total_trigger_events} ({trigger_total_eff:.1f}% efficiency)")
    
    # Analyze individual trigger requirements
    analyze_trigger_details(data)
    
    # Analyze physics selection
    physics_selected = selector.apply_physics_selection(trigger_selected)
    
    total_physics_events = 0
    for key, events in physics_selected.items():
        physics_eff = 100.0 * len(events) / len(trigger_selected[key]) if len(trigger_selected[key]) > 0 else 0
        total_eff = 100.0 * len(events) / len(data[key]) if len(data[key]) > 0 else 0
        logger.info(f"Sample {key}: {len(events)} events after physics selection")
        logger.info(f"  - {physics_eff:.1f}% efficiency relative to trigger selection")
        logger.info(f"  - {total_eff:.1f}% total efficiency")
        total_physics_events += len(events)
    
    physics_relative_eff = 100.0 * total_physics_events / total_trigger_events if total_trigger_events > 0 else 0
    physics_total_eff = 100.0 * total_physics_events / total_raw_events if total_raw_events > 0 else 0
    
    logger.info(f"Total events after physics: {total_physics_events}")
    logger.info(f"  - {physics_relative_eff:.1f}% efficiency relative to trigger selection")
    logger.info(f"  - {physics_total_eff:.1f}% total efficiency")
    
    # Analyze individual physics requirements
    analyze_physics_details(trigger_selected)

def analyze_trigger_details(data):
    """Analyze the effect of individual trigger requirements"""
    logger = logging.getLogger("Bu2LambdaPKK_Diagnostic")
    
    logger.info("Analyzing individual trigger requirements:")
    
    total_events = 0
    l0_global_tis_events = 0
    l0_hadron_tos_events = 0
    l0_combined_events = 0
    l1_track_mva_tos_events = 0
    l1_two_track_mva_tos_events = 0
    l1_combined_events = 0
    l2_topo2_tos_events = 0
    l2_topo3_tos_events = 0
    l2_topo4_tos_events = 0
    l2_combined_events = 0
    all_combined_events = 0
    
    for key, events in data.items():
        n_events = len(events)
        total_events += n_events
        
        # L0 requirements
        l0_global_tis = ak.sum(events.Bu_L0Global_TIS)
        l0_hadron_tos = ak.sum(events.Bu_L0HadronDecision_TOS)
        l0_combined = ak.sum(events.Bu_L0Global_TIS | events.Bu_L0HadronDecision_TOS)
        
        l0_global_tis_events += l0_global_tis
        l0_hadron_tos_events += l0_hadron_tos
        l0_combined_events += l0_combined
        
        # L1 requirements
        l1_track_mva_tos = ak.sum(events.Bu_Hlt1TrackMVADecision_TOS)
        l1_two_track_mva_tos = ak.sum(events.Bu_Hlt1TwoTrackMVADecision_TOS)
        l1_combined = ak.sum(events.Bu_Hlt1TrackMVADecision_TOS | events.Bu_Hlt1TwoTrackMVADecision_TOS)
        
        l1_track_mva_tos_events += l1_track_mva_tos
        l1_two_track_mva_tos_events += l1_two_track_mva_tos
        l1_combined_events += l1_combined
        
        # L2 requirements
        l2_topo2_tos = ak.sum(events.Bu_Hlt2Topo2BodyDecision_TOS)
        l2_topo3_tos = ak.sum(events.Bu_Hlt2Topo3BodyDecision_TOS)
        l2_topo4_tos = ak.sum(events.Bu_Hlt2Topo4BodyDecision_TOS)
        l2_combined = ak.sum(events.Bu_Hlt2Topo2BodyDecision_TOS | 
                          events.Bu_Hlt2Topo3BodyDecision_TOS | 
                          events.Bu_Hlt2Topo4BodyDecision_TOS)
        
        l2_topo2_tos_events += l2_topo2_tos
        l2_topo3_tos_events += l2_topo3_tos
        l2_topo4_tos_events += l2_topo4_tos
        l2_combined_events += l2_combined
        
        # All combined
        all_combined = ak.sum((events.Bu_L0Global_TIS | events.Bu_L0HadronDecision_TOS) &
                            (events.Bu_Hlt1TrackMVADecision_TOS | events.Bu_Hlt1TwoTrackMVADecision_TOS) &
                            (events.Bu_Hlt2Topo2BodyDecision_TOS | events.Bu_Hlt2Topo3BodyDecision_TOS | 
                             events.Bu_Hlt2Topo4BodyDecision_TOS))
        
        all_combined_events += all_combined
        
        logger.info(f"Sample {key} trigger breakdown:")
        logger.info(f"  - L0 Global TIS: {l0_global_tis}/{n_events} ({100.0*l0_global_tis/n_events:.1f}%)")
        logger.info(f"  - L0 Hadron TOS: {l0_hadron_tos}/{n_events} ({100.0*l0_hadron_tos/n_events:.1f}%)")
        logger.info(f"  - L0 Combined: {l0_combined}/{n_events} ({100.0*l0_combined/n_events:.1f}%)")
        logger.info(f"  - L1 Track MVA TOS: {l1_track_mva_tos}/{n_events} ({100.0*l1_track_mva_tos/n_events:.1f}%)")
        logger.info(f"  - L1 Two Track MVA TOS: {l1_two_track_mva_tos}/{n_events} ({100.0*l1_two_track_mva_tos/n_events:.1f}%)")
        logger.info(f"  - L1 Combined: {l1_combined}/{n_events} ({100.0*l1_combined/n_events:.1f}%)")
        logger.info(f"  - L2 Topo2 TOS: {l2_topo2_tos}/{n_events} ({100.0*l2_topo2_tos/n_events:.1f}%)")
        logger.info(f"  - L2 Topo3 TOS: {l2_topo3_tos}/{n_events} ({100.0*l2_topo3_tos/n_events:.1f}%)")
        logger.info(f"  - L2 Topo4 TOS: {l2_topo4_tos}/{n_events} ({100.0*l2_topo4_tos/n_events:.1f}%)")
        logger.info(f"  - L2 Combined: {l2_combined}/{n_events} ({100.0*l2_combined/n_events:.1f}%)")
        logger.info(f"  - All Combined: {all_combined}/{n_events} ({100.0*all_combined/n_events:.1f}%)")
    
    # Overall summary
    logger.info("Overall trigger breakdown:")
    logger.info(f"  - L0 Global TIS: {l0_global_tis_events}/{total_events} ({100.0*l0_global_tis_events/total_events:.1f}%)")
    logger.info(f"  - L0 Hadron TOS: {l0_hadron_tos_events}/{total_events} ({100.0*l0_hadron_tos_events/total_events:.1f}%)")
    logger.info(f"  - L0 Combined: {l0_combined_events}/{total_events} ({100.0*l0_combined_events/total_events:.1f}%)")
    logger.info(f"  - L1 Track MVA TOS: {l1_track_mva_tos_events}/{total_events} ({100.0*l1_track_mva_tos_events/total_events:.1f}%)")
    logger.info(f"  - L1 Two Track MVA TOS: {l1_two_track_mva_tos_events}/{total_events} ({100.0*l1_two_track_mva_tos_events/total_events:.1f}%)")
    logger.info(f"  - L1 Combined: {l1_combined_events}/{total_events} ({100.0*l1_combined_events/total_events:.1f}%)")
    logger.info(f"  - L2 Topo2 TOS: {l2_topo2_tos_events}/{total_events} ({100.0*l2_topo2_tos_events/total_events:.1f}%)")
    logger.info(f"  - L2 Topo3 TOS: {l2_topo3_tos_events}/{total_events} ({100.0*l2_topo3_tos_events/total_events:.1f}%)")
    logger.info(f"  - L2 Topo4 TOS: {l2_topo4_tos_events}/{total_events} ({100.0*l2_topo4_tos_events/total_events:.1f}%)")
    logger.info(f"  - L2 Combined: {l2_combined_events}/{total_events} ({100.0*l2_combined_events/total_events:.1f}%)")
    logger.info(f"  - All Combined: {all_combined_events}/{total_events} ({100.0*all_combined_events/total_events:.1f}%)")

def analyze_physics_details(data):
    """Analyze the effect of individual physics selection requirements"""
    logger = logging.getLogger("Bu2LambdaPKK_Diagnostic")
    
    logger.info("Analyzing individual physics selection requirements:")
    
    total_events = 0
    p_prob_nn_p_events = 0
    lambda_dz_events = 0
    lambda_fd_chi2_events = 0
    lambda_p_prob_events = 0
    b_pt_events = 0
    b_dtf_chi2_events = 0
    b_dtf_status_events = 0
    b_ip_chi2_events = 0
    b_fd_chi2_events = 0
    all_combined_events = 0
    
    for key, events in data.items():
        n_events = len(events)
        total_events += n_events
        
        # Proton selection
        p_prob_nn_p = ak.sum(events.p_MC15TuneV1_ProbNNp > 0.05)
        p_prob_nn_p_events += p_prob_nn_p
        
        # Lambda selection
        lambda_dz = ak.sum((events.L0_ENDVERTEX_Z - events.L0_OWNPV_Z) > 20.0)
        lambda_fd_chi2 = ak.sum(events.L0_FDCHI2_OWNPV < 6.0)
        lambda_p_prob = ak.sum(events.Lp_MC15TuneV1_ProbNNp > 0.2)
        
        lambda_dz_events += lambda_dz
        lambda_fd_chi2_events += lambda_fd_chi2
        lambda_p_prob_events += lambda_p_prob
        
        # B+ selection
        b_pt = ak.sum(events.Bu_PT > 3000.0)
        b_dtf_chi2 = ak.sum(events.Bu_DTF_chi2 < 30.0)
        b_dtf_status = ak.sum(events.Bu_DTF_status == 0)
        b_ip_chi2 = ak.sum(events.Bu_IPCHI2_OWNPV < 10.0)
        b_fd_chi2 = ak.sum(events.Bu_FDCHI2_OWNPV > 175.0)
        
        b_pt_events += b_pt
        b_dtf_chi2_events += b_dtf_chi2
        b_dtf_status_events += b_dtf_status
        b_ip_chi2_events += b_ip_chi2
        b_fd_chi2_events += b_fd_chi2
        
        # All combined
        all_combined = ak.sum((events.p_MC15TuneV1_ProbNNp > 0.05) &
                            ((events.L0_ENDVERTEX_Z - events.L0_OWNPV_Z) > 20.0) &
                            (events.L0_FDCHI2_OWNPV < 6.0) &
                            (events.Lp_MC15TuneV1_ProbNNp > 0.2) &
                            (events.Bu_PT > 3000.0) &
                            (events.Bu_DTF_chi2 < 30.0) &
                            (events.Bu_DTF_status == 0) &
                            (events.Bu_IPCHI2_OWNPV < 10.0) &
                            (events.Bu_FDCHI2_OWNPV > 175.0))
        
        all_combined_events += all_combined
        
        logger.info(f"Sample {key} physics breakdown:")
        logger.info(f"  - p ProbNNp > 0.05: {p_prob_nn_p}/{n_events} ({100.0*p_prob_nn_p/n_events:.1f}%)")
        logger.info(f"  - Lambda dZ > 20mm: {lambda_dz}/{n_events} ({100.0*lambda_dz/n_events:.1f}%)")
        logger.info(f"  - Lambda FD chi2 < 6: {lambda_fd_chi2}/{n_events} ({100.0*lambda_fd_chi2/n_events:.1f}%)")
        logger.info(f"  - Lambda p ProbNNp > 0.2: {lambda_p_prob}/{n_events} ({100.0*lambda_p_prob/n_events:.1f}%)")
        logger.info(f"  - B+ pT > 3000 MeV: {b_pt}/{n_events} ({100.0*b_pt/n_events:.1f}%)")
        logger.info(f"  - B+ DTF chi2 < 30: {b_dtf_chi2}/{n_events} ({100.0*b_dtf_chi2/n_events:.1f}%)")
        logger.info(f"  - B+ DTF status == 0: {b_dtf_status}/{n_events} ({100.0*b_dtf_status/n_events:.1f}%)")
        logger.info(f"  - B+ IP chi2 < 10: {b_ip_chi2}/{n_events} ({100.0*b_ip_chi2/n_events:.1f}%)")
        logger.info(f"  - B+ FD chi2 > 175: {b_fd_chi2}/{n_events} ({100.0*b_fd_chi2/n_events:.1f}%)")
        logger.info(f"  - All Combined: {all_combined}/{n_events} ({100.0*all_combined/n_events:.1f}%)")
    
    # Overall summary
    logger.info("Overall physics breakdown:")
    logger.info(f"  - p ProbNNp > 0.05: {p_prob_nn_p_events}/{total_events} ({100.0*p_prob_nn_p_events/total_events:.1f}%)")
    logger.info(f"  - Lambda dZ > 20mm: {lambda_dz_events}/{total_events} ({100.0*lambda_dz_events/total_events:.1f}%)")
    logger.info(f"  - Lambda FD chi2 < 6: {lambda_fd_chi2_events}/{total_events} ({100.0*lambda_fd_chi2_events/total_events:.1f}%)")
    logger.info(f"  - Lambda p ProbNNp > 0.2: {lambda_p_prob_events}/{total_events} ({100.0*lambda_p_prob_events/total_events:.1f}%)")
    logger.info(f"  - B+ pT > 3000 MeV: {b_pt_events}/{total_events} ({100.0*b_pt_events/total_events:.1f}%)")
    logger.info(f"  - B+ DTF chi2 < 30: {b_dtf_chi2_events}/{total_events} ({100.0*b_dtf_chi2_events/total_events:.1f}%)")
    logger.info(f"  - B+ DTF status == 0: {b_dtf_status_events}/{total_events} ({100.0*b_dtf_status_events/total_events:.1f}%)")
    logger.info(f"  - B+ IP chi2 < 10: {b_ip_chi2_events}/{total_events} ({100.0*b_ip_chi2_events/total_events:.1f}%)")
    logger.info(f"  - B+ FD chi2 > 175: {b_fd_chi2_events}/{total_events} ({100.0*b_fd_chi2_events/total_events:.1f}%)")
    logger.info(f"  - All Combined: {all_combined_events}/{total_events} ({100.0*all_combined_events/total_events:.1f}%)")

if __name__ == "__main__":
    args = parse_args()
    years = args.years.split(",")
    polarities = args.polarity.split(",")
    track_types = args.track_types.split(",")
    
    analyze_selection_stages(args.data_dir, years, polarities, track_types)