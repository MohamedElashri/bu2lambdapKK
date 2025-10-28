#!/usr/bin/env python3
"""
Integration test for the complete analysis system

Tests that all modules work together with the new branch configuration system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Import modules directly
from data_loader import DataLoader
from mc_loader import MCLoader
from branch_config import BranchConfig
from selection import SelectionProcessor
from mass_calculator import MassCalculator
from efficiency import EfficiencyCalculator

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_branch_configuration():
    """Test branch configuration system"""
    logger.info("Testing branch configuration...")
    
    config = BranchConfig()
    assert len(config.list_available_sets()) > 0
    assert len(config.list_available_presets()) > 0
    
    # Test presets
    for preset in ['minimal', 'standard', 'full_data', 'mc_reco']:
        branches = config.get_branches_from_preset(preset, exclude_mc=True)
        assert len(branches) > 0, f"Preset {preset} returned no branches"
    
    logger.info("✓ Branch configuration tests passed")

def test_data_loader():
    """Test data loader with branch configuration"""
    logger.info("\nTesting data loader...")
    
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    config = BranchConfig()
    loader = DataLoader(data_dir, config)
    
    # Load minimal data for quick test
    data = loader.load_data(
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm',
        preset='minimal'
    )
    
    assert len(data) > 0, "No data loaded"
    assert '16_MD_LL' in data, "Expected dataset not found"
    
    logger.info(f"✓ Data loader test passed ({len(data['16_MD_LL'])} events)")

def test_mc_loader():
    """Test MC loader with branch configuration"""
    logger.info("\nTesting MC loader...")
    
    mc_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc"
    config = BranchConfig()
    loader = MCLoader(mc_dir, config)
    
    # Test reconstructed MC loading
    mc_reco = loader.load_reconstructed(
        sample_name='KpKm',
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm',
        preset='minimal'
    )
    
    assert len(mc_reco) > 0, "No MC data loaded"
    assert '16_MD_LL' in mc_reco, "Expected MC dataset not found"
    
    logger.info(f"✓ MC loader test passed ({len(mc_reco['16_MD_LL'])} events)")

def test_selection_processor():
    """Test selection processor with loaded data"""
    logger.info("\nTesting selection processor...")
    
    # Load data
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    config = BranchConfig()
    loader = DataLoader(data_dir, config)
    
    # Use standard preset which includes selection branches
    data = loader.load_data(
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm',
        preset='standard'
    )
    
    # Test selection
    selector = SelectionProcessor()
    selected_data = selector.apply_basic_selection(data)
    
    assert len(selected_data) > 0, "Selection returned no data"
    assert '16_MD_LL' in selected_data, "Expected dataset not in selection output"
    
    initial_events = len(data['16_MD_LL'])
    selected_events = len(selected_data['16_MD_LL'])
    efficiency = 100 * selected_events / initial_events if initial_events > 0 else 0
    
    logger.info(f"✓ Selection test passed ({selected_events}/{initial_events} events, {efficiency:.1f}% efficiency)")

def test_mass_calculator():
    """Test mass calculator"""
    logger.info("\nTesting mass calculator...")
    
    # Load and select data
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    config = BranchConfig()
    loader = DataLoader(data_dir, config)
    
    # Need kinematics for mass calculation
    data = loader.load_data(
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm',
        preset='standard'  # Includes kinematics
    )
    
    # Calculate masses
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(data)
    
    assert len(data_with_masses) > 0, "Mass calculation returned no data"
    
    # Check that mass was calculated
    first_dataset = next(iter(data_with_masses.values()))
    assert 'M_pKLambdabar' in first_dataset.fields, "Mass branch not found"
    
    logger.info("✓ Mass calculator test passed")

def test_efficiency_calculator():
    """Test efficiency calculator"""
    logger.info("\nTesting efficiency calculator...")
    
    mc_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/mc"
    eff_calc = EfficiencyCalculator(mc_dir)
    
    # Test basic functionality
    efficiency = eff_calc.calculate_efficiency()
    assert efficiency > 0, "Efficiency calculation failed"
    
    total_yield = eff_calc.estimate_total_jpsi_yield(['16'])
    assert total_yield > 0, "Yield estimation failed"
    
    logger.info("✓ Efficiency calculator test passed")

def test_integration():
    """Test full integration workflow"""
    logger.info("\nTesting full integration workflow...")
    
    # Complete workflow: load -> select -> calculate masses
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
    config = BranchConfig()
    
    # Step 1: Load
    loader = DataLoader(data_dir, config)
    data = loader.load_data(
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm',
        preset='standard'  # Includes all needed branches
    )
    
    logger.info(f"  Loaded {len(data['16_MD_LL'])} events")
    
    # Step 2: Select
    selector = SelectionProcessor()
    selected_data = selector.apply_basic_selection(data)
    
    logger.info(f"  Selected {len(selected_data['16_MD_LL'])} events")
    
    # Step 3: Calculate masses
    mass_calc = MassCalculator()
    data_with_masses = mass_calc.calculate_jpsi_candidates(selected_data)
    
    logger.info(f"  Calculated masses for {len(data_with_masses['16_MD_LL'])} events")
    
    assert len(data_with_masses) > 0, "Integration workflow failed"
    
    logger.info("✓ Full integration test passed")

def main():
    """Run all tests"""
    print("="*70)
    print("RUNNING INTEGRATION TESTS")
    print("="*70)
    
    try:
        test_branch_configuration()
        test_data_loader()
        test_mc_loader()
        test_selection_processor()
        test_mass_calculator()
        test_efficiency_calculator()
        test_integration()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe analysis system is working correctly with the new")
        print("branch configuration system. You can now:")
        print("  - Run main.py for full analysis")
        print("  - Run resonances.py for resonance studies")
        print("  - Run event_loss.py for diagnostic analysis")
        print("  - Use custom branch configurations via branches_config.toml")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
