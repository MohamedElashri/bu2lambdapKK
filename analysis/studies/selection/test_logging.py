#!/usr/bin/env python3
"""
Test script to show new logging output format
"""
import logging

def setup_logger():
    logger = logging.getLogger('SelectionStudy')
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

print("\n" + "="*80)
print("EXAMPLE: New Logging Output with 2D Grid Optimization")
print("="*80 + "\n")

# Simulated configuration
config = {
    'metadata': {
        'description': 'Selection optimization study for B+ → pK⁻Λ̄ K+ with J/ψ focus'
    },
    'optimization': {
        'perform_2d_grid': True,
        'grid_scan_steps': 10
    }
}

output_dir = "output"

# Initialization messages
logger.info("Created two-phase output structure with 6 subdirectories")
logger.info("="*80)
logger.info("Selection Study Initialized")
logger.info(f"Description: {config['metadata']['description']}")
logger.info(f"Output: {output_dir}")
logger.info("")
logger.info("Optimization Features:")
logger.info("  ✓ 1D Grid Search: Independent variable optimization")
if config.get('optimization', {}).get('perform_2d_grid', True):
    logger.info("  ✓ 2D Grid Search: Multi-dimensional optimization (ENABLED)")
    n_steps = config.get('optimization', {}).get('grid_scan_steps', 10)
    logger.info(f"    Grid resolution: {n_steps} steps per variable")
else:
    logger.info("  ✗ 2D Grid Search: Multi-dimensional optimization (DISABLED)")
logger.info("="*80)

# Main workflow messages
logger.info("="*80)
logger.info("SELECTION OPTIMIZATION STUDY - TWO-PHASE WORKFLOW")
logger.info("="*80)
logger.info(f"Description: {config['metadata']['description']}")
logger.info("")

# Phase 1
logger.info("\n" + "="*80)
logger.info("PHASE 1: MC OPTIMIZATION (1D + 2D Grid Search)")
logger.info("="*80)
logger.info("Goal: Optimize cuts on J/ψ MC to maximize signal efficiency")
logger.info("Strategy 1: 1D grid search per variable")
logger.info("Strategy 2: 2D grid search across all combinations")
logger.info("Output: mc/ directory\n")

# 2D Grid optimization
logger.info("\n" + "="*80)
logger.info("2D GRID SEARCH OPTIMIZATION")
logger.info("="*80)
logger.info("Scanning all combinations of cuts to maximize S/√B")

print("\n" + "="*80)
print("This is how the output will look with the new 2D grid feature!")
print("="*80 + "\n")
