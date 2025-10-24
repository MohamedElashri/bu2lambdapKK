#!/usr/bin/env python3
"""
Example script demonstrating the flexible selection system

This shows how to:
1. Load data
2. Apply selections with different cut sets (tight vs loose)
3. Get detailed cut summaries
4. Compare efficiencies between cut sets
"""

import logging
import sys
from pathlib import Path

from data_loader import DataLoader
from selection import SelectionProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("Bu2LambdaPKK.Example")


def main():
    """Example usage of the flexible selection system"""
    
    logger.info("="*70)
    logger.info("Example: Flexible Selection System")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    data_dir = "/share/lazy/Mohamed/Bu2LambdaPPP/RD/restripped.data/reduced"
    loader = DataLoader(data_dir=data_dir)
    
    # Load a subset for quick testing
    data = loader.load_data(
        years=['16'],
        polarities=['MD'],
        track_types=['LL'],
        channel_name='B2L0barPKpKm'
    )
    
    if not data:
        logger.error("No data loaded!")
        return
    
    total_events = sum(len(events) for events in data.values())
    logger.info(f"Loaded {total_events} events")
    
    # Example 1: Apply tight cuts (default)
    logger.info("\n" + "="*70)
    logger.info("Example 1: Applying TIGHT cuts (default)")
    logger.info("="*70)
    
    selector_tight = SelectionProcessor()  # Uses default 'tight' from config
    selected_tight, summary_tight = selector_tight.apply_basic_selection(data, return_summary=True)
    
    # Print summary
    selector_tight.print_cut_summary(summary_tight)
    
    # Example 2: Apply loose cuts
    logger.info("\n" + "="*70)
    logger.info("Example 2: Applying LOOSE cuts")
    logger.info("="*70)
    
    # To use loose cuts, we need to create a config with loose cuts enabled
    # For this example, let's read the config and modify it manually
    
    config_path = Path(__file__).parent / "selection.toml"
    
    # Read the config file as text and modify the active cut set
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Replace tight with loose in the active setting
    config_text_loose = config_text.replace('active = "tight"', 'active = "loose"')
    
    # Save temporary config
    temp_config_path = Path(__file__).parent / "selection_loose.toml"
    with open(temp_config_path, 'w') as f:
        f.write(config_text_loose)
    
    # Load with loose cuts
    selector_loose = SelectionProcessor(config_path=temp_config_path)
    selected_loose, summary_loose = selector_loose.apply_basic_selection(data, return_summary=True)
    
    # Print summary
    selector_loose.print_cut_summary(summary_loose)
    
    # Example 3: Compare efficiencies
    logger.info("\n" + "="*70)
    logger.info("Example 3: Comparing TIGHT vs LOOSE cut efficiencies")
    logger.info("="*70)
    
    for key in data.keys():
        tight_final = summary_tight[key]['final_selected']
        loose_final = summary_loose[key]['final_selected']
        
        print(f"\nDataset: {key}")
        print(f"  Initial events: {summary_tight[key]['initial_events']}")
        print(f"  Tight cuts:     {tight_final['passed']} ({tight_final['efficiency']:.2f}%)")
        print(f"  Loose cuts:     {loose_final['passed']} ({loose_final['efficiency']:.2f}%)")
        print(f"  Difference:     {loose_final['passed'] - tight_final['passed']} events "
              f"({loose_final['efficiency'] - tight_final['efficiency']:.2f}%)")
    
    # Clean up temporary config
    temp_config_path.unlink()
    
    logger.info("\n" + "="*70)
    logger.info("Example complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
