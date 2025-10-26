#!/bin/bash
# Script to explore branches in the B+ → pK⁻Λ̄ K+ data files

DATA_DIR="/share/lazy/Mohamed/Bu2LambdaPPP/files/data"
SAMPLE_FILE="dataBu2L0barPHH_16MD_reduced.root"
DECAY_CHANNEL="B2L0barPKpKm"
TRACK_TYPE="LL"

# Check if file exists
if [ ! -f "$DATA_DIR/$SAMPLE_FILE" ]; then
    echo "File not found: $DATA_DIR/$SAMPLE_FILE"
    exit 1
fi

# Print header
echo "===== Exploring branches in $SAMPLE_FILE ====="
echo "Channel: $DECAY_CHANNEL, Track type: $TRACK_TYPE"
echo ""
echo "Running: python3 get_branches.py"
echo ""

# Create Python script
cat > get_branches.py << EOL
#!/usr/bin/env python3
import uproot
import sys
from pathlib import Path

DATA_DIR = "${DATA_DIR}"
SAMPLE_FILE = "${SAMPLE_FILE}"
DECAY_CHANNEL = "${DECAY_CHANNEL}"
TRACK_TYPE = "${TRACK_TYPE}"

def explore_branches():
    file_path = Path(DATA_DIR) / SAMPLE_FILE
    
    try:
        with uproot.open(file_path) as file:
            # List directories in the file
            print(f"Contents of {SAMPLE_FILE}:")
            for key in file.keys():
                print(f"  {key}")
            
            # Access the specific channel
            channel_path = f"{DECAY_CHANNEL}_{TRACK_TYPE}"
            if channel_path not in file:
                print(f"Error: Channel {channel_path} not found in file")
                return
                
            # Access the DecayTree
            tree_path = f"{channel_path}/DecayTree"
            tree = file[tree_path]
            
            # Print tree information
            print(f"\nTree: {tree_path}")
            print(f"Entries: {tree.num_entries}")
            
            # Print branch names
            branches = tree.keys()
            print(f"\nBranches ({len(branches)} total):")
            for i, branch in enumerate(sorted(branches), 1):
                branch_type = tree[branch].typename
                print(f"{i:3d}. {branch:<40} ({branch_type})")
            
            # Print some example values from the first event
            print("\nExample values from first event:")
            first_event = tree.arrays(branches[:5], entry_start=0, entry_stop=1)
            for branch in first_event.fields:
                print(f"{branch}: {first_event[branch][0]}")
                
    except Exception as e:
        print(f"Error accessing file: {e}")

if __name__ == "__main__":
    explore_branches()
EOL

# Make the Python script executable
chmod +x get_branches.py

# Run the Python script
python3 get_branches.py

# Clean up
rm -f get_branches.py

echo ""
echo "===== Branch exploration complete ====="