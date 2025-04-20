#!/bin/bash

# Master script to run sequential processing in tmux
# Save this as "start_processing.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first."
    exit 1
fi

# Create a script that will run the sequential processing directly
cat > "${SCRIPT_DIR}/sequential_processor.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Array of configurations to process
CONFIGS=(
    "2015_magup"
    "2015_magdown"
    "2016_magdown"
    "2016_magup"
    "2017_magdown"
    "2017_magup"
    "2018_magdown"
    "2018_magup"
)

# Declare dataset mappings
declare -A DATASETS=(
    ["2015_magup"]="00085557 /eos/lhcb/grid/prod/lhcb/LHCb/Collision15/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085557/0000"
    ["2015_magdown"]="00085559 /eos/lhcb/grid/prod/lhcb/LHCb/Collision15/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085559/0000"
    ["2016_magdown"]="00085555 /eos/lhcb/grid/prod/lhcb/LHCb/Collision16/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085555/0000"
    ["2016_magup"]="00085553 /eos/lhcb/grid/prod/lhcb/LHCb/Collision16/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085553/0000"
    ["2017_magdown"]="00085551 /eos/lhcb/grid/prod/lhcb/LHCb/Collision17/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085551/0000"
    ["2017_magup"]="00085549 /eos/lhcb/grid/prod/lhcb/LHCb/Collision17/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085549/0000"
    ["2018_magdown"]="00085547 /eos/lhcb/grid/prod/lhcb/LHCb/Collision18/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085547/0000"
    ["2018_magup"]="00085545 /eos/lhcb/grid/prod/lhcb/LHCb/Collision18/BHADRON_B2KSHHH_DVNTUPLE.ROOT/00085545/0000"
)

# Source virtual environment if it exists
if [ -d "${SCRIPT_DIR}/venv" ]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
    echo "Activated virtual environment"
else
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/venv"
    source "${SCRIPT_DIR}/venv/bin/activate"
    pip install uproot numpy tqdm psutil
    echo "Virtual environment created and activated"
fi

echo "Starting sequential processing of all datasets..."

# Create necessary directories
mkdir -p "${SCRIPT_DIR}/processed"
mkdir -p "${SCRIPT_DIR}/merged"

# Process each configuration sequentially
for config in "${CONFIGS[@]}"; do
    echo "==============================================="
    echo "Starting processing for $config"
    echo "==============================================="
    
    # Get dataset info
    if [ -n "${DATASETS[$config]}" ]; then
        read -r dataset_id dataset_path <<< "${DATASETS[$config]}"
        echo "Dataset ID: $dataset_id"
        echo "Dataset Path: $dataset_path"
        
        # Create output directory for this config
        output_dir="${SCRIPT_DIR}/processed/${config}"
        mkdir -p "$output_dir"
        
        # Process all files in this dataset
        echo "Running process_dataset.py..."
        python3 "${SCRIPT_DIR}/process_dataset.py" "$dataset_id" "$dataset_path" "$output_dir"
        
        echo "Processing for $config completed. Starting merge..."
        
        # Merge files for this configuration
        echo "Running merge_dataset.py..."
        python3 "${SCRIPT_DIR}/merge_dataset.py" "$config" "$output_dir" "${SCRIPT_DIR}/merged"
        
        echo "==============================================="
        echo "Completed processing and merging for $config"
        echo "==============================================="
    else
        echo "Dataset $config not found, skipping"
    fi
    
    # Optional: Add a delay between configurations to allow system resources to stabilize
    sleep 30
done

echo "All processing and merging completed!"
echo "Merged files are available in: ${SCRIPT_DIR}/merged/"

# Print summary of the results
echo "Summary of processed files:"
for config in "${CONFIGS[@]}"; do
    merged_file="${SCRIPT_DIR}/merged/${config}.root"
    
    if [ -f "$merged_file" ]; then
        size=$(du -h "$merged_file" | cut -f1)
        echo "- $config: $size"
    else
        echo "- $config: Not found or processing failed"
    fi
done

# Keep the session open for inspection
echo "Process completed at $(date)"
echo "Press Enter to close this session"
read
EOF

chmod +x "${SCRIPT_DIR}/sequential_processor.sh"

# Start a new tmux session for the sequential processing
echo "Starting tmux session 'root_process'..."
tmux new-session -d -s "root_process" "${SCRIPT_DIR}/sequential_processor.sh"

# Check if the session was created successfully
if tmux has-session -t "root_process" 2>/dev/null; then
    echo "Sequential processing started in tmux session 'root_process'"
    echo "You can attach to this session with: tmux attach -t root_process"
    echo "You can safely disconnect from the server now, and the processing will continue."
    echo "To detach from the tmux session after attaching, press Ctrl+B then D"
else
    echo "Failed to start tmux session. Running the script directly..."
    ${SCRIPT_DIR}/sequential_processor.sh
fi
