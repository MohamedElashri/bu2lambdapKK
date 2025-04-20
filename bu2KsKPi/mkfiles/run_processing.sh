#!/bin/bash

# Master script to process and merge ROOT files

# Setup Python virtual environment if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    echo "Installing required packages..."
    "$VENV_DIR/bin/pip" install uproot numpy tqdm psutil
    
    echo "Virtual environment setup complete"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Create necessary directories
mkdir -p processed
mkdir -p merged

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

# Create a tmux session wrapper script for each process
cat > "${SCRIPT_DIR}/tmux_wrapper.sh" << 'EOF'
#!/bin/bash
# Wrapper script to ensure venv is activated in tmux sessions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

# Execute the command
$@

# Keep the session open for inspection
echo "Process completed with exit code $?"
echo "Press Enter to close this session"
read
EOF

chmod +x "${SCRIPT_DIR}/tmux_wrapper.sh"

# Function to setup a new tmux session for processing
function setup_processing_session() {
    local dataset_name=$1
    local dataset_id=$2
    local dataset_path=$3
    
    # Create output directory
    local output_dir="${SCRIPT_DIR}/processed/${dataset_name}"
    mkdir -p "$output_dir"
    
    # Setup tmux session
    tmux new-session -d -s "process_${dataset_name}" \
        "${SCRIPT_DIR}/tmux_wrapper.sh python3 ${SCRIPT_DIR}/process_dataset.py ${dataset_id} ${dataset_path} ${output_dir}"
    
    echo "Started processing session for ${dataset_name}"
}

# Function to setup a new tmux session for merging
function setup_merging_session() {
    local dataset_name=$1
    
    # Setup tmux session
    tmux new-session -d -s "merge_${dataset_name}" \
        "${SCRIPT_DIR}/tmux_wrapper.sh python3 ${SCRIPT_DIR}/merge_dataset.py ${dataset_name} ${SCRIPT_DIR}/processed/${dataset_name} ${SCRIPT_DIR}/merged"
    
    echo "Started merging session for ${dataset_name}"
}

# Process command line arguments
ACTION=$1
DATASET=$2

if [ "$ACTION" == "process" ]; then
    if [ -z "$DATASET" ]; then
        # Process all datasets
        for ds_name in "${!DATASETS[@]}"; do
            read -r ds_id ds_path <<< "${DATASETS[$ds_name]}"
            setup_processing_session "$ds_name" "$ds_id" "$ds_path"
            # Wait a bit to avoid overloading the system
            sleep 5
        done
    else
        # Process specific dataset
        if [ -n "${DATASETS[$DATASET]}" ]; then
            read -r ds_id ds_path <<< "${DATASETS[$DATASET]}"
            setup_processing_session "$DATASET" "$ds_id" "$ds_path"
        else
            echo "Dataset $DATASET not found"
            exit 1
        fi
    fi
elif [ "$ACTION" == "merge" ]; then
    if [ -z "$DATASET" ]; then
        # Merge all datasets
        for ds_name in "${!DATASETS[@]}"; do
            # Check if processing is complete
            if [ -d "${SCRIPT_DIR}/processed/${ds_name}" ] && [ "$(ls -A "${SCRIPT_DIR}/processed/${ds_name}" | wc -l)" -gt 0 ]; then
                setup_merging_session "$ds_name"
                # Wait a bit to avoid overloading the system
                sleep 5
            else
                echo "No processed files found for ${ds_name}, skipping merge"
            fi
        done
    else
        # Merge specific dataset
        if [ -n "${DATASETS[$DATASET]}" ]; then
            # Check if processing is complete
            if [ -d "${SCRIPT_DIR}/processed/${DATASET}" ] && [ "$(ls -A "${SCRIPT_DIR}/processed/${DATASET}" | wc -l)" -gt 0 ]; then
                setup_merging_session "$DATASET"
            else
                echo "No processed files found for ${DATASET}, skipping merge"
            fi
        else
            echo "Dataset $DATASET not found"
            exit 1
        fi
    fi
else
    echo "Usage: $0 [process|merge] [dataset_name]"
    echo "Available datasets:"
    for ds_name in "${!DATASETS[@]}"; do
        echo "  $ds_name"
    done
    exit 1
fi

echo "All jobs started. Use 'tmux attach -t SESSION_NAME' to view progress."
echo "Available sessions:"
tmux list-sessions

# Function to get dataset ID and path
function get_dataset_info() {
    local dataset_name=$1
    
    if [ -n "${DATASETS[$dataset_name]}" ]; then
        read -r DATASET_ID DATASET_PATH <<< "${DATASETS[$dataset_name]}"
        export DATASET_ID DATASET_PATH
        return 0
    else
        echo "Dataset $dataset_name not found"
        return 1
    fi
}

# Process get_dataset_info command
if [ "$ACTION" == "get_dataset_info" ]; then
    get_dataset_info "$DATASET"
fi
