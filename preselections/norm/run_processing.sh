#!/bin/bash

# Set error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PYTHON_PROCESSES=4  # Default: use 4 parallel processes within Python
LOG_DIR="preselection_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
MAIN_LOG="${LOG_DIR}/processing_${TIMESTAMP}.log"

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

# Default behavior: process all configurations
PROCESS_ALL=true
SPECIFIC_YEAR=""
SPECIFIC_POLARITY=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            PROCESS_ALL=true
            SPECIFIC_YEAR=""
            SPECIFIC_POLARITY=""
            ;;
        --year)
            SPECIFIC_YEAR="$2"
            PROCESS_ALL=false
            shift
            ;;
        --magnet)
            SPECIFIC_POLARITY="$2"
            PROCESS_ALL=false
            shift
            ;;
        --processes)
            PYTHON_PROCESSES="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Function to process data
process_data() {
    local year=$1
    local polarity=$2
    local job_id=$3
    local input_dir=$4
    
    echo "Starting processing for ${year} ${polarity} (Job ID: ${job_id})" | tee -a "$MAIN_LOG"
    
    local output_dir="/eos/lhcb/user/m/melashri/data/bu2kskpik/RD/${year}_${polarity}"
    local log_file="${LOG_DIR}/${year}_${polarity}_${TIMESTAMP}.log"

    # Expand files from directory
    local input_files=("$input_dir"/*.root)

    if [[ ${#input_files[@]} -eq 0 ]]; then
        echo "⚠️ No files found for ${year} ${polarity}! Check the path: ${input_dir}" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    echo "📂 Files to process: ${#input_files[@]}" | tee -a "$MAIN_LOG"
    echo "$(date): Starting ${year} ${polarity}" | tee -a "$MAIN_LOG"

    # Run the Python processing with internal parallelism
    # Pass the PYTHON_PROCESSES value to file_process.py
    python file_process.py \
        --input "${input_files[@]}" \
        --output "${output_dir}" \
        --processes $PYTHON_PROCESSES \
        > "$log_file" 2>&1

    local exit_status=$?

    if [ $exit_status -eq 0 ]; then
        echo "$(date): ✅ Completed ${year} ${polarity} successfully" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "$(date): ❌ Failed processing ${year} ${polarity} with exit status ${exit_status}" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# Start processing
echo "🚀 Starting data processing at $(date)" | tee "$MAIN_LOG"
echo "Using $PYTHON_PROCESSES parallel processes within Python script" | tee -a "$MAIN_LOG"

# Build the list of datasets to process
DATASETS_TO_PROCESS=()

if $PROCESS_ALL; then
    # Process all datasets
    for key in "${!DATASETS[@]}"; do
        DATASETS_TO_PROCESS+=("$key")
    done
else
    # Process only specified datasets
    for key in "${!DATASETS[@]}"; do
        year="${key%_*}"
        polarity="${key#*_}"
        
        if [[ -n "$SPECIFIC_YEAR" && "$SPECIFIC_YEAR" != "$year" ]]; then
            continue
        fi
        
        if [[ -n "$SPECIFIC_POLARITY" && "$SPECIFIC_POLARITY" != "$polarity" ]]; then
            continue
        fi
        
        DATASETS_TO_PROCESS+=("$key")
    done
fi

# Sort datasets chronologically (from 2015 to 2018, magup then magdown)
IFS=$'\n' DATASETS_TO_PROCESS=($(sort <<<"${DATASETS_TO_PROCESS[*]}"))
unset IFS

echo "Datasets to process (in order):" | tee -a "$MAIN_LOG"
for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    echo "  - $dataset" | tee -a "$MAIN_LOG"
done

# Process datasets sequentially
for key in "${DATASETS_TO_PROCESS[@]}"; do
    year="${key%_*}"
    polarity="${key#*_}"
    job_id_input_path=(${DATASETS[$key]})
    
    echo "===== Processing ${year}_${polarity} =====" | tee -a "$MAIN_LOG"
    
    # Process this dataset (no &, run sequentially)
    process_data "$year" "$polarity" "${job_id_input_path[0]}" "${job_id_input_path[1]}"
    
    # Report status after each dataset
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ ${year}_${polarity} completed successfully" | tee -a "$MAIN_LOG" 
    else
        echo "❌ ${year}_${polarity} failed with status $status" | tee -a "$MAIN_LOG"
        
        # Ask whether to continue or abort
        read -p "Dataset failed. Continue processing? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting processing at user request" | tee -a "$MAIN_LOG"
            exit 1
        fi
    fi
    
    echo "-----------------------------------------" | tee -a "$MAIN_LOG"
done

echo "✅ All processing completed at $(date)" | tee -a "$MAIN_LOG"