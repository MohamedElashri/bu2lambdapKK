#!/bin/bash

# Set strict error handling
set -e
set -u
set -o pipefail

# Default configuration
LOG_DIR="merge_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ALL=true
SPECIFIC_YEAR=""
SPECIFIC_POLARITY=""
SPECIFIC_DECAY=""
MAX_MEMORY=7.0
CHUNK_SIZE=10000
MAX_RETRIES=3

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

MAIN_LOG="${LOG_DIR}/merge_${TIMESTAMP}.log"  # Define MAIN_LOG here

# Create log directory with proper permissions
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    chmod 755 "$LOG_DIR"
fi

# Ensure log file is writable before first use
touch "$MAIN_LOG"
chmod 644 "$MAIN_LOG"

# Help message
show_help() {
    echo -e "${BOLD}Usage: $0 [OPTIONS]${NC}"
    echo
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}-h, --help${NC}                Show this help message"
    echo -e "  ${GREEN}--all${NC}                    Process all years, polarities and decay channels (default)"
    echo -e "  ${GREEN}-y, --year YEAR${NC}          Process specific year (2015-2018)"
    echo -e "  ${GREEN}-m, --magnet POL${NC}         Process specific magnet polarity (magup/magdown)"
    echo -e "  ${GREEN}-d, --decay CHANNELS${NC}     Process specific decay channels (comma-separated, e.g., \"KSKmKpPip,KSKpKpPim\")"
    echo -e "  ${GREEN}--max-memory GB${NC}          Maximum memory usage in GB (default: 7.0)"
    echo -e "  ${GREEN}--chunk-size N${NC}           Number of entries per chunk (default: 10000)"
    echo -e "  ${GREEN}--max-retries N${NC}          Maximum number of retries per failed job (default: 3)"
}

# Function to check disk space
check_disk_space() {
    local dir=$1
    local required_gb=${2:-10}
    local available_gb=$(df -BG "$dir" | awk 'NR==2 {print $4}' | tr -d 'G')
    
    if [ "${available_gb}" -lt "${required_gb}" ]; then
        echo -e "${RED}⚠️ Warning: Low disk space on ${dir}: ${available_gb}GB available${NC}"
        return 1
    fi
    return 0
}

# Function to check memory usage
check_memory_usage() {
    local pid=$1
    local max_gb=$2
    local used_gb=$(ps -o rss= -p "$pid" | awk '{print $1/1024/1024}')
    
    if (( $(echo "$used_gb > $max_gb" | bc -l) )); then
        echo -e "${RED}⚠️ Warning: High memory usage: ${used_gb}GB${NC}"
        return 1
    fi
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --all)
            RUN_ALL=true
            SPECIFIC_YEAR=""
            SPECIFIC_POLARITY=""
            SPECIFIC_DECAY=""
            shift
            ;;
        -y|--year)
            if [[ $# -lt 2 ]] || ! [[ $2 =~ ^201[5-8]$ ]]; then
                echo -e "${RED}Error: --year requires a valid year (2015-2018)${NC}"
                exit 1
            fi
            SPECIFIC_YEAR="$2"
            RUN_ALL=false
            shift 2
            ;;
        -m|--magnet)
            if [[ $# -lt 2 ]] || ! [[ $2 =~ ^mag(up|down)$ ]]; then
                echo -e "${RED}Error: --magnet requires either 'magup' or 'magdown'${NC}"
                exit 1
            fi
            SPECIFIC_POLARITY="$2"
            RUN_ALL=false
            shift 2
            ;;
        -d|--decay)
            if [[ $# -lt 2 ]]; then
                echo -e "${RED}Error: --decay requires a decay channel name${NC}"
                exit 1
            fi
            SPECIFIC_DECAY="$2"
            RUN_ALL=false
            shift 2
            ;;
        --max-memory)
            if [[ $# -lt 2 ]] || ! [[ $2 =~ ^[0-9]+\.?[0-9]*$ ]]; then
                echo -e "${RED}Error: --max-memory requires a valid number${NC}"
                exit 1
            fi
            MAX_MEMORY="$2"
            shift 2
            ;;
        --chunk-size)
            if [[ $# -lt 2 ]] || ! [[ $2 =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: --chunk-size requires a valid integer${NC}"
                exit 1
            fi
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --max-retries)
            if [[ $# -lt 2 ]] || ! [[ $2 =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: --max-retries requires a valid integer${NC}"
                exit 1
            fi
            MAX_RETRIES="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done




# Function to get available decay channels
get_decay_channels() {
    local input_dir=$1
    local pattern="${input_dir}/**/*.root"
    local files=($(ls $pattern 2>/dev/null))
    local channels=()
    for file in "${files[@]}"; do
        channels+=($(basename "$file" .root))
    done
    echo "$(printf "%s\n" "${channels[@]}" | sort -u)"
}

# Function to process data for a specific configuration
process_config() {
    local year=$1
    local polarity=$2
    local decay_list=$3
    local retry_count=0
    local input_dir="/eos/lhcb/user/m/melashri/data/bu2kskpik/RD/${year}_${polarity}"
    local output_dir="/eos/lhcb/user/m/melashri/data/bu2kskpik/RD/merged/${year}_${polarity}"
    local log_file="${LOG_DIR}/${year}_${polarity}_${TIMESTAMP}.log"
    
    echo -e "${CYAN}Processing ${year} ${polarity}${decay_list:+ decays: $decay_list}${NC}" | tee -a "$MAIN_LOG"
    
    if [ ! -d "$input_dir" ]; then
        echo -e "${YELLOW}⚠️ Input directory not found: $input_dir${NC}" | tee -a "$MAIN_LOG"
        return
    fi
    
    # Check disk space
    if ! check_disk_space "$output_dir" 10; then
        echo -e "${RED}❌ Insufficient disk space for ${year} ${polarity}${NC}" | tee -a "$MAIN_LOG"
        return 1
    fi
    
    mkdir -p "$output_dir"
    
    # Build Python command
    local cmd="python merge_root_files.py --input \"$input_dir\" --output \"$output_dir\" --max-memory $MAX_MEMORY --chunk-size $CHUNK_SIZE"
    if [ -n "$decay_list" ]; then
        cmd+=" --decay \"$decay_list\""
    fi
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        # Run the Python merger script
        eval $cmd 2>&1 | tee "$log_file"
        
        local exit_status=${PIPESTATUS[0]}
        
        if [ $exit_status -eq 0 ]; then
            echo -e "${GREEN}✅ Completed merging ${year} ${polarity}${decay_list:+ decays: $decay_list}${NC}" | tee -a "$MAIN_LOG"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo -e "${YELLOW}⚠️ Retry $retry_count/$MAX_RETRIES for ${year} ${polarity}${decay_list:+ decays: $decay_list}${NC}" | tee -a "$MAIN_LOG"
                sleep 5  # Wait before retrying
            else
                echo -e "${RED}❌ Failed merging ${year} ${polarity}${decay_list:+ decays: $decay_list} after $MAX_RETRIES attempts${NC}" | tee -a "$MAIN_LOG"
                return $exit_status
            fi
        fi
    done
}

# List of all possible configurations
YEARS=(2015 2016 2017 2018)
POLARITIES=(magup magdown)

# Print configuration
echo -e "${PURPLE}🚀 Starting merge process at $(date)${NC}" | tee "$MAIN_LOG"
echo -e "${BOLD}Configuration:${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}- Run all: $RUN_ALL${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}- Max memory: ${MAX_MEMORY}GB${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}- Chunk size: $CHUNK_SIZE${NC}" | tee -a "$MAIN_LOG"
echo -e "${BLUE}- Max retries: $MAX_RETRIES${NC}" | tee -a "$MAIN_LOG"
[ -n "$SPECIFIC_YEAR" ] && echo -e "${BLUE}- Year: $SPECIFIC_YEAR${NC}" | tee -a "$MAIN_LOG"
[ -n "$SPECIFIC_POLARITY" ] && echo -e "${BLUE}- Magnet: $SPECIFIC_POLARITY${NC}" | tee -a "$MAIN_LOG"
[ -n "$SPECIFIC_DECAY" ] && echo -e "${BLUE}- Decay: $SPECIFIC_DECAY${NC}" | tee -a "$MAIN_LOG"

# Monitor system resources
monitor_resources() {
    while true; do
        mem_usage=$(free -g | awk 'NR==2{printf "%.1f/%.1f GB", $3, $2}')
        disk_usage=$(df -h "$output_dir" | awk 'NR==2{printf "%s/%s", $4, $2}')
        echo -e "${CYAN}Memory: $mem_usage | Disk: $disk_usage${NC}" >> "$MAIN_LOG"
        sleep 300  # Update every 5 minutes
    done
}

# Start resource monitoring in background
monitor_resources &
MONITOR_PID=$!

# Trap for cleanup
trap 'kill $MONITOR_PID 2>/dev/null' EXIT

# Process configurations based on user input
process_configurations() {
    local year=$1
    local polarity=$2
    local base_dir="/eos/lhcb/user/m/melashri/data/bu2kskpik/RD/${year}_${polarity}"
    
    if [ -n "$SPECIFIC_DECAY" ]; then
        process_config "$year" "$polarity" "$SPECIFIC_DECAY"
    else
        local decay_channels=($(get_decay_channels "$base_dir"))
        if [ ${#decay_channels[@]} -eq 0 ]; then
            echo -e "${YELLOW}No decay channels found in ${base_dir}${NC}" | tee -a "$MAIN_LOG"
            return
        fi
        echo -e "${CYAN}Found ${#decay_channels[@]} decay channels in ${year} ${polarity}${NC}" | tee -a "$MAIN_LOG"
        for decay in "${decay_channels[@]}"; do
            process_config "$year" "$polarity" "$decay"
        done
    fi
}

if $RUN_ALL; then
    echo -e "${CYAN}Processing all configurations${NC}" | tee -a "$MAIN_LOG"
    for year in "${YEARS[@]}"; do
        for polarity in "${POLARITIES[@]}"; do
            process_configurations "$year" "$polarity"
        done
    done
else
    if [ -n "$SPECIFIC_YEAR" ] && [ -n "$SPECIFIC_POLARITY" ]; then
        process_configurations "$SPECIFIC_YEAR" "$SPECIFIC_POLARITY"
    elif [ -n "$SPECIFIC_YEAR" ]; then
        for polarity in "${POLARITIES[@]}"; do
            process_configurations "$SPECIFIC_YEAR" "$polarity"
        done
    elif [ -n "$SPECIFIC_POLARITY" ]; then
        for year in "${YEARS[@]}"; do
            process_configurations "$year" "$SPECIFIC_POLARITY"
        done
    else
        echo -e "${YELLOW}No specific configuration provided, processing all${NC}" | tee -a "$MAIN_LOG"
        for year in "${YEARS[@]}"; do
            for polarity in "${POLARITIES[@]}"; do
                process_configurations "$year" "$polarity"
            done
        done
    fi
fi

# Final status
echo -e "${GREEN}✅ All merging completed at $(date)${NC}" | tee -a "$MAIN_LOG"
