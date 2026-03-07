#!/bin/bash

# ANSI Color Codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script Settings
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
MAIN_SCRIPT="${SCRIPT_DIR}/main.py"
PROJECT_ROOT=$(pwd)

# Function to run a specific optimization option
run_study() {
    local option=$1
    local description=$2

    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${CYAN}Running Option ${option}: ${description}${NC}"
    echo -e "${BLUE}================================================================================${NC}"

    start_time=$(date +%s)
    echo -e "${YELLOW}Start time:${NC} $(date)"

    # Run the python script
    python "$MAIN_SCRIPT" --option "$option"
    exit_code=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo -e "${YELLOW}End time:${NC} $(date)"

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Option ${option} completed successfully in ${duration} seconds.${NC}"
        return 0
    else
        echo -e "${RED}✗ Option ${option} failed with exit code ${exit_code}.${NC}"
        return 1
    fi
}

echo -e "${BLUE}Starting FOM Optimization Automation Suite${NC}"
echo -e "${BLUE}Project Root: ${YELLOW}${PROJECT_ROOT}${NC}"
echo ""

overall_start=$(date +%s)

# Array of options and descriptions
options=("A" "B" "C")
descriptions=("Grouped Optimization" "Per-State Optimization" "Sequential Optimization")

# Record results
declare -a results
declare -a durations

for i in "${!options[@]}"; do
    opt=${options[$i]}
    desc=${descriptions[$i]}

    if run_study "$opt" "$desc"; then
        results[$i]="${GREEN}Success${NC}"
    else
        results[$i]="${RED}Failed${NC}"
        # Decide whether to continue on failure or exit
        echo -e "${RED}Stopping suite due to failure in Option ${opt}.${NC}"
        exit 1
    fi
    durations[$i]=$(( $(date +%s) - start_time ))
    echo ""
done

overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}SUMMARY OF RUNS${NC}"
echo -e "${BLUE}================================================================================${NC}"
for i in "${!options[@]}"; do
    printf "${CYAN}Option %s:${NC} %-25s | Status: %b | Duration: %d seconds\n" \
           "${options[$i]}" "${descriptions[$i]}" "${results[$i]}" "${durations[$i]}"
done
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}All tasks completed in ${overall_duration} seconds.${NC}"
