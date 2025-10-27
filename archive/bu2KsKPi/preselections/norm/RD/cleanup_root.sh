#!/bin/bash

# Default root directory is current directory
ROOT_DIR="."

# Check if a directory argument was provided
if [ $# -eq 1 ]; then
    if [ -d "$1" ]; then
        ROOT_DIR="$1"
    else
        echo "Error: Provided path is not a directory"
        echo "Usage: $0 [root_directory]"
        exit 1
    fi
elif [ $# -gt 1 ]; then
    echo "Error: Too many arguments"
    echo "Usage: $0 [root_directory]"
    exit 1
fi

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored text
print_color() {
    local color="$1"
    local text="$2"
    echo -e "${color}${text}${NC}"
}

# Function to print section header
print_header() {
    local text="$1"
    echo ""
    print_color "${BOLD}${PURPLE}" "===================================================="
    print_color "${BOLD}${PURPLE}" "  $text"
    print_color "${BOLD}${PURPLE}" "===================================================="
}

# Files to keep
KEEP_FILES=("KSKmKpPip.root" "KSKpKpPim.root")
# Base directories
BASE_DIRS=("2015_magdown" "2016_magdown" "2017_magdown" "2018_magdown" 
           "2015_magup" "2016_magup" "2017_magup" "2018_magup")

# Stats counters
TOTAL_REMOVED=0
TOTAL_KEPT=0
TOTAL_ERRORS=0

print_header "ROOT File Cleanup Script"
print_color "${YELLOW}" "Working directory: ${ROOT_DIR}"
print_color "${YELLOW}" "This script will keep only the following files in each data directory:"
for file in "${KEEP_FILES[@]}"; do
    print_color "${CYAN}" " - $file"
done
print_color "${YELLOW}" "All other ROOT files will be removed."
echo ""

# Confirmation prompt
read -p "$(echo -e "${BOLD}${RED}Do you want to proceed? (y/n): ${NC}")" CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    print_color "${YELLOW}" "Operation cancelled by user."
    exit 0
fi

# Process each base directory
for base_dir in "${BASE_DIRS[@]}"; do
    # Use the provided root directory
    full_path="$ROOT_DIR/$base_dir"
    if [ ! -d "$full_path" ]; then
        print_color "${RED}" "Directory not found: $full_path - skipping"
        continue
    fi
    
    print_header "Processing $full_path"
    
    # Find all subdirectories (numeric folders)
    SUB_DIRS=$(find "$full_path" -mindepth 1 -maxdepth 1 -type d | sort)
    
    if [ -z "$SUB_DIRS" ]; then
        print_color "${YELLOW}" "No subdirectories found in $base_dir"
        continue
    fi
    
    DIR_COUNT=0
    DIR_TOTAL=$(echo "$SUB_DIRS" | wc -l)
    
    # Process each subdirectory
    for sub_dir in $SUB_DIRS; do
        DIR_COUNT=$((DIR_COUNT + 1))
        sub_dir_name=$(basename "$sub_dir")
        
        print_color "${BLUE}" "[$DIR_COUNT/$DIR_TOTAL] Processing directory: $sub_dir_name"
        
        # Stats for this directory
        DIR_REMOVED=0
        DIR_KEPT=0
        DIR_ERRORS=0
        
        # Find all ROOT files in this directory
        ROOT_FILES=$(find "$sub_dir" -name "*.root" -type f)
        
        for file in $ROOT_FILES; do
            filename=$(basename "$file")
            
            # Check if this file should be kept
            KEEP=0
            for keep_file in "${KEEP_FILES[@]}"; do
                if [[ "$filename" == "$keep_file" ]]; then
                    KEEP=1
                    break
                fi
            done
            
            if [ $KEEP -eq 1 ]; then
                print_color "${GREEN}" "  Keeping: $filename"
                DIR_KEPT=$((DIR_KEPT + 1))
                TOTAL_KEPT=$((TOTAL_KEPT + 1))
            else
                print_color "${RED}" "  Removing: $filename"
                if rm "$file"; then
                    DIR_REMOVED=$((DIR_REMOVED + 1))
                    TOTAL_REMOVED=$((TOTAL_REMOVED + 1))
                else
                    print_color "${RED}" "  Error removing $filename"
                    DIR_ERRORS=$((DIR_ERRORS + 1))
                    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
                fi
            fi
        done
        
        print_color "${CYAN}" "  Summary for $sub_dir_name: Kept $DIR_KEPT files, Removed $DIR_REMOVED files, Errors: $DIR_ERRORS"
    done
done

print_header "Final Summary"
print_color "${GREEN}" "Total files kept: $TOTAL_KEPT"
print_color "${RED}" "Total files removed: $TOTAL_REMOVED"
if [ $TOTAL_ERRORS -gt 0 ]; then
    print_color "${RED}" "Total errors: $TOTAL_ERRORS"
else
    print_color "${GREEN}" "No errors occurred"
fi

# Calculate space saved (approximate)
if command -v du &> /dev/null; then
    print_color "${YELLOW}" "Calculating space saved..."
    CURRENT_SIZE=$(du -sh "$ROOT_DIR" | awk '{print $1}')
    print_color "${GREEN}" "Current directory size: $CURRENT_SIZE"
    print_color "${CYAN}" "Approximately $(($TOTAL_REMOVED * 100 / ($TOTAL_KEPT + $TOTAL_REMOVED)))% of ROOT files were removed"
fi

print_color "${BOLD}${GREEN}" "Operation completed!"