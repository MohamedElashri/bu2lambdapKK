#!/bin/bash

#############################################################################
#
# ROOT File Merging Script
# 
# PURPOSE:
# This script merges ROOT files containing Monte Carlo simulation data
# for B meson decays, organizing them by year and decay type.
#
# DATA STRUCTURE:
# - Base directory: /share/lazy/Mohamed/bu2kskpik/MC/data/data
# - Year folders: mc_2015, mc_2016, mc_2017, mc_2018
# - Each file contains Monte Carlo simulation data for B meson decays
# - Files are identified by decay type codes in their names
#
# DECAY TYPES:
# | Decay Code | Description                             | Output Filename           |
# |------------|-----------------------------------------|---------------------------|
# | 12105160   | B+ → (K0_S → π+π-)K-π+K+                | B2K0s2PipPimKmPipKp      |
# | 12135100   | B+ → (J/ψ → (K0_S → π+π-)K-π+)K+        | B2Jpsi2K0s2PipPimKmPipKp |
# | 12135102   | B+ → (ηc → (K0_S → π+π-)K-π+)K+         | B2Etac2K0s2PipPimKmPipKp |
# | 12135104   | B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+     | B2Etac2S2K0s2PipPimKmPipKp |
# | 12135106   | B+ → (χc1 → (K0_S → π+π-)K-π+)K+        | B2Chic12K0s2PipPimKmPipKp |
#
# TTREE FILTERING:
# The script will keep only the following TTrees from each file and rename them:
# | Original TTree Name    | New TTree Name     |
# |------------------------|---------------------|
# | KSKmKpPip_DD_Tuple     | KSKmKpPip_DD       |
# | KSKmKpPip_LL_Tuple     | KSKmKpPip_LL       |
# | KSKpKpPim_DD_Tuple     | KSKpKpPim_DD       |
# | KSKpKpPim_LL_Tuple     | KSKpKpPim_LL       |
#
# USAGE:
# 1. Make the script executable: chmod +x merge_root_files.sh
# 2. Run the script: ./merge_root_files.sh
#
# OUTPUT:
# - Creates a 'merged' directory inside the base data directory
# - Generates merged ROOT files named with pattern {year}_{decay_description}.root
# - Creates a log file with statistics about the merged files
#
# DEPENDENCIES:
# - Requires the CERN ROOT 'hadd' command to be available in your PATH
#
# NOTES:
# - For large files, merging may take significant time and memory
# - Consider running in a screen or tmux session for large merges
#############################################################################

# Base directory where all the data is located
BASE_DIR="/share/lazy/Mohamed/bu2kskpik/MC/data/data"
OUTPUT_DIR="${BASE_DIR}/merged"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/merge_log.txt"
echo "Starting ROOT file merge process at $(date)" > $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Decay code to name mapping
declare -A DECAY_NAMES
DECAY_NAMES["12105160"]="B2K0s2PipPimKmPipKp"
DECAY_NAMES["12135100"]="B2Jpsi2K0s2PipPimKmPipKp"
DECAY_NAMES["12135102"]="B2Etac2K0s2PipPimKmPipKp"
DECAY_NAMES["12135104"]="B2Etac2S2K0s2PipPimKmPipKp"
DECAY_NAMES["12135106"]="B2Chic12K0s2PipPimKmPipKp"

# Decay descriptions for logging
declare -A DECAY_DESCRIPTIONS
DECAY_DESCRIPTIONS["12105160"]="B+ → (K0_S → π+π-)K-π+K+"
DECAY_DESCRIPTIONS["12135100"]="B+ → (J/ψ → (K0_S → π+π-)K-π+)K+"
DECAY_DESCRIPTIONS["12135102"]="B+ → (ηc → (K0_S → π+π-)K-π+)K+"
DECAY_DESCRIPTIONS["12135104"]="B+ → (ηc(2S) → (K0_S → π+π-)K-π+)K+"
DECAY_DESCRIPTIONS["12135106"]="B+ → (χc1 → (K0_S → π+π-)K-π+)K+"

# Function to find files for a specific year and decay code
function find_files() {
    local year=$1
    local decay_code=$2
    
    if [[ "$year" == "mc_2018" ]]; then
        # Special handling for 2018 which has a nested structure
        find "${BASE_DIR}/${year}" -type f -name "*${decay_code}*dvntuple.root"
    else
        # Standard handling for other years
        find "${BASE_DIR}/${year}" -maxdepth 1 -type f -name "*${decay_code}*dvntuple.root"
    fi
}

# Print script header
echo "==============================================" | tee -a $LOG_FILE
echo "  ROOT File Merging Script for B Decays" | tee -a $LOG_FILE
echo "==============================================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Process each year and decay combination
for year in mc_2015 mc_2016 mc_2017 mc_2018; do
    year_num=${year#mc_}
    echo "Processing $year (${year_num})" | tee -a $LOG_FILE
    echo "--------------------------------------------" | tee -a $LOG_FILE
    
    for decay_code in 12105160 12135100 12135102 12135104 12135106; do
        decay_name=${DECAY_NAMES[$decay_code]}
        decay_desc=${DECAY_DESCRIPTIONS[$decay_code]}
        
        echo "  Decay code: $decay_code" | tee -a $LOG_FILE
        echo "  Description: $decay_desc" | tee -a $LOG_FILE
        
        # Create temporary file with list of files to merge
        temp_list=$(mktemp)
        find_files "$year" "$decay_code" > "$temp_list"
        
        file_count=$(wc -l < "$temp_list")
        if [[ $file_count -gt 0 ]]; then
            output_file="${OUTPUT_DIR}/${year_num}_${decay_name}.root"
            
            echo "  Found $file_count files to merge into $output_file" | tee -a $LOG_FILE
            
            # Display first few files for verification
            if [[ $file_count -gt 5 ]]; then
                echo "  Sample of files (showing 5 of $file_count):" | tee -a $LOG_FILE
                head -n 5 "$temp_list" | sed 's/^/    - /' | tee -a $LOG_FILE
                echo "    - ... and $(($file_count - 5)) more files" | tee -a $LOG_FILE
            else
                echo "  Files to merge:" | tee -a $LOG_FILE
                cat "$temp_list" | sed 's/^/    - /' | tee -a $LOG_FILE
            fi
            
            # Create a temporary file for the ROOT selection options
            temp_options=$(mktemp)
            cat > "$temp_options" << EOF
KSKmKpPip_DD_Tuple KSKmKpPip_DD
KSKmKpPip_LL_Tuple KSKmKpPip_LL
KSKpKpPim_DD_Tuple KSKpKpPim_DD
KSKpKpPim_LL_Tuple KSKpKpPim_LL
EOF
            
            # Merge the ROOT files with tree selection and renaming
            echo "  Merging files... (this may take some time)" | tee -a $LOG_FILE
            hadd -f "$output_file" $(cat "$temp_list") >> $LOG_FILE 2>&1
            merge_status=$?
            
            if [[ $merge_status -eq 0 ]]; then
                # Create a temporary file that will have only the trees we want
                temp_output="${output_file}.temp"
                
                echo "  Filtering and renaming TTrees..." | tee -a $LOG_FILE
                
                # Create ROOT script to filter and rename trees
                temp_script=$(mktemp)
                cat > "$temp_script" << EOF
{
    // Open the input file
    TFile *inFile = TFile::Open("$output_file", "READ");
    if (!inFile || inFile->IsZombie()) {
        cout << "Error: Cannot open input file: $output_file" << endl;
        return 1;
    }
    
    // Create output file
    TFile *outFile = TFile::Open("$temp_output", "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        cout << "Error: Cannot create output file: $temp_output" << endl;
        return 1;
    }
    
    // List of trees to keep and their new names
    const char* treesToKeep[] = {
        "KSKmKpPip_DD_Tuple", "KSKmKpPip_DD",
        "KSKmKpPip_LL_Tuple", "KSKmKpPip_LL",
        "KSKpKpPim_DD_Tuple", "KSKpKpPim_DD", 
        "KSKpKpPim_LL_Tuple", "KSKpKpPim_LL"
    };
    int numTrees = sizeof(treesToKeep) / sizeof(const char*) / 2;
    
    bool anyTreeFound = false;
    
    // Copy and rename selected trees
    for (int i = 0; i < numTrees; i++) {
        TTree *oldTree = (TTree*)inFile->Get(treesToKeep[i*2]);
        if (oldTree) {
            anyTreeFound = true;
            outFile->cd();
            TTree *newTree = oldTree->CloneTree();
            newTree->SetName(treesToKeep[i*2+1]);
            newTree->Write();
            cout << "Processed tree: " << treesToKeep[i*2] << " -> " << treesToKeep[i*2+1] << endl;
        }
    }
    
    outFile->Close();
    inFile->Close();
    
    return anyTreeFound ? 0 : 1;
}
EOF
                
                # Execute the ROOT script
                root -l -b -q "$temp_script" >> $LOG_FILE 2>&1
                filter_status=$?
                rm "$temp_script" "$temp_options"
                
                if [[ $filter_status -eq 0 ]]; then
                    # Replace the original merged file with the filtered one
                    mv "$temp_output" "$output_file"
                    
                    # Get file size
                    size_mb=$(du -m "$output_file" | cut -f1)
                    echo "  ✓ Successfully created and filtered $output_file (${size_mb} MB)" | tee -a $LOG_FILE
                else
                    echo "  ✗ WARNING: File created but no matching trees found to filter" | tee -a $LOG_FILE
                    rm -f "$temp_output"
                    
                    # Get file size of original merged file
                    size_mb=$(du -m "$output_file" | cut -f1)
                    echo "  ✓ Merged file created without filtering: $output_file (${size_mb} MB)" | tee -a $LOG_FILE
                fi
            else
                echo "  ✗ ERROR: Failed to create $output_file" | tee -a $LOG_FILE
            fi
        else
            echo "  No files found for $year with decay code $decay_code" | tee -a $LOG_FILE
        fi
        
        rm "$temp_list"
        echo "" | tee -a $LOG_FILE
    done
    
    echo "" | tee -a $LOG_FILE
done

echo "Merging process completed at $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Summary of created files
echo "==============================================" | tee -a $LOG_FILE
echo "  Merged File Statistics" | tee -a $LOG_FILE
echo "==============================================" | tee -a $LOG_FILE
total_size=0
total_files=0

for file in "${OUTPUT_DIR}"/*.root; do
    if [[ -f "$file" ]]; then
        size=$(du -m "$file" | cut -f1)
        echo "$(basename "$file"): ${size} MB" | tee -a $LOG_FILE
        total_size=$((total_size + size))
        total_files=$((total_files + 1))
    fi
done

echo "--------------------------------------------" | tee -a $LOG_FILE
echo "Total merged files: $total_files" | tee -a $LOG_FILE
echo "Total size: ${total_size} MB" | tee -a $LOG_FILE
echo "==============================================" | tee -a $LOG_FILE

# Print some helpful information
echo ""
echo "Log file has been saved to: $LOG_FILE"
echo "Merged files are located in: $OUTPUT_DIR"
echo ""