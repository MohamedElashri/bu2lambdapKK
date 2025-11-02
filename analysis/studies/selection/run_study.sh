#!/bin/bash
# ============================================================================
# Selection Study Execution Script
# B+ → pK⁻Λ̄ K+ Analysis - J/ψ Focus
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Root directory (bu2lambdapKK, where .venv is located)
ROOT_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"
VENV_PATH="$ROOT_DIR/.venv"

# Configuration
CONFIG_FILE="${1:-config.toml}"
PYTHON_SCRIPT="main.py"
OUTPUT_DIR="output"
LOG_FILE="$OUTPUT_DIR/run_study.log"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

print_header "Selection Study - Pre-flight Checks"

# Check and activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    print_info "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    print_info "✓ Virtual environment activated"
else
    print_warning "Virtual environment not found at: $VENV_PATH"
    print_info "Proceeding with system Python..."
fi

# Check Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi
print_info "✓ Python script found: $PYTHON_SCRIPT"

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Usage: $0 [config_file.toml]"
    exit 1
fi
print_info "✓ Config file found: $CONFIG_FILE"

# Check Python 3 available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
print_info "✓ Python 3 found: $(python3 --version)"

# Check required Python modules
print_info "Checking Python dependencies..."
python3 -c "import numpy, awkward, matplotlib, mplhep, tomli" 2>/dev/null
if [ $? -eq 0 ]; then
    print_info "✓ All Python dependencies available"
else
    print_error "Missing Python dependencies"
    print_info "Required: numpy, awkward, matplotlib, mplhep, tomli"
    print_info "Install with: pip install numpy awkward matplotlib mplhep tomli"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_info "✓ Output directory: $OUTPUT_DIR"

echo ""

# ============================================================================
# Execution
# ============================================================================

print_header "Running Selection Study"

print_info "Study started at: $(date)"
print_info "Configuration: $CONFIG_FILE"
print_info "Logging to: $LOG_FILE"
echo ""

# Run the study
print_info "Executing two-phase workflow with 2D grid optimization:"
print_info "  Phase 1: MC Optimization"
print_info "           - 1D grid search on J/ψ signal MC (12 variables)"
print_info "           - 2D grid optimization (all combinations)"
print_info "           - Maximize S/√B to find optimal cuts"
print_info "  Phase 2: Data Application - Apply optimal cuts to real data"
print_info "           Generate mass spectrum and yield estimates"
echo ""

# Execute with output to both console and log file
python3 "$PYTHON_SCRIPT" -c "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

echo ""
print_header "Execution Complete"

if [ $EXIT_CODE -eq 0 ]; then
    print_info "✓ Study completed successfully!"
    print_info "Study finished at: $(date)"
    echo ""
    
    # Report outputs
    print_header "Output Files"
    
    if [ -d "$OUTPUT_DIR" ]; then
        # Count output files
        PDF_COUNT=$(find "$OUTPUT_DIR" -name "*.pdf" 2>/dev/null | wc -l)
        PNG_COUNT=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
        
        print_info "Generated outputs:"
        print_info "  PDF plots: $PDF_COUNT"
        print_info "  PNG plots: $PNG_COUNT"
        print_info "  Log file: $LOG_FILE"
        echo ""
        
        print_info "Output directory structure:"
        tree -L 2 "$OUTPUT_DIR" 2>/dev/null || ls -lh "$OUTPUT_DIR"
    fi
    
    echo ""
    print_info "To view results:"
    print_info "  cd $OUTPUT_DIR"
    print_info "  ls *.pdf"
    print_info "  open jpsi_mass.pdf  # or your PDF viewer"
    
else
    print_error "Study failed with exit code: $EXIT_CODE"
    print_error "Check log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi

echo ""
print_header "Done!"
