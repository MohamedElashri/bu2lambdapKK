#!/bin/bash
#
# Quick-start script for ηc(2S) vs ψ(2S) discrimination study
#
# Usage:
#   ./run_study.sh           # Run with default settings
#   ./run_study.sh verbose   # Run with verbose output
#   ./run_study.sh custom    # Run with custom options (edit script first)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}           ηc(2S) vs ψ(2S) Discrimination Study${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment from project root
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${GREEN}Activating virtual environment:${NC} $PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo ""
else
    echo -e "${YELLOW}Warning: Virtual environment not found at $PROJECT_ROOT/.venv${NC}"
    echo "Continuing with system Python..."
    echo ""
fi

# Check if Python script exists
if [ ! -f "etac2s_vs_psi2s.py" ]; then
    echo -e "${RED}Error: Study script not found!${NC}"
    echo "Expected: etac2s_vs_psi2s.py"
    exit 1
fi

# Parse command line argument
MODE="${1:-default}"

echo -e "${GREEN}Study directory:${NC} $SCRIPT_DIR"
echo -e "${GREEN}Mode:${NC} $MODE"
echo ""

# Set Python command (try python3 first, then python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Python:${NC} $PYTHON_CMD ($(${PYTHON_CMD} --version))"
echo ""

# Check required packages
echo -e "${YELLOW}Checking required packages...${NC}"
REQUIRED_PACKAGES=("numpy" "scipy" "matplotlib" "awkward")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! ${PYTHON_CMD} -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${RED}Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo "Install with: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi
echo -e "${GREEN}✓ All required packages found${NC}"
echo ""

# Run the study based on mode
case "$MODE" in
    default)
        echo -e "${YELLOW}Running study with default settings...${NC}"
        echo "  - Years: 2016, 2017, 2018"
        echo "  - Track types: LL, DD"
        echo "  - Polarities: Both (MD, MU)"
        echo "  - Bins: 50 (4 MeV/bin in 3550-3750 MeV region)"
        echo ""
        
        ${PYTHON_CMD} etac2s_vs_psi2s.py
        ;;
        
    verbose)
        echo -e "${YELLOW}Running study with verbose output...${NC}"
        echo ""
        
        ${PYTHON_CMD} etac2s_vs_psi2s.py --verbose
        ;;
        
    custom)
        echo -e "${YELLOW}Running study with custom options...${NC}"
        echo ""
        
        # Customize these options as needed
        CUSTOM_YEARS="16 17 18"
        CUSTOM_BINS=60
        CUSTOM_OUTPUT="output/"
        
        echo "  - Years: $CUSTOM_YEARS"
        echo "  - Bins: $CUSTOM_BINS"
        echo "  - Output: $CUSTOM_OUTPUT"
        echo ""
        
        ${PYTHON_CMD} etac2s_vs_psi2s.py \
            --years $CUSTOM_YEARS \
            --bins $CUSTOM_BINS \
            --output "$CUSTOM_OUTPUT" \
            --verbose
        ;;
        
    quick)
        echo -e "${YELLOW}Running quick test with single year...${NC}"
        echo "  - Years: 2018 only (for testing)"
        echo "  - Bins: 40"
        echo ""
        
        ${PYTHON_CMD} etac2s_vs_psi2s.py \
            --years 18 \
            --bins 40 \
            --output "output/studies/etac2s_vs_psi2s_quick"
        ;;
        
    help|--help|-h)
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  default  - Run with standard settings (all years, 50 bins)"
        echo "  verbose  - Run with detailed logging"
        echo "  custom   - Run with customized options (edit script to modify)"
        echo "  quick    - Quick test with single year"
        echo "  help     - Show this help message"
        echo ""
        echo "For more options, run directly:"
        echo "  python etac2s_vs_psi2s.py --help"
        exit 0
        ;;
        
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Use '$0 help' to see available modes"
        exit 1
        ;;
esac

# Check if study completed successfully
EXIT_CODE=$?

echo ""
echo -e "${BLUE}======================================================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Study completed successfully!${NC}"
    echo ""
    echo "Output files:"
    
    # Find output directory
    OUTPUT_DIR=$(ls -td output/etac2s_vs_psi2s* 2>/dev/null | head -1)
    
    if [ -n "$OUTPUT_DIR" ]; then
        echo -e "${GREEN}Directory:${NC} $OUTPUT_DIR"
        echo ""
        
        if [ -f "$OUTPUT_DIR/etac2s_vs_psi2s_summary.pdf" ]; then
            echo -e "  ${GREEN}✓${NC} etac2s_vs_psi2s_summary.pdf (plots)"
        fi
        
        if [ -f "$OUTPUT_DIR/fit_results.txt" ]; then
            echo -e "  ${GREEN}✓${NC} fit_results.txt (numerical results)"
            echo ""
            echo -e "${YELLOW}Quick look at results:${NC}"
            head -30 "$OUTPUT_DIR/fit_results.txt"
        fi
        
        if [ -f "etac2s_vs_psi2s.log" ]; then
            echo ""
            echo -e "  ${GREEN}✓${NC} etac2s_vs_psi2s.log (detailed log)"
        fi
        
        echo ""
        echo -e "${YELLOW}To view the plots:${NC}"
        echo "  evince $OUTPUT_DIR/etac2s_vs_psi2s_summary.pdf"
        echo "  # or"
        echo "  okular $OUTPUT_DIR/etac2s_vs_psi2s_summary.pdf"
    fi
else
    echo -e "${RED}Study failed with exit code $EXIT_CODE${NC}"
    echo "Check the log file for details: etac2s_vs_psi2s.log"
fi

echo -e "${BLUE}======================================================================${NC}"

exit $EXIT_CODE
