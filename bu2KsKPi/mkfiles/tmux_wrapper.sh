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
