#!/bin/bash
# condor_wrapper.sh

# Script directory (AFS)
BASEDIR="/afs/cern.ch/work/m/melashri/public/reduction_norm"
cd "$BASEDIR"

# EOS data directories
EOS_BASE="/eos/lhcb/user/m/melashri/data/bu2kskpik/RD"
PROCESSED_DIR="${EOS_BASE}/processed"
MERGED_DIR="${EOS_BASE}/merged"

# Local logs directory
mkdir -p logs

# Ensure EOS directories exist
eos mkdir -p "${PROCESSED_DIR}" "${MERGED_DIR}" 2>/dev/null || true

DS_NAME="$1"
echo "=== [$(date)] PROCESS ${DS_NAME} ==="

# Run processing
bash run_processing.sh process "${DS_NAME}" "${PROCESSED_DIR}/${DS_NAME}"
RC1=$?
if [ $RC1 -ne 0 ]; then
  echo ">>> PROCESS failed for ${DS_NAME} (exit $RC1)" >&2
  exit $RC1
fi

echo "=== [$(date)] MERGE ${DS_NAME} ==="

# Run merging
bash run_processing.sh merge "${DS_NAME}" "${PROCESSED_DIR}/${DS_NAME}" "${MERGED_DIR}"
RC2=$?
if [ $RC2 -ne 0 ]; then
  echo ">>> MERGE failed for ${DS_NAME} (exit $RC2)" >&2
  exit $RC2
fi

echo "=== [$(date)] DONE ${DS_NAME} ==="
exit 0