#!/bin/bash

# Define the directory
DIR="/eos/lhcb/wg/BnoC/Bu2LambdaPPP/MC/DaVinciTuples/restripped.MC"

# Ensure the directory exists
if [[ ! -d "$DIR" ]]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

# List files, replace MC16/MC17/MC18 with MCXX while keeping other parts intact
ls "$DIR" | sed -E 's/MC(16|17|18)(MD|MU)/MCXX/g' | sort -u
