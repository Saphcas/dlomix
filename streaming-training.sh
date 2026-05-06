#!/bin/bash
#SBATCH -A naiss2025-22-837
# Add correct time, currently 4h
#SBATCH -t 004:00:00 # HHH:MM:SS
#SBATCH -p alvis
# See if this is correct
#SBATCH -N 1 --gpus-per-node=T4:4
# Getting output and error files
#SBATCH --output=output%].out
#SBATCH --error=error%].error

# Load needed GPU modules?

<run-code>