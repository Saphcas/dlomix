#!/bin/bash
#SBATCH -A NAISS2025-22-837 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:8
#SBATCH -t 0-88:00:00 
#SBATCH --o /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/outputs/MSD_training.out

# Change GPU amount if needed/possible
# Time format: D-HH:MM:SS
# For a conservative estimate slowest training is running at 3 batches/s (3.9), and add additional time
# 20 epochs, 27 430 + 7 126 + 1 = 34 557 batches/epoch => total batches = 691 140, 3 batches/s => 64 H 
# Estimate print was 2 h for one epoch training on 8 gpu:s, 40 h only on training, cpu for streaming can be what limits batch/s (only 4 workers)

apptainer shell --bind /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/ /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/dlomix_container.sif
# Should be within container now
export HF_HOME=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/hf_cache
export HF_HUB_CACHE=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/hf_cache/hub
export HF_DATASETS_CACHE=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/hf_cache/datasets
export TRANSFORMERS_CACHE=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/hf_cache/transformers

export DATA_LOCATION=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/data

export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=False

python /opt/dlomix/run_scripts/train_prosit_intensity_ptms_torch.py