#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-837 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-42:00:00
#SBATCH --output /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/outputs/MSD_training.out

# Change GPU amount if needed/possible
# Time format: D-HH:MM:SS
# Using 16 workers on A100 (Need A100 due to worker limitation on T4)

scp -r /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/data $TMPDIR/wivegg

mkdir $TMPDIR/wivegg/checkpoints
mkdir $TMPDIR/wivegg/checkpoints/msd

export HF_HOME=$TMPDIR/wivegg/.hf
export HF_HUB_CACHE=$TMPDIR/wivegg/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/wivegg/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/wivegg/.hf/transformers

export DATA_LOCATION=$TMPDIR/wivegg
export CHECKPOINT_DIR=$TMPDIR/wivegg/checkpoints/msd
export WANDB_NAME=standard-prosit-run-9

export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=False

apptainer exec --bind $TMPDIR/wivegg/ /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/dlomix_container.sif python /opt/dlomix/run_scripts/train_prosit_intensity_ptms_torch.py

scp -r $TMPDIR/wivegg/checkpoints/msd /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/outputs/checkpoints
