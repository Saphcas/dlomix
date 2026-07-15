#!/usr/bin/env bash
#SBATCH -A NAISS2026-3-479-gpu -p gpu --gpus=1
#SBATCH -t 42:00:00
#SBATCH -J dlomix-ua-train
#SBATCH --output /nobackup/proj/disk/kall/personal/$USER/logs/%x-%j.out

scp -r /nobackup/proj/disk/kall/shared/datasets/Prosit_PTMs/PTMs_Train $TMPDIR

mkdir $TMPDIR/ua_checkpoints

export HF_HOME=$TMPDIR/.hf
export HF_HUB_CACHE=$TMPDIR/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/.hf/transformers

export DATA_LOCATION=$TMPDIR/PTMs_Train
export CHECKPOINT_DIR=$TMPDIR/ua_checkpoints
export WANDB_NAME=uncertainty-aware

export USE_CLR=True
export LEARNING_RATE=2e-4
export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=120
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

apptainer exec --bind $TMPDIR/ /nobackup/proj/disk/kall/personal/$USER/containers/dlomix-ngc-26.06.sif python /opt/dlomix/run_scripts/train_prosit_intensity_ptms_torch.py

scp -r $TMPDIR/ua_checkpoints /nobackup/proj/disk/kall/personal/$USER/checkpoints/
