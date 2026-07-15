# Change GPU amount if needed/possible
# Time format: D-HH:MM:SS
# Using 16 workers on A100 (Need A100 due to worker limitation on T4)

scp -r /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/data $TMPDIR/wivegg

export HF_HOME=$TMPDIR/wivegg/.hf
export HF_HUB_CACHE=$TMPDIR/wivegg/.hf/hub
export HF_DATASETS_CACHE=$TMPDIR/wivegg/.hf/datasets
export TRANSFORMERS_CACHE=$TMPDIR/wivegg/.hf/transformers

export DATA_LOCATION=$TMPDIR/wivegg
export CHECKPOINT_DIR=/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/outputs/checkpoints
export WANDB_NAME=ua-metric-test-1

export NUM_WORKERS=16
export DLOMIX_BACKEND=pytorch
export UNCERTAINTY_AWARE=True

apptainer exec --bind $TMPDIR/wivegg/,/mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/storage/outputs/checkpoints /mimer/NOBACKUP/groups/youcan-from-2023-232/users/wivegg/dlomix_container.sif python /opt/dlomix/run_scripts/train_prosit_intensity_ptms_torch.py