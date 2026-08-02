#!/bin/bash
#SBATCH -A Berzelius-2026-159
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=ua-head-init
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT=/proj/bedrock/users/x_alfni/dlomix_Saphcas
PYTHON=/proj/bedrock/users/x_alfni/conda-envs/dlomix-torch/bin/python
DATA_DIR=/proj/bedrock/datasets/Prosit_PTMs/PTMs_Train
HEAD_INIT_STATS="${PROJECT_ROOT}/experiment_stats/ua_head_init_full.json"
CHECKPOINT_ROOT=/proj/bedrock/users/x_alfni/checkpoints/ua-head-init

MODES=(
    default
    mean
    presence
    mean-presence
    mean-presence-marginal-variance
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= ${#MODES[@]} )); then
    echo "Invalid array task id: ${TASK_ID}" >&2
    exit 2
fi
if [[ ! -f "${HEAD_INIT_STATS}" ]]; then
    echo "Missing head-initialization statistics: ${HEAD_INIT_STATS}" >&2
    exit 2
fi

MODE=${MODES[$TASK_ID]}
RUN_NAME="ua-head-init-${MODE}-bs${BATCH_SIZE:-4096}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

case "${MODE}" in
    default)
        MEAN_HEAD_INIT=default
        PRESENCE_HEAD_INIT=default
        VARIANCE_HEAD_INIT=default
        ;;
    mean)
        MEAN_HEAD_INIT=empirical
        PRESENCE_HEAD_INIT=default
        VARIANCE_HEAD_INIT=default
        ;;
    presence)
        MEAN_HEAD_INIT=default
        PRESENCE_HEAD_INIT=empirical
        VARIANCE_HEAD_INIT=default
        ;;
    mean-presence)
        MEAN_HEAD_INIT=empirical
        PRESENCE_HEAD_INIT=empirical
        VARIANCE_HEAD_INIT=default
        ;;
    mean-presence-marginal-variance)
        MEAN_HEAD_INIT=empirical
        PRESENCE_HEAD_INIT=empirical
        VARIANCE_HEAD_INIT=empirical_marginal
        ;;
esac

mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_ROOT}"

module load buildenv-gcccuda/12.1.1-gcc12.3.0
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export DATA_LOCATION="${DATA_DIR}"
export DLOMIX_BACKEND=torch
export NUM_WORKERS="${NUM_WORKERS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-4096}"
# Stop after five epochs, but retain the normalization sweep's 30-epoch
# warmup-cosine horizon below so the early learning-rate trajectory is identical.
export N_EPOCHS="${N_EPOCHS:-5}"
export EARLY_STOPPING_PATIENCE=0
export SELECT_BEST_BY_VAL_LOSS=1

export USE_AMP=0
export GRAD_CLIP_MAX_NORM=1.0
export LOSS_NORMALIZATION=component_mean
export VARIANCE_PARAMETERIZATION=log_var
export MIN_VARIANCE=1e-4

export LR_SCHEDULE=warmup_cosine
export WARMUP_COSINE_WARMUP_STEPS="${WARMUP_COSINE_WARMUP_STEPS:-6845}"
export WARMUP_COSINE_START_LR="${WARMUP_COSINE_START_LR:-1.6e-5}"
export WARMUP_COSINE_PEAK_LR="${WARMUP_COSINE_PEAK_LR:-1.2e-4}"
export WARMUP_COSINE_MIN_LR="${WARMUP_COSINE_MIN_LR:-1.6e-5}"
# 13,715 estimated steps/epoch at batch size 4,096 x 30 normalization-sweep
# epochs. Do not infer this from N_EPOCHS=5, as that would change cosine decay.
export WARMUP_COSINE_TOTAL_STEPS="${WARMUP_COSINE_TOTAL_STEPS:-411450}"

export HEAD_INIT_STATS
export MEAN_HEAD_INIT
export PRESENCE_HEAD_INIT
export VARIANCE_HEAD_INIT
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=ua-head-init-ablation
export CHECKPOINT_DIR
export SAVE="${CHECKPOINT_DIR}/best_model.pth"

echo "Starting ${RUN_NAME}"
echo "  init: mean=${MEAN_HEAD_INIT} presence=${PRESENCE_HEAD_INIT} variance=${VARIANCE_HEAD_INIT}"
echo "  batch_size=${BATCH_SIZE} epochs=${N_EPOCHS} normalization=${LOSS_NORMALIZATION}"
echo "  checkpoint=${CHECKPOINT_DIR}"

exec "${PYTHON}" run_scripts/train_prosit_intensity_ptms_torch.py
