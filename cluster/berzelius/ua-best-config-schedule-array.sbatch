#!/bin/bash
#SBATCH -A Berzelius-2026-159
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=ua-best-config
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT=/proj/bedrock/users/x_alfni/dlomix_Saphcas
PYTHON=/proj/bedrock/users/x_alfni/conda-envs/dlomix-torch/bin/python
DATA_DIR=/proj/bedrock/datasets/Prosit_PTMs/PTMs_Train
HEAD_INIT_STATS="${PROJECT_ROOT}/experiment_stats/ua_head_init_full.json"
CHECKPOINT_ROOT=/proj/bedrock/users/x_alfni/checkpoints/ua-best-config

SCHEDULES=(clr warmup_cosine)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= ${#SCHEDULES[@]} )); then
    echo "Invalid array task id: ${TASK_ID}" >&2
    exit 2
fi
if [[ ! -f "${HEAD_INIT_STATS}" ]]; then
    echo "Missing head-initialization statistics: ${HEAD_INIT_STATS}" >&2
    exit 2
fi

SCHEDULE=${SCHEDULES[$TASK_ID]}
RUN_NAME="ua-best-${SCHEDULE}-bs1024-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_ROOT}"

module load buildenv-gcccuda/12.1.1-gcc12.3.0
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export DATA_LOCATION="${DATA_DIR}"
export DLOMIX_BACKEND=torch
export NUM_WORKERS=16
export BATCH_SIZE=1024
export N_EPOCHS=30
export EARLY_STOPPING_PATIENCE=0
export SELECT_BEST_BY_VAL_LOSS=1
export USE_AMP=0
export GRAD_CLIP_MAX_NORM=1.0

# Selected uncertainty-aware configuration.
export LOSS_NORMALIZATION=component_mean
export VARIANCE_PARAMETERIZATION=softplus_variance
export MIN_VARIANCE=1e-3
export BETA_NLL_BETA=0.5
export FAITHFUL_MODE=off
export VARIANCE_PRIOR_WEIGHT=0
unset VARIANCE_PRIOR_STATS
export MEAN_HEAD_INIT=empirical
export PRESENCE_HEAD_INIT=empirical
export VARIANCE_HEAD_INIT=empirical_marginal
export HEAD_INIT_STATS

if [[ "${SCHEDULE}" == "clr" ]]; then
    export LR_SCHEDULE=clr
    export USE_CLR=True
    export CLR_BASE_LR=1e-5
    export CLR_MAX_LR=4e-4
    export LEARNING_RATE=2e-4
else
    export LR_SCHEDULE=warmup_cosine
    export WARMUP_COSINE_START_LR=1.6e-5
    export WARMUP_COSINE_PEAK_LR=1.2e-4
    export WARMUP_COSINE_MIN_LR=1.6e-5
    export WARMUP_COSINE_TOTAL_STEPS=0
fi

export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=ua-best-config-berzelius
export CHECKPOINT_DIR
export SAVE="${CHECKPOINT_DIR}/best_model.pth"

echo "Starting ${RUN_NAME}"
echo "  schedule=${LR_SCHEDULE} batch_size=${BATCH_SIZE} epochs=${N_EPOCHS}"
echo "  beta=${BETA_NLL_BETA} variance=${VARIANCE_PARAMETERIZATION} floor=${MIN_VARIANCE}"
echo "  init: mean=${MEAN_HEAD_INIT} presence=${PRESENCE_HEAD_INIT} variance=${VARIANCE_HEAD_INIT}"

exec "${PYTHON}" run_scripts/train_prosit_intensity_ptms_torch.py
