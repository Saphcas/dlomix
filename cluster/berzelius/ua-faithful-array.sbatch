#!/bin/bash
#SBATCH -A Berzelius-2026-159
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=ua-faithful
#SBATCH --output=slurm-%x-%A_%a.out

set -Eeuo pipefail

PROJECT_ROOT=/proj/bedrock/users/x_alfni/dlomix_Saphcas
PYTHON=/proj/bedrock/users/x_alfni/conda-envs/dlomix-torch/bin/python
DATA_DIR=/proj/bedrock/datasets/Prosit_PTMs/PTMs_Train
CHECKPOINT_ROOT=/proj/bedrock/users/x_alfni/checkpoints/ua-faithful

MODES=(off faithful_variance_only faithful_strict)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= ${#MODES[@]} )); then
    echo "Invalid array task id: ${TASK_ID}" >&2
    exit 2
fi

MODE=${MODES[$TASK_ID]}
RUN_NAME="ua-faithful-${MODE}-bs${BATCH_SIZE:-4096}-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"

mkdir -p "${CHECKPOINT_DIR}"
cd "${PROJECT_ROOT}"

module load buildenv-gcccuda/12.1.1-gcc12.3.0
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export DATA_LOCATION="${DATA_DIR}"
export DLOMIX_BACKEND=torch
export NUM_WORKERS="${NUM_WORKERS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-4096}"
export N_EPOCHS="${N_EPOCHS:-30}"
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
export WARMUP_COSINE_TOTAL_STEPS=0

# Isolate faithful routing from beta-NLL, initialization, and variance-prior modes.
export MEAN_HEAD_INIT=default
export PRESENCE_HEAD_INIT=default
export VARIANCE_HEAD_INIT=default
export VARIANCE_PRIOR_WEIGHT=0
unset VARIANCE_PRIOR_STATS
export BETA_NLL_BETA=0
export FAITHFUL_MODE="${MODE}"
unset FAITHFUL_HETEROSCEDASTIC

export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=ua-faithful-default-init
export CHECKPOINT_DIR
export SAVE="${CHECKPOINT_DIR}/best_model.pth"

echo "Starting ${RUN_NAME}"
echo "  faithful_mode=${FAITHFUL_MODE} beta=${BETA_NLL_BETA} batch_size=${BATCH_SIZE}"
echo "  normalization=${LOSS_NORMALIZATION} prior_weight=${VARIANCE_PRIOR_WEIGHT}"
echo "  checkpoint=${CHECKPOINT_DIR}"

exec "${PYTHON}" run_scripts/train_prosit_intensity_ptms_torch.py
