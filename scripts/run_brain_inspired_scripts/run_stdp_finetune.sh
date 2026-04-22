#!/bin/bash
# =============================================================================
# R-STDP Hybrid Fine-tuning Launch Script for NeuroVLA
#
# Loads a pretrained NeuroVLA checkpoint and fine-tunes the SNN action head
# with Reward-Modulated STDP (hybrid: alpha*backprop + beta*STDP).
#
# Default training steps:
#   single suite (libero_goal / spatial / object / 10)  -> 30000 steps
#   multi-suite  (libero_all)                           -> 50000 steps
# Override with --steps <N>.
#
# Usage:
#   bash run_stdp_finetune.sh                                          # libero_goal, 30k steps
#   bash run_stdp_finetune.sh --pretrained /path/to/ckpt               # override base ckpt
#   bash run_stdp_finetune.sh --steps 10000                            # override step count
#   bash run_stdp_finetune.sh --dataset libero_all                     # all 4 suites, 50k
#   bash run_stdp_finetune.sh --pretrained CKPT --dataset libero_spatial
#
# Supported --dataset values:
#   libero_goal      (default)  - 10 goal-directed tasks
#   libero_spatial              - 10 spatial reasoning tasks
#   libero_object               - 10 object manipulation tasks
#   libero_10                   - 10 long-horizon tasks (aka libero_long)
#   libero_all                  - combined mixture of all 4 suites
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

[ -f .env ] && { set -a; source .env; set +a; }

# ---------- defaults ----------
CONFIG_YAML="${CONFIG_YAML:-configs/finetune_config.yaml}"
MODE="${MODE:-neuro_vla_stdp}"
DATASET="libero_goal"
PRETRAINED=""
MAX_STEPS=""
RUN_ID=""
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PORT="${MAIN_PROCESS_PORT:-29500}"
DS_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/accelerate_zero2.yaml}"
CONDA_ENV="${CONDA_ENV:-alphabrain}"

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --pretrained|--ckpt) PRETRAINED="$2"; shift 2 ;;
        --steps|--max-steps) MAX_STEPS="$2"; shift 2 ;;
        --run-id)     RUN_ID="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- dataset-dependent default step count ----------
if [ -z "$MAX_STEPS" ]; then
    if [ "$DATASET" = "libero_all" ]; then
        MAX_STEPS=50000
    else
        MAX_STEPS=30000
    fi
fi

# ---------- build OmegaConf overrides ----------
OVERRIDES=(
    --datasets.vla_data.dataset_mix "$DATASET"
    --trainer.max_train_steps "$MAX_STEPS"
)
[ -n "$PRETRAINED" ] && OVERRIDES+=(--trainer.pretrained_checkpoint "$PRETRAINED")
[ -n "$RUN_ID" ]     && OVERRIDES+=(--run_id "$RUN_ID")

echo "=============================================="
echo "  NeuroVLA R-STDP Hybrid Fine-tuning"
echo "  Mode:         $MODE"
echo "  Dataset:      $DATASET"
echo "  Max steps:    $MAX_STEPS"
echo "  GPUs:         $NUM_GPUS"
[ -n "$PRETRAINED" ] && echo "  Pretrained:   $PRETRAINED"
[ -n "$RUN_ID" ]     && echo "  Run ID:       $RUN_ID"
echo "=============================================="

# ---------- resolve accelerate binary ----------
ENV_ACC=""
    for __p in /root/miniconda3/envs/${CONDA_ENV}/bin/accelerate /opt/conda/envs/${CONDA_ENV}/bin/accelerate; do
        if [ -x "$__p" ]; then ENV_ACC="$__p"; break; fi
    done
if [ -n "$ENV_ACC" ] && [ -x "$ENV_ACC" ]; then
    ACC="$ENV_ACC"
elif command -v accelerate >/dev/null 2>&1; then
    ACC=accelerate
else
    echo "ERROR: accelerate binary not found in conda env ${CONDA_ENV} or PATH" >&2
    exit 1
fi

# ---------- launch ----------
"$ACC" launch \
    --config_file "$DS_CONFIG" \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PORT" \
    AlphaBrain/training/train_stdp.py \
    --config_yaml "$CONFIG_YAML" \
    --mode "$MODE" \
    "${OVERRIDES[@]}"
