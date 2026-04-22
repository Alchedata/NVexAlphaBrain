#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash benchmarks/LIBERO/data_preparation.sh
# or
#   bash benchmarks/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash benchmarks/LIBERO/data_preparation.sh"
  echo "  or: bash benchmarks/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

python -m pip install -U "huggingface-hub==0.35.3"

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

download_dataset() {
  local repo="$1"
  local target_dir="$2"
  local attempts="${HF_DOWNLOAD_RETRIES:-5}"
  local max_workers="${HF_DOWNLOAD_MAX_WORKERS:-1}"
  local backoff=15
  local attempt

  mkdir -p "$target_dir"

  for ((attempt=1; attempt<=attempts; attempt++)); do
    if hf download "$repo" \
      --repo-type dataset \
      --local-dir "$target_dir" \
      --max-workers "$max_workers"; then
      return 0
    fi

    if (( attempt == attempts )); then
      echo "ERROR: failed to download $repo after $attempts attempts." >&2
      return 1
    fi

    echo "Download failed for $repo (attempt $attempt/$attempts). Retrying in ${backoff}s..." >&2
    sleep "$backoff"
    backoff=$((backoff * 2))
  done
}

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  download_dataset "$repo" "$DEST/libero/${repo##*/}"
done

download_dataset "AlphaBrain/LLaVA-OneVision-COCO" "$DEST/LLaVA-OneVision-COCO"
if [[ -f "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" ]] && [[ ! -d "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco" ]]; then
  unzip -o -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"
fi

mkdir -p "$CUR/data/datasets"
ln -sfn "$DEST/libero" "$CUR/data/datasets/LEROBOT_LIBERO_DATA"
ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/data/datasets/LLaVA-OneVision-COCO"

## move modality
cp "$CUR/benchmarks/LIBERO/train/modality.json" "$CUR/data/datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/benchmarks/LIBERO/train/modality.json" "$CUR/data/datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/benchmarks/LIBERO/train/modality.json" "$CUR/data/datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/benchmarks/LIBERO/train/modality.json" "$CUR/data/datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
