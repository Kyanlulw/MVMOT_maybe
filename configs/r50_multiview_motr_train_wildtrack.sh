#!/usr/bin/env sh
set -eu
# Usage:
#   bash configs/r50_multiview_motr_train_wildtrack.sh "0,1" /path/to/wildtrack_mvmot ./output/wildtrack_motr
# If GPU_IDS is omitted, this script will use all visible GPUs.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
GPU_IDS=${1:-""}
MOT_PATH=${2:-"$REPO_ROOT/wildtrack_mvmot"}
OUTPUT_DIR=${3:-"$REPO_ROOT/output/wildtrack_motr"}
TRAIN_SPLIT=${DATA_TXT_PATH_TRAIN:-"$REPO_ROOT/datasets/data_path/multiview_wildtrack.train"}
VAL_SPLIT=${DATA_TXT_PATH_VAL:-"$REPO_ROOT/datasets/data_path/multiview_wildtrack.val"}
PYTHON=${PYTHON:-python}

# Optional checkpoint. Set PRETRAIN to an explicit file, or leave it empty to
# start from the model initialization. The Kaggle checkpoint is used only when
# it is actually mounted in the current notebook.
PRETRAIN=${PRETRAIN:-""}
PRETRAIN_PATH=${PRETRAIN_PATH:-""}
if [ -z "$PRETRAIN" ] && [ -f "$PRETRAIN_PATH" ]; then
	PRETRAIN="$PRETRAIN_PATH"
fi
if [ -n "$PRETRAIN" ] && [ ! -f "$PRETRAIN" ]; then
	echo "Pretrained checkpoint not found: $PRETRAIN" >&2
	echo "Set PRETRAIN=\"\" to train from initialization or provide a valid path." >&2
	exit 1
fi

if [ -z "$GPU_IDS" ]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
	else
		GPU_IDS="0"
	fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
PORT=${MASTER_PORT:-29500}
echo "Using GPUs: $CUDA_VISIBLE_DEVICES (nproc_per_node=$GPUS)"
echo "Pretrained checkpoint: ${PRETRAIN:-<none>}"

set -- \
    --meta_arch fusiontrack_motr \
    --dataset_file e2e_mv_mot \
    --epochs 80 \
    --use_uncertainty_loss \
    --use_reid_query \
    --with_box_refine \
    --lr 1e-5 \
    --lr_backbone 1e-6 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --sample_mode fixed_interval \
    --sample_interval 1 \
    --sampler_steps 20 \
    --sampler_lengths 10 10 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --use_checkpoint \
    --mot_path "$MOT_PATH" \
    --data_txt_path_train "$TRAIN_SPLIT" \
    --data_txt_path_val "$VAL_SPLIT" \
    --wandb \
    --wandb_project fusiontrack_wildtrack \
    --num_cams 7 \
    --reid_num_layers 12 \
    --reid_num_heads 6 \
    --reid_tau1 30 \
    --reid_tau2 10
if [ -n "$PRETRAIN" ]; then
	set -- "$@" --pretrained "$PRETRAIN"
fi

if [ "$GPUS" -eq 1 ]; then
	"$PYTHON" main.py "$@"
else
	"$PYTHON" -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" --use_env main.py "$@"
fi
