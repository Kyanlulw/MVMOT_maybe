#!/usr/bin/env bash
# Usage:
#   bash configs/r50_multiview_motr_train_wildtrack.sh "0,1" /path/to/wildtrack_mvmot ./output/wildtrack_motr
# If GPU_IDS is omitted, this script will use all visible GPUs.
GPU_IDS=${1:-""}
MOT_PATH=${2:-"/kaggle/working/MVMOT_maybe/wildtrack_mvmot"}
OUTPUT_DIR=${3:-"./output/wildtrack_motr"}

PRETRAIN=coco_model_final.pth

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
PORT=$(( RANDOM % 1000 + 29500 ))
PRETRAIN_PATH="/kaggle/input/models/trnlqung/9epochmvmot/pytorch/default/1/checkpoint0009.pth"

echo "Using GPUs: $CUDA_VISIBLE_DEVICES (nproc_per_node=$GPUS)"

python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" --use_env main.py --meta_arch multiview_motr --dataset_file e2e_mv_mot --epochs 50 --use_uncertainty_loss --with_box_refine --lr_drop 25 --lr 1e-5 --lr_backbone 2e-5 --pretrained "$PRETRAIN" --output_dir "$OUTPUT_DIR" --batch_size 1 --sample_mode random_interval --sample_interval 2 --sampler_steps 12 24 36 --sampler_lengths 2 2 2 2 --update_query_pos --merger_dropout 0 --dropout 0 --random_drop 0.1 --fp_ratio 0.3 --query_interaction_layer QIM --extra_track_attn --mot_path "$MOT_PATH" --data_txt_path_train /kaggle/working/MVMOT_maybe/datasets/data_path/multiview_wildtrack.train --data_txt_path_val /kaggle/working/MVMOT_maybe/datasets/data_path/multiview_wildtrack.val --wandb --wandb_project multiview_motr --num_cams 2 --resume "$PRETRAIN_PATH"
