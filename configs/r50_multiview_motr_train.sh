#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Multi-View MOTR Training Script
# -----------------------------------------------------------------------
# Usage:
#   bash configs/r50_multiview_motr_train.sh [GPU_IDS] [DATA_ROOT] [OUTPUT_DIR]
#
# Example:
#   bash configs/r50_multiview_motr_train.sh 0,1 /data/multiview_mot ./output/multiview
# -----------------------------------------------------------------------

GPU_IDS=${1:-"0"}
DATA_ROOT=${2:-"/data/multiview_mot"}
OUTPUT_DIR=${3:-"./output/multiview_motr"}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
PORT=$(( RANDOM % 1000 + 29500 ))

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    --use_env \
    main.py \
    --meta_arch multiview_motr \
    --dataset_file e2e_mv_mot \
    --epochs 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained coco_model_final.pth \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 3 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --mot_path ${DATA_ROOT} \
    --data_txt_path_train ./datasets/data_path/multiview.train \
    --data_txt_path_val ./datasets/data_path/multiview.val \
    --num_queries 300 \
    --num_views 2 \
    --cross_view_fusion_layers 2 \
    --cross_view_nhead 8 \
    --cross_view_dropout 0.1 \
    --enable_cross_view_query_exchange \
    --cross_view_match_thresh 0.3 \
    --cross_view_loss_coef 1.0 \
    --memory_bank_type MemoryBank \
    --memory_bank_len 4 \
    --memory_bank_score_thresh 0.0 \
    --memory_bank_with_self_attn
