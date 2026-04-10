#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Multi-View MOTR Training Script for Wildtrack
# -----------------------------------------------------------------------
# Usage:
#   bash configs/r50_multiview_motr_train_wildtrack.sh [GPU_IDS] [MOT_PATH] [OUTPUT_DIR]
#
# Example:
#   bash configs/r50_multiview_motr_train_wildtrack.sh 0,1 /data/wildtrack_motr ./output/wildtrack
# -----------------------------------------------------------------------

GPU_IDS=${1:-"0"}
MOT_PATH=${2:-"/data/wildtrack_motr"}
OUTPUT_DIR=${3:-"./output/wildtrack_motr"}

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
    --sample_interval 2 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer QIM \
    --extra_track_attn \
    --mot_path ${MOT_PATH} \
    --data_txt_path_train ./datasets/data_path/multiview_wildtrack.train \
    --data_txt_path_val ./datasets/data_path/multiview_wildtrack.val \
    --num_queries 300 \
    --num_views 7 \
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
