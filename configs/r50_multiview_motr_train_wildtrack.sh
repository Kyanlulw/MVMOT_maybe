#!/usr/bin/env bash
GPU_IDS=${1:-"0"}
MOT_PATH=${2:-"/kaggle/working/MVMOT_maybe/wildtrack_mvmot"}
OUTPUT_DIR=${3:-"./output/wildtrack_motr"}

PRETRAIN=coco_model_final.pth
GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
PORT=$(( RANDOM % 1000 + 29500 ))

python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" --use_env main.py --meta_arch multiview_motr --dataset_file e2e_mv_mot --epochs 50 --with_box_refine --lr_drop 100 --lr 2e-4 --lr_backbone 2e-5 --pretrained "$PRETRAIN" --output_dir "$OUTPUT_DIR" --batch_size 1 --sample_mode random_interval --sample_interval 2 --sampler_steps 12 24 36 --sampler_lengths 2 3 4 5 --update_query_pos --merger_dropout 0 --dropout 0 --random_drop 0.1 --fp_ratio 0.3 --query_interaction_layer QIM --extra_track_attn --mot_path "$MOT_PATH" --data_txt_path_train /kaggle/working/MVMOT_maybe/datasets/data_path/multiview_wildtrack.train --data_txt_path_val /kaggle/working/MVMOT_maybe/datasets/data_path/multiview_wildtrack.val --num_cams 7
