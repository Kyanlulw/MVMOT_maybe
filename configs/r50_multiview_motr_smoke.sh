#!/usr/bin/env sh
set -eu

# Tiny one-epoch seven-camera smoke run. This checks dataset parsing, the separate
# FusionTrack detection/tracking decoders, ReID wiring, loss/backpropagation,
# checkpoint writing, and the multi-view training loop. It is intentionally not
# a quality-training configuration.
DATA_ROOT=${1:-"./tmp_smoke_mv"}
TRAIN_SPLIT=${2:-"./datasets/data_path/multiview_smoke.train.json"}
VAL_SPLIT=${3:-"./datasets/data_path/multiview_smoke.val.json"}
OUTPUT_DIR=${4:-"./output/smoke_fusiontrack_7cam"}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

set -- \
    --meta_arch fusiontrack_motr \
    --dataset_file e2e_mv_mot \
    --epochs 1 \
    --batch_size 1 \
    --num_workers 0 \
    --device "$DEVICE" \
    --num_cams 7 \
    --mot_path "$DATA_ROOT" \
    --data_txt_path_train "$TRAIN_SPLIT" \
    --data_txt_path_val "$VAL_SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --sample_mode fixed_interval \
    --sample_interval 1 \
    --sampler_lengths 1 \
    --num_queries 10 \
    --enc_layers 1 \
    --dec_layers 1 \
    --dim_feedforward 256 \
    --use_checkpoint \
    --with_box_refine \
    --use_reid_query \
    --reid_warmup_epochs 0 \
    --reid_num_layers 1 \
    --reid_num_heads 6 \
    --reid_tau1 2 \
    --reid_tau2 1 \
    --use_uncertainty_loss \
    --smoke_train

if [ -n "${PRETRAINED_CHECKPOINT:-}" ]; then
    set -- "$@" --pretrained "$PRETRAINED_CHECKPOINT"
fi

"$PYTHON" main.py "$@"
