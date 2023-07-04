#!/bin/bash

set -x
set -e
export CUDA_VISIBLE_DEVICES=1

CONFIG_DIR=../../../configs
DATASET_DIR=../../../dataset
MODEL_DIR=../../../pretrained_ckpt/fewsol-198-F

python proto_clip_results_node.py --config $CONFIG_DIR/fewsol_198.yml \
        --memory_bank_v_path $MODEL_DIR/memory_bank_v.pt \
        --memory_bank_t_path $MODEL_DIR/memory_bank_t.pt \
        --adapter_weights_path $MODEL_DIR/adapter_weights_path.pt
