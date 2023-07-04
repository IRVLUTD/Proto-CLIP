#!/bin/bash

set -x
set -e
export CUDA_VISIBLE_DEVICES=1

CONFIG_DIR=../../../configs
DATASET_DIR=../../../dataset
MODEL_DIR=../../../pretrained_ckpt/fewsol-198-F

python proto_clip_node.py --config $CONFIG_DIR/fewsol_198.yml \
        --memory_bank_v_path $MODEL_DIR/memory_bank_v.pt \
        --memory_bank_t_path $MODEL_DIR/memory_bank_t.pt \
        --adapter_weights_path $MODEL_DIR/adapter_weights_path.pt \
        --asr_verbs_path ../pos/configs/verbs_dictionary.txt \
        --asr_nouns_path ../pos/configs/nouns_dictionary.txt \
        --asr_config_path ../asr/configs/asr_config.json \
        --splits_path $DATASET_DIR/fewsol_splits_198.json
