#!/bin/sh
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export GLUE_DIR=./dat
export TRAINED_CLASSIFIER=./output
python freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER/ \
    -max_seq_len 128 \
    -num_labels 15