#!bin/sh
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=./output

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/ \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/ \
    -mode CLASS \
    -max_seq_len 128 \
    -port 5575 \
    -port_out 5576 \