#!/bin/bash -e

TRAINING_DATA=data/acl2016-training.txt

OUT_DIR=output/acl2016/baseline
MODEL=$OUT_DIR/model.lisp

mkdir -p $OUT_DIR

./src/scripts/train_baseline.py $TRAINING_DATA $MODEL
