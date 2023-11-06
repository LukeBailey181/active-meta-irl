#!/bin/bash

# For example:
#  bash scripts/evaluate_models.sh configs/bc/bc_exp_cfg.yaml logs/bc logs/eval/bc/bc_eval.csv
#  bash scripts/evaluate_models.sh configs/bc/bc_al_cfg.yaml logs/bc-al logs/eval/bc/bc_al_eval.csv
#  bash scripts/evaluate_models.sh configs/bc/bc_cnn_cfg.yaml logs/bc_cnn logs/eval/bc/bc_cnn_eval.csv
#  bash scripts/evaluate_models.sh configs/bc/bc_cnn_al_cfg.yaml logs/bc_cnn-al logs/eval/bc/bc_cnn_al_eval.csv

for entry in "$2"/*
do
  echo "$entry"
  python evaluation.py -y $1 -s $entry -n $3
done

python visualize_results.py -y $1 -n $3
