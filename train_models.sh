#!/usr/bin/bash

MODELS=$@
DATASETS=(weather.csv traffic.csv electricity.csv ETTm1.csv ETTm2.csv)
for model in ${MODELS} ; do
    for dataset in ${DATASETS[@]}; do
        python run.py --data_path $dataset --model $model --batch_size 16 --pred_len 720
    done
done