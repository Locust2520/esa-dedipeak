#!/usr/bin/bash

MODELS=$@
DATASETS=(weather.csv traffic.csv electricity.csv ETTm1.csv ETTm2.csv)
for model in ${MODELS} ; do
    for dataset in ${DATASETS[@]}; do
        python run.py --data_path $dataset --model InverseHeatDissipation --batch_size 16 --pred_len 720 --is_training 0 --period 24 --transformer_model $model --transformer_blurring 7 --transformer_deblurring 15
    done
done