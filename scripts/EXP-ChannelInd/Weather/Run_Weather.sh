#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("weather.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.001204 --d_model 32 --d_ff 64 --e_layers 1 --dropout 0.4388 --wavelet_j 4 --patch_len 4 --wavelet db4 --batch_size 32"
    "--learning_rate 0.001613 --d_model 32 --d_ff 512 --e_layers 3 --dropout 0.1733 --wavelet_j 2 --patch_len 4 --wavelet db4 --batch_size 32"
    "--learning_rate 0.000524 --d_model 32 --d_ff 64 --e_layers 3 --dropout 0.6386 --wavelet_j 3 --patch_len 4 --wavelet db4 --batch_size 32"
    "--learning_rate 0.002029 --d_model 128 --d_ff 128 --e_layers 1 --dropout 0.1078 --wavelet_j 5 --patch_len 4 --wavelet db4 --batch_size 32"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="Weather_pl${p_len}_seed${seed}"
    
    python3 -u $base_dir/run_longExp.py \
        --model $model_name \
        --model_id "$model_id" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path ${datasets[0]} \
        --data custom \
        --features M \
        --freq h \
        --seq_len 96 \
        --pred_len $p_len \
        --enc_in 21 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 1 \
        --fix_seed $seed \
        --revin True
done