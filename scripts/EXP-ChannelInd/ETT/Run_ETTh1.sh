#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("ETTh1.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.001514 --d_model 64 --d_ff 1024 --e_layers 1 --dropout 0.2762 --wavelet_j 1 --patch_len 24 --wavelet db2 --batch_size 256"
    "--learning_rate 0.008076 --d_model 32 --d_ff 32 --e_layers 4 --dropout 0.7739 --wavelet_j 4 --patch_len 16 --wavelet db1 --batch_size 64"
    "--learning_rate 0.004639 --d_model 32 --d_ff 32 --e_layers 4 --dropout 0.8131 --wavelet_j 4 --patch_len 4 --wavelet db1 --batch_size 64"
    "--learning_rate 0.001700 --d_model 128 --d_ff 512 --e_layers 4 --dropout 0.5793 --wavelet_j 1 --patch_len 24 --wavelet sym4 --batch_size 256"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="ETTh1_pl${p_len}_seed${seed}"
    
    python3 -u $base_dir/run_longExp.py \
        --model $model_name \
        --model_id "$model_id" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path ${datasets[0]} \
        --data ETTh1 \
        --features M \
        --freq h \
        --seq_len 96 \
        --pred_len $p_len \
        --enc_in 7 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 5 \
        --fix_seed $seed \
        --revin True
done