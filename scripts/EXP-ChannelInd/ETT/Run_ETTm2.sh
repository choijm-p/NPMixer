#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("ETTm2.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.005248 --d_model 64 --d_ff 128 --e_layers 1 --dropout 0.5479 --wavelet_j 2 --patch_len 12 --wavelet bior3.1 --batch_size 64"
    "--learning_rate 0.005069 --d_model 512 --d_ff 32 --e_layers 2 --dropout 0.6868 --wavelet_j 4 --patch_len 16 --wavelet db2 --batch_size 64"
    "--learning_rate 0.001221 --d_model 128 --d_ff 256 --e_layers 2 --dropout 0.8845 --wavelet_j 5 --patch_len 24 --wavelet db2 --batch_size 64"
    "--learning_rate 0.003051 --d_model 64 --d_ff 2048 --e_layers 4 --dropout 0.8046 --wavelet_j 2 --patch_len 32 --wavelet db2 --batch_size 64"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="ETTm2_pl${p_len}_seed${seed}"
    
    python3 -u $base_dir/run_longExp.py \
        --model $model_name \
        --model_id "$model_id" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path ${datasets[0]} \
        --data ETTm2 \
        --features M \
        --freq t \
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