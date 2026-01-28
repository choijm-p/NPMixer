#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("ETTh2.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.003472 --d_model 512 --d_ff 1024 --e_layers 5 --dropout 0.7294 --wavelet_j 1 --patch_len 32 --wavelet db1 --batch_size 64"
    "--learning_rate 0.004623 --d_model 64 --d_ff 128 --e_layers 3 --dropout 0.5538 --wavelet_j 3 --patch_len 48 --wavelet db1 --batch_size 64"
    "--learning_rate 0.001669 --d_model 64 --d_ff 128 --e_layers 3 --dropout 0.1705 --wavelet_j 5 --patch_len 4 --wavelet bior3.1 --batch_size 256"
    "--learning_rate 0.000366 --d_model 256 --d_ff 128 --e_layers 1 --dropout 0.2422 --wavelet_j 4 --patch_len 24 --wavelet bior3.1 --batch_size 256"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="ETTh2_pl${p_len}_seed${seed}"
    
    python3 -u $base_dir/run_longExp.py \
        --model $model_name \
        --model_id "$model_id" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path ${datasets[0]} \
        --data ETTh2 \
        --features M \
        --freq h \
        --seq_len 96 \
        --pred_len $p_len \
        --enc_in 7 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 10 \
        --fix_seed $seed \
        --revin True
done