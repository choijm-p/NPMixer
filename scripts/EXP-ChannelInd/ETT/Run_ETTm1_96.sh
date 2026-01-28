#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("ETTm1.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.003750 --d_model 256 --d_ff 128 --e_layers 5 --dropout 0.3467 --wavelet_j 1 --patch_len 12 --wavelet db2 --batch_size 64"
    "--learning_rate 0.005069 --d_model 512 --d_ff 32 --e_layers 2 --dropout 0.6868 --wavelet_j 4 --patch_len 16 --wavelet db3 --batch_size 64"
    "--learning_rate 0.002029 --d_model 512 --d_ff 256 --e_layers 3 --dropout 0.7644 --wavelet_j 1 --patch_len 8 --wavelet db5 --batch_size 64"
    "--learning_rate 0.005639 --d_model 128 --d_ff 2048 --e_layers 3 --dropout 0.4727 --wavelet_j 3 --patch_len 48 --wavelet db5 --batch_size 64"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="ETTm1_pl${p_len}_seed${seed}"
    
    python3 -u $base_dir/run_longExp.py \
        --model $model_name \
        --model_id "$model_id" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path ${datasets[0]} \
        --data ETTm1 \
        --features M \
        --freq t \
        --seq_len 96 \
        --pred_len $p_len \
        --enc_in 7 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 1 \
        --fix_seed $seed \
        --revin True
done