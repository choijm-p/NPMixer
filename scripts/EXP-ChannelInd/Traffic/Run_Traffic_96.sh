#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("traffic.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.000298 --d_model 512 --d_ff 2048 --e_layers 3 --dropout 0.4411 --wavelet_j 3 --patch_len 48 --wavelet db1 --batch_size 8"
    "--learning_rate 0.000448 --d_model 512 --d_ff 1024 --e_layers 2 --dropout 0.4589 --wavelet_j 4 --patch_len 48 --wavelet db1 --batch_size 8"
    "--learning_rate 0.000822 --d_model 256 --d_ff 1024 --e_layers 2 --dropout 0.4453 --wavelet_j 4 --patch_len 8 --wavelet db1 --batch_size 8"
    "--learning_rate 0.000411 --d_model 256 --d_ff 2048 --e_layers 4 --dropout 0.4071 --wavelet_j 2 --patch_len 32 --wavelet db1 --batch_size 8"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="Traffic_pl${p_len}_seed${seed}"
    
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
        --enc_in 862 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 5 \
        --fix_seed $seed \
        --revin True
done