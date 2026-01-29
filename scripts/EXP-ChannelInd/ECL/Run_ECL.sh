#!/bin/bash

root_path=../../../dataset/
base_dir=../../..
datasets=("electricity.csv")
model_name="NPMixer"
pred_lens=(96 192 336 720)
seed=3000

configs=(
    "--learning_rate 0.000790 --d_model 512 --d_ff 2048 --e_layers 2 --dropout 0.2426 --wavelet_j 5 --patch_len 16 --wavelet sym3 --batch_size 32"
    "--learning_rate 0.001385 --d_model 128 --d_ff 1024 --e_layers 3 --dropout 0.3072 --wavelet_j 3 --patch_len 16 --wavelet sym4 --batch_size 16"
    "--learning_rate 0.001173 --d_model 128 --d_ff 512 --e_layers 4 --dropout 0.1117 --wavelet_j 4 --patch_len 4 --wavelet db1 --batch_size 32"
    "--learning_rate 0.003123 --d_model 128 --d_ff 128 --e_layers 1 --dropout 0.3069 --wavelet_j 5 --patch_len 16 --wavelet db1 --batch_size 32"
)

for i in "${!pred_lens[@]}"; do
    p_len=${pred_lens[$i]}
    cfg="${configs[$i]}"
    model_id="ECL_pl${p_len}_seed${seed}"
    
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
        --enc_in 321 \
        $cfg \
        --use_gpu True \
        --gpu 0 \
        --patience 5 \
        --fix_seed $seed \
        --revin True
done