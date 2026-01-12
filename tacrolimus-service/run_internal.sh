#!/bin/bash

# Internal test용 - checkpoint 경로만 지정하면 파라미터를 자동으로 파싱
ckpt_names=(
    am_lstm_ep100_bs32_lr0.001_hd64_tw1_seed42_nl2_msl23.pth
    am_gru_ep100_bs32_lr0.0005_hd128_tw1_seed42_nl2_msl23.pth
    pm_gru_ep100_bs32_lr0.0005_hd128_tw1_seed42_nl3_msl10.pth
    pm_lstm_ep100_bs32_lr0.0005_hd128_tw1_seed42_nl2_msl10.pth
)

for ckpt_name in ${ckpt_names[@]}; do
    python internal_test.py \
        --checkpoint checkpoints/${ckpt_name} \
        --use_timeseries
done

ckpt_names=(
    am_lstm_static_ep100_bs32_lr0.0005_hd128_tw1_seed42_nl2_msl10.pth
    am_gru_static_ep100_bs32_lr0.0005_hd64_tw1_seed42_nl2_msl23.pth
    pm_gru_static_ep100_bs32_lr0.001_hd64_tw1_seed42_nl2_msl23.pth
    pm_lstm_static_ep100_bs32_lr0.0005_hd64_tw1_seed42_nl2_msl10.pth
)

for ckpt_name in ${ckpt_names[@]}; do
    python internal_test.py \
        --checkpoint checkpoints/${ckpt_name} \
        --use_timeseries \
        --use_static_features
done