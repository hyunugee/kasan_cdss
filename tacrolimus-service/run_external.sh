#!/bin/bash

python external_test.py \
    --checkpoint checkpoints/pm_gru_ep100_bs32_lr0.0005_hd128_tw1_seed42_nl3_msl10.pth \
    --model_type pm \
    --use_timeseries \
    --external_data data/AM_model_prediction_result_with_DR2_fixed_filled.csv

python external_test.py \
    --checkpoint checkpoints/am_lstm_ep100_bs32_lr0.001_hd64_tw1_seed42_nl2_msl23.pth \
    --model_type am \
    --use_timeseries \
    --external_data data/AM_model_prediction_result_with_DR2_fixed_filled.csv
