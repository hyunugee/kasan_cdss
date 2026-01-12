#!/bin/bash

# 시계열 모델 하이퍼파라미터 실험 스크립트
# hidden_dim, learning_rate, batch_size, num_layers, rnn_type (lstm/gru) 조합 실험

# GPU 설정
# export CUDA_VISIBLE_DEVICES=0
# BASE_CMD="python tacrolimus_dose_prediction.py --use_timeseries --model_type am --use_static_features"

# export CUDA_VISIBLE_DEVICES=1
# BASE_CMD="python tacrolimus_dose_prediction.py --use_timeseries --model_type am"

# export CUDA_VISIBLE_DEVICES=2
# BASE_CMD="python tacrolimus_dose_prediction.py --use_timeseries --model_type pm --use_static_features"

export CUDA_VISIBLE_DEVICES=3
BASE_CMD="python tacrolimus_dose_prediction.py --use_timeseries --model_type pm"


# 실험할 하이퍼파라미터 조합
HIDDEN_DIMS=(32 64 128)
LEARNING_RATES=(0.0005 0.001 0.005)
BATCH_SIZES=(32)
NUM_LAYERS=(2 3)
RNN_TYPES=("lstm" "gru")
MAX_SEQ_LENS=(10 23)

# 실험 카운터
counter=0
total_experiments=$((${#HIDDEN_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#NUM_LAYERS[@]} * ${#RNN_TYPES[@]} * ${#MAX_SEQ_LENS[@]}))

echo "=========================================="
echo "시계열 모델 하이퍼파라미터 실험 시작"
echo "총 실험 수: $total_experiments"
echo "=========================================="
echo ""

# 각 조합에 대해 실험 실행
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for num_layers in "${NUM_LAYERS[@]}"; do
                for rnn_type in "${RNN_TYPES[@]}"; do
                    for max_seq_len in "${MAX_SEQ_LENS[@]}"; do
                        counter=$((counter + 1))
                        
                        echo "=========================================="
                        echo "[$counter/$total_experiments] 실험 시작"
                        echo "hidden_dim=$hidden_dim, lr=$lr, batch_size=$batch_size, num_layers=$num_layers, rnn_type=$rnn_type, max_seq_len=$max_seq_len"
                        echo "=========================================="
                        
                        # 명령어 실행
                        $BASE_CMD \
                            --hidden_dim $hidden_dim \
                            --learning_rate $lr \
                            --batch_size $batch_size \
                            --num_layers $num_layers \
                            --rnn_type $rnn_type \
                            --max_seq_len $max_seq_len
                        
                        echo ""
                        echo "실험 완료: hidden_dim=$hidden_dim, lr=$lr, batch_size=$batch_size, num_layers=$num_layers, rnn_type=$rnn_type, max_seq_len=$max_seq_len"
                        echo ""
                        
                        # GPU 메모리 정리 (필요시)
                        # sleep 2
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "모든 실험 완료!"
echo "=========================================="

