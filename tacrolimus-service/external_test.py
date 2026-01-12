#!/usr/bin/env python3
"""
타크로리무스 투여량 예측 - External Test
저장된 모델을 로드하여 external 데이터셋에 대해 테스트 수행

Author: AI Assistant
Date: 2024
"""

import argparse
import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 기존 모듈 import
from tacrolimus_dose_prediction import TacrolimusDataset, prepare_sequence_data
from timeseries_models import create_lstm_model, create_lstm_trainer
from tabular_models import TacrolimusRegressorModel, create_regressor_trainer
from utils import set_random_seed, visualize_predictions, save_experiment_results

warnings.filterwarnings('ignore')


def parse_checkpoint_params(model_name: str, args):
    """
    체크포인트 파일명에서 파라미터 추출
    
    예: pm_lstm_ep100_bs32_lr0.005_hd128_tw1_seed42_nl2_msl10
    """
    # hidden_dim 파싱 (hd128 -> 128)
    hd_match = re.search(r'_hd(\d+)', model_name)
    if hd_match:
        args.hidden_dim = int(hd_match.group(1))
    
    # num_layers 파싱 (nl2 -> 2)
    nl_match = re.search(r'_nl(\d+)', model_name)
    if nl_match:
        args.num_layers = int(nl_match.group(1))
    
    # max_seq_len 파싱 (msl10 -> 10)
    msl_match = re.search(r'_msl(\d+)', model_name)
    if msl_match:
        args.max_seq_len = int(msl_match.group(1))
    
    # learning_rate 파싱 (lr0.005 -> 0.005)
    lr_match = re.search(r'_lr([\d.]+)', model_name)
    if lr_match:
        args.learning_rate = float(lr_match.group(1))
    
    # time_window 파싱 (tw3 -> 3)
    tw_match = re.search(r'_tw(\d+)', model_name)
    if tw_match:
        args.time_window = int(tw_match.group(1))
    
    # rnn_type 파싱 (lstm 또는 gru)
    if '_lstm_' in model_name or model_name.endswith('_lstm'):
        args.rnn_type = 'lstm'
    elif '_gru_' in model_name or model_name.endswith('_gru'):
        args.rnn_type = 'gru'
    
    return args


def load_external_data(data_path: str, model_type: str = 'pm') -> pd.DataFrame:
    """
    External 데이터 로드 및 전처리
    
    Args:
        data_path: external 데이터 경로
        model_type: 'pm' 또는 'am'
    
    Returns:
        전처리된 DataFrame
    """
    # CSV 로드
    data = pd.read_csv(data_path)
    
    # 컬럼명 매핑
    column_mapping = {
        'before_pm': 'prev_dose_pm',
        'today_am': 'today_dose_am',
        'today_tdm': 'today_tdm',
        'y1_pm': 'today_dose_pm',
        'y2_am': 'next_dose_am',
        'y1_pm_dr2': 'today_dose_pm_dr2',
        'y2_am_dr2': 'next_dose_am_dr2'
    }
    
    data = data.rename(columns=column_mapping)
    
    # 필요한 컬럼 선택 (DR2 컬럼 포함)
    if model_type == 'pm':
        required_cols = ['patient_id', 'prev_dose_pm', 'today_dose_am', 'today_tdm', 'today_dose_pm']
        # DR2 컬럼이 있으면 추가
        if 'today_dose_pm_dr2' in data.columns:
            required_cols.append('today_dose_pm_dr2')
    else:  # am
        required_cols = ['patient_id', 'prev_dose_pm', 'today_dose_am', 'today_tdm', 'today_dose_pm', 'next_dose_am']
        # DR2 컬럼이 있으면 추가
        if 'next_dose_am_dr2' in data.columns:
            required_cols.append('next_dose_am_dr2')
    
    # 존재하는 컬럼만 선택
    available_cols = [col for col in required_cols if col in data.columns]
    data = data[available_cols]
    
    # 결측값 제거 (DR2 컬럼 제외)
    essential_cols = ['patient_id', 'prev_dose_pm', 'today_dose_am', 'today_tdm']
    if model_type == 'pm':
        essential_cols.append('today_dose_pm')
    else:
        essential_cols.extend(['today_dose_pm', 'next_dose_am'])
    
    data = data.dropna(subset=essential_cols)
    
    # day_number 추가 (시각화용)
    data['day_number'] = data.groupby('patient_id').cumcount() + 1
    
    print(f"\nExternal 데이터 로딩 완료:")
    print(f"  - 환자 수: {data['patient_id'].nunique()}")
    print(f"  - 총 샘플 수: {len(data)}")
    print(f"  - 평균 시퀀스 길이: {data.groupby('patient_id').size().mean():.1f}")
    if model_type == 'pm' and 'today_dose_pm_dr2' in data.columns:
        print(f"  - DR2 데이터 포함됨")
    elif model_type == 'am' and 'next_dose_am_dr2' in data.columns:
        print(f"  - DR2 데이터 포함됨")
    
    return data


def evaluate_model_on_external(model_type: str, external_data: pd.DataFrame, 
                                checkpoint_path: str, args, device: torch.device,
                                visualization_dir: str = "visualizations_external"):
    """
    External 데이터에 대해 저장된 모델 평가
    
    Args:
        model_type: 'pm' 또는 'am'
        external_data: external 테스트 데이터
        checkpoint_path: 체크포인트 파일 경로
        args: 명령행 인자
        device: 사용할 디바이스
        visualization_dir: 시각화 저장 디렉토리
    """
    model_name_upper = model_type.upper()
    print(f"\n=== {model_name_upper} Model External Test ===")
    print(f"체크포인트: {checkpoint_path}")
    
    # 체크포인트 확인
    if not os.path.exists(checkpoint_path):
        print(f"Error: 체크포인트 파일이 없습니다: {checkpoint_path}")
        return
    
    # 모델 이름 추출
    model_name = os.path.basename(checkpoint_path).replace('.pth', '').replace('.pkl', '')
    
    # 체크포인트 파일명에서 파라미터 파싱
    args = parse_checkpoint_params(model_name, args)
    
    # Scaler 로드 (Tabular 모델인 경우)
    scaler = None
    if not args.use_timeseries:
        scaler_path = os.path.join(os.path.dirname(checkpoint_path), f'{model_name}_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Warning: Scaler 파일이 없습니다: {scaler_path}")
            print("정규화 없이 진행합니다. 성능이 낮을 수 있습니다!")
    
    # DR2 데이터 미리 추출 (시각화용)
    dr2_col = f"{'today_dose_pm' if model_type == 'pm' else 'next_dose_am'}_dr2"
    dr2_data_original = external_data[dr2_col].values if dr2_col in external_data.columns else None
    
    # 시계열 데이터 준비 (필요시)
    if args.use_timeseries:
        print(f"Preparing sequence data for {model_name_upper} model...")
        patients = external_data['patient_id'].unique().tolist()
        test_data = prepare_sequence_data(
            external_data, 
            patients, 
            model_type=model_type, 
            max_seq_len=args.max_seq_len,
            include_day_number=True
        )
    else:
        test_data = external_data
    
    # 데이터셋 생성 (정적 특성 없음)
    test_dataset = TacrolimusDataset(
        test_data, 
        static_features=None,
        scaler=scaler,
        is_training=False,
        model_type=model_type,
        time_window=args.time_window,
        use_timeseries=args.use_timeseries,
        max_seq_len=args.max_seq_len
    )
    
    # 데이터로더 생성
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 및 트레이너 생성
    if args.use_timeseries:
        # 시계열 모델
        if model_type == 'am':
            input_dim = 4  # prev_dose_pm, today_dose_am, today_tdm, today_dose_pm
        else:  # pm
            input_dim = 3  # prev_dose_pm, today_dose_am, today_tdm
        
        static_dim = 0  # external 데이터는 정적 특성 없음
        model = create_lstm_model(
            input_dim, 
            args.hidden_dim, 
            num_layers=args.num_layers,
            use_static=False,  # external 데이터는 정적 특성 없음
            static_dim=static_dim, 
            rnn_type=args.rnn_type
        )
        trainer = create_lstm_trainer(
            model, 
            device, 
            args.learning_rate, 
            model_name, 
            model_type,
            checkpoint_dir=os.path.dirname(checkpoint_path)
        )
        
        print(f"{model_name_upper} {args.rnn_type.upper()} 모델 입력 차원: {input_dim} (각 시점), 레이어 수: {args.num_layers}")
    else:
        # Tabular 모델
        model = TacrolimusRegressorModel(
            model_type=args.regressor_type,
            random_state=args.seed
        )
        trainer = create_regressor_trainer(
            model, 
            model_name, 
            model_type,
            checkpoint_dir=os.path.dirname(checkpoint_path)
        )
        
        # 입력 차원 정보 출력
        sample_features, _ = next(iter(test_loader))
        input_dim = sample_features.shape[1]
        print(f"{model_name_upper} Tabular 모델 입력 차원: {input_dim} (time_window={args.time_window})")
    
    # 체크포인트 로드
    loaded = trainer.load_checkpoint()
    if not loaded:
        print(f"Error: 체크포인트를 로드할 수 없습니다.")
        return
    
    print("체크포인트 로드 완료!")
    
    # 테스트 평가 (예측값과 실제값 포함)
    test_loss, test_metrics, (test_predictions, test_actuals) = trainer.validate(
        test_loader, 
        return_predictions=True
    )
    
    print(f"\n{model_name_upper} Model External Test Results:")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    
    # 시각화 디렉토리 생성
    os.makedirs(visualization_dir, exist_ok=True)
    
    # DR2 데이터 확인 (시계열 모델의 경우 original data에서, tabular는 test_data에서)
    if args.use_timeseries:
        # 시계열 모델: original data에서 추출한 것 사용
        dr2_data = dr2_data_original[:len(test_predictions)] if dr2_data_original is not None else None
    else:
        # Tabular 모델: test_data에서 추출
        dr2_col = f"{'today_dose_pm' if model_type == 'pm' else 'next_dose_am'}_dr2"
        dr2_data = test_data[dr2_col].values[:len(test_predictions)] if dr2_col in test_data.columns else None
    
    # 시각화
    visualize_predictions(
        test_data, 
        test_predictions, 
        test_actuals, 
        model_type, 
        f"{model_name}_external",
        max_day=8,
        test_dataset=test_dataset,
        show_std=args.show_std,
        visualization_dir=visualization_dir,
        use_static_features=False,  # external 데이터는 정적 특성 없음
        dr2_data=dr2_data,  # DR2 데이터 전달
        is_external_test=True  # external test 플래그
    )
    
    # 결과를 CSV로 저장
    results_df = pd.DataFrame({
        'patient_id': test_data['patient_id'].values[:len(test_predictions)],
        'actual': test_actuals,
        'predicted': test_predictions,
        'error': test_actuals - test_predictions,
        'absolute_error': np.abs(test_actuals - test_predictions)
    })
    
    results_path = os.path.join(visualization_dir, f"{model_name}_external_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n결과 저장: {results_path}")
    
    # 실험 결과를 JSON으로 저장
    external_results_dir = os.path.join(os.path.dirname(visualization_dir), "experiment_results_external")
    save_experiment_results(
        model_type, 
        f"{model_name}_external", 
        args, 
        test_metrics, 
        test_loss, 
        results_dir=external_results_dir
    )
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='타크로리무스 투여량 예측 - External Test')
    
    # 데이터 경로
    parser.add_argument('--external_data', type=str,
                       default='data/PM_model_prediction_newset_250420_filled.csv',
                       help='External 데이터 경로')
    
    # 모델 설정
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='체크포인트 파일 경로')
    parser.add_argument('--model_type', type=str, choices=['pm', 'am', 'both'],
                       default='both', help='테스트할 모델 타입')
    
    # 모델 타입별 설정
    parser.add_argument('--use_timeseries', action='store_true',
                       help='시계열 모델 사용 여부')
    parser.add_argument('--regressor_type', type=str,
                       choices=['random_forest', 'gradient_boosting', 'svr', 'ridge', 
                               'lasso', 'elastic_net', 'knn', 'decision_tree', 'catboost', 'xgboost'],
                       default='catboost', help='Regressor 모델 타입')
    parser.add_argument('--time_window', type=int, default=1,
                       help='시계열 윈도우 크기 (tabular 모델용)')
    
    # 시계열 모델 설정
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='은닉층 차원')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='RNN 레이어 수')
    parser.add_argument('--rnn_type', type=str, choices=['lstm', 'gru'], default='lstm',
                       help='RNN 타입')
    parser.add_argument('--max_seq_len', type=int, default=23,
                       help='최대 시퀀스 길이')
    
    # 기타
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='학습률 (모델 이름 생성용)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='에포크 수 (모델 이름 생성용)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (cpu/cuda/auto)')
    parser.add_argument('--show_std', action='store_true',
                       help='시각화에서 표준편차 표시')
    
    args = parser.parse_args()
    
    # 시드 설정
    set_random_seed(args.seed)
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # External 데이터 로딩
    if args.model_type == 'both':
        # PM과 AM 모두 테스트
        print("\n" + "="*50)
        print("PM Model External Test")
        print("="*50)
        pm_data = load_external_data(args.external_data, model_type='pm')
        pm_checkpoint = args.checkpoint.replace('am_', 'pm_')  # checkpoint 이름에서 모델 타입 변경
        evaluate_model_on_external('pm', pm_data, pm_checkpoint, args, device)
        
        print("\n" + "="*50)
        print("AM Model External Test")
        print("="*50)
        am_data = load_external_data(args.external_data, model_type='am')
        am_checkpoint = args.checkpoint.replace('pm_', 'am_')
        evaluate_model_on_external('am', am_data, am_checkpoint, args, device)
    else:
        # 단일 모델 테스트
        data = load_external_data(args.external_data, model_type=args.model_type)
        evaluate_model_on_external(args.model_type, data, args.checkpoint, args, device)
    
    print("\n" + "="*50)
    print("External Test 완료!")
    print("="*50)


if __name__ == "__main__":
    main()

