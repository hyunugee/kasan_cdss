#!/usr/bin/env python3
"""
타크로리무스 투여량 예측 ML 모델 개발
PyTorch 기반 구현

Author: AI Assistant
Date: 2024
"""

import argparse
import os
import random
import warnings
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import joblib

# 분리된 모델들 import
from timeseries_models import RNNModel, LSTMModel, TacrolimusLSTMTrainer, create_lstm_model, create_lstm_trainer
from tabular_models import (
    TacrolimusRegressorModel, 
    TacrolimusRegressorTrainer,
    create_regressor_model, create_regressor_trainer
)

# 유틸리티 함수 import
from utils import set_random_seed, generate_model_name, visualize_predictions, load_data, patient_wise_split, save_experiment_results

warnings.filterwarnings('ignore')


class TacrolimusDataset(Dataset):
    """타크로리무스 데이터셋 클래스"""
    
    def __init__(self, data: pd.DataFrame, static_features: Optional[pd.DataFrame] = None, 
                 scaler: Optional[StandardScaler] = None, is_training: bool = True,
                 model_type: str = 'pm', time_window: int = 1, use_timeseries: bool = False,
                 max_seq_len: int = 23):
        """
        Args:
            data: 투여량 및 TDM 데이터
            static_features: 환자 정적 특성 데이터
            scaler: 정규화 스케일러
            is_training: 학습용 데이터 여부
            model_type: 'pm' 또는 'am' 모델 타입
            time_window: 시계열 윈도우 크기 (과거 며칠의 데이터를 사용할지)
        """
        self.data = data.reset_index(drop=True)
        self.static_features = static_features
        self.scaler = scaler
        self.is_training = is_training
        self.model_type = model_type
        self.time_window = time_window
        self.use_timeseries = use_timeseries
        self.max_seq_len = max_seq_len  # 시퀀스 최대 길이 (패딩용)
        
        # 모델 타입에 따른 피처 및 타겟 컬럼 정의
        if model_type == 'pm':
            if use_timeseries:
                self.feature_cols = ['sequence']
                self.target_cols = ['today_dose_pm']
            else:   
                self.feature_cols = ['prev_dose_pm', 'today_dose_am', 'today_tdm']
                self.target_cols = ['today_dose_pm']
        else:  # am
            if use_timeseries:
                self.feature_cols = ['sequence']
                self.target_cols = ['next_dose_am']
            else:
                self.feature_cols = ['prev_dose_pm', 'today_dose_am', 'today_tdm', 'today_dose_pm']
                self.target_cols = ['next_dose_am']
        
        # 정적 특성 컬럼 (있는 경우)
        if static_features is not None:
            self.static_cols = [col for col in static_features.columns if col != 'patient_id']
        else:
            self.static_cols = []
        
        # 데이터 전처리
        self._preprocess_data()
    
    def _preprocess_data(self):
        """데이터 전처리"""
        # 결측값 처리 (day_number는 제외, sequence는 시퀀스 데이터이므로 제외)
        cols_to_check = [col for col in (self.feature_cols + self.target_cols) 
                        if col not in ['day_number', 'sequence']]
        if cols_to_check:  # 빈 리스트가 아닐 때만 dropna 실행
            self.data = self.data.dropna(subset=cols_to_check)
        
        # 정적 특성 결합
        if self.static_features is not None:
            self.data = self.data.merge(
                self.static_features, 
                on='patient_id', 
                how='left'
            )
            # 정적 특성 결측값을 0으로 채움
            self.data[self.static_cols] = self.data[self.static_cols].fillna(0)
        
        # 피처 정규화 (시계열 모델의 경우 sequence는 배열이므로 정규화 건너뜀)
        if not self.use_timeseries:
            if self.scaler is None and self.is_training:
                self.scaler = StandardScaler()
                self.data[self.feature_cols] = self.scaler.fit_transform(self.data[self.feature_cols])
            elif self.scaler is not None:
                self.data[self.feature_cols] = self.scaler.transform(self.data[self.feature_cols])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.use_timeseries:
            # 시계열 모델: sequence 컬럼 사용
            # sequence는 (seq_len, input_dim) 형태의 numpy array
            # PM 모델: input_dim=3 (prev_dose_pm, today_dose_am, today_tdm)
            # AM 모델: input_dim=4 (prev_dose_pm, today_dose_am, today_tdm, today_dose_pm)
            sequence = row['sequence']
            if isinstance(sequence, np.ndarray):
                features = torch.FloatTensor(sequence)  # (seq_len, input_dim)
            else:
                # 리스트나 다른 형태일 경우 변환
                features = torch.FloatTensor(np.array(sequence))
            
            # 실제 시퀀스 길이 저장 (패딩 전)
            original_seq_len = features.shape[0]
            input_dim = features.shape[1]  # PM=3, AM=4
            
            # 시퀀스 길이가 max_seq_len보다 작으면 패딩 (0으로 채움)
            if original_seq_len < self.max_seq_len:
                # 패딩: (max_seq_len - original_seq_len, input_dim) 형태의 0 텐서 추가
                padding = torch.zeros(self.max_seq_len - original_seq_len, input_dim)
                features = torch.cat([features, padding], dim=0)  # (max_seq_len, input_dim)
            elif original_seq_len > self.max_seq_len:
                # 시퀀스가 max_seq_len보다 크면 truncate
                features = features[:self.max_seq_len]
                original_seq_len = self.max_seq_len
            
            # 정적 특성은 별도로 반환 (LSTM 모델에서 사용)
            # DataLoader collate를 위해 None 대신 더미 텐서 사용
            if self.static_cols:
                static_feat = torch.FloatTensor([row[col] for col in self.static_cols])
            else:
                # 정적 특성이 없을 때 더미 텐서 반환
                static_feat = torch.FloatTensor([0.0])  # 크기 1인 더미 텐서
            
            # 타겟 (단일 값)
            target = torch.FloatTensor([row[self.target_cols[0]]])
            
            # LSTM 모델은 (features, sequence_length, static_features), target 형태로 반환
            # sequence_length는 pack_padded_sequence에서 사용
            return (features, torch.tensor(original_seq_len, dtype=torch.long), static_feat), target
        else:
            # Tabular 모델: 기존 방식
            if self.time_window == 1:
                # time_window=1: 당일 데이터만 사용
                # PM: [prev_dose_pm, today_dose_am, today_tdm] (3 features)
                # AM: [prev_dose_pm, today_dose_am, today_tdm, today_dose_pm] (4 features)
                features = torch.FloatTensor([row[col] for col in self.feature_cols])
            else:
                # time_window > 1: 과거 며칠의 데이터 사용
                features = self._get_window_features(idx)
            
            # 정적 특성 추가
            if self.static_cols:
                static_feat = torch.FloatTensor([row[col] for col in self.static_cols])
                features = torch.cat([features, static_feat])
            
            # 타겟 (단일 값)
            target = torch.FloatTensor([row[self.target_cols[0]]])
            
            return features, target
    
    def _get_window_features(self, idx):
        """
        시계열 윈도우 피처 생성
        
        PM 모델 (time_window=n):
        - 입력: 과거 n일의 [prev_dose_pm, today_dose_am, today_tdm] (3*n features)
        - 타겟: today_dose_pm
        
        AM 모델 (time_window=n):
        - 입력: 과거 n일의 [prev_dose_pm, today_dose_am, today_tdm] + 마지막 today_dose_pm (3*n+1 features)
        - 타겟: next_dose_am
        """
        patient_id = self.data.iloc[idx]['patient_id']
        
        # 현재 환자의 모든 데이터 가져오기 (시간순 정렬 가정)
        patient_mask = self.data['patient_id'] == patient_id
        patient_indices = self.data[patient_mask].index.tolist()
        
        # 현재 인덱스가 환자 데이터에서 몇 번째인지 찾기
        current_pos = patient_indices.index(idx)
        
        # 윈도우 크기만큼의 과거 데이터 수집
        window_features = []
        
        # PM 모델의 경우: 각 일의 [prev_dose_pm, today_dose_am, today_tdm] 수집
        # AM 모델의 경우: 각 일의 [prev_dose_pm, today_dose_am, today_tdm] + 마지막 today_dose_pm 수집
        
        base_feature_cols = ['prev_dose_pm', 'today_dose_am', 'today_tdm']  # 하루당 3개 피처
        
        # 과거 n일의 데이터 수집 (time_window일부터 현재까지)
        for i in range(self.time_window):
            # current_pos - i: 현재로부터 i일 전
            pos = current_pos - i
            
            if pos >= 0:
                # 과거 데이터 사용 가능
                past_idx = patient_indices[pos]
                past_row = self.data.iloc[past_idx]
                
                # 각 일의 기본 피처 3개: [prev_dose_pm, today_dose_am, today_tdm]
                window_features.extend([past_row[col] for col in base_feature_cols])
            else:
                # 과거 데이터가 없으면 0으로 패딩
                window_features.extend([0.0] * len(base_feature_cols))
        
        # AM 모델의 경우: 마지막 today_dose_pm 추가
        if self.model_type == 'am':
            # 현재 일의 today_dose_pm 추가
            if current_pos >= 0:
                current_row = self.data.iloc[patient_indices[current_pos]]
                window_features.append(current_row['today_dose_pm'])
            else:
                window_features.append(0.0)
        
        return torch.FloatTensor(window_features)


def prepare_sequence_data(data: pd.DataFrame, patients: List[str], 
                          model_type: str = 'pm',
                          max_seq_len: int = 23, include_day_number: bool = False) -> pd.DataFrame:
    """
    시계열 데이터 준비 - 각 시점별로 샘플 생성
    
    Args:
        data: 투여량 데이터
        patients: 환자 ID 리스트
        model_type: 'pm' 또는 'am' 모델 타입
        max_seq_len: 최대 시퀀스 길이 (기본값: 10)
        include_day_number: day_number 컬럼 포함 여부 (기본값: False)
    
    Returns:
        각 시점별 샘플이 포함된 DataFrame
    """
    sequence_data = []
    
    for patient in patients:
        patient_data = data[data["patient_id"] == patient].sort_values("patient_id")
        
        if len(patient_data) == 0:
            continue
        
        # 최대 길이로 truncate
        patient_data = patient_data.head(max_seq_len)
        
        # 각 시점별로 샘플 생성 (1일부터 마지막 날까지)
        for t in range(len(patient_data)):
            # 모델 타입에 따라 시퀀스 피처 선택
            if model_type == 'am':
                # AM 모델: 각 시점마다 4개 피처 (prev_dose_pm, today_dose_am, today_tdm, today_dose_pm)
                sequence = patient_data.iloc[:t+1][['prev_dose_pm', 'today_dose_am', 'today_tdm', 'today_dose_pm']].values
            else:  # pm
                # PM 모델: 각 시점마다 3개 피처 (prev_dose_pm, today_dose_am, today_tdm)
                sequence = patient_data.iloc[:t+1][['prev_dose_pm', 'today_dose_am', 'today_tdm']].values
            
            # t번째 시점의 타겟 값들
            target_row = patient_data.iloc[t]
            sample_dict = {
                'patient_id': patient,
                'sequence': sequence
            }
            
            # 모델 타입에 따라 필요한 타겟만 추가
            if 'today_dose_pm' in target_row.index:
                sample_dict['today_dose_pm'] = target_row['today_dose_pm']
            if model_type == 'am' and 'next_dose_am' in target_row.index:
                sample_dict['next_dose_am'] = target_row['next_dose_am']
            
            # day_number 추가 (요청된 경우)
            if include_day_number:
                day_number = target_row.get('day_number', t + 1) if 'day_number' in target_row.index else t + 1
                sample_dict['day_number'] = day_number
            
            sequence_data.append(sample_dict)
    
    return pd.DataFrame(sequence_data)


def train_and_evaluate_model(model_type: str, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                            test_data: pd.DataFrame, train_static: pd.DataFrame, 
                            val_static: pd.DataFrame, test_static: pd.DataFrame,
                            args, device: torch.device,
                            checkpoint_dir: str = "checkpoints", visualization_dir: str = "visualizations") -> str:
    """
    모델 학습 및 평가를 수행하는 통합 함수
    
    Args:
        model_type: 'pm' 또는 'am'
        train_data, val_data, test_data: 학습/검증/테스트 데이터
        train_static, val_static, test_static: 정적 특성 데이터
        args: 명령행 인자
        device: 사용할 디바이스
        checkpoint_dir: 체크포인트 디렉토리
        visualization_dir: 시각화 디렉토리
    
    Returns:
        생성된 모델 이름
    """
    model_name_upper = model_type.upper()
    print(f"\n=== {model_name_upper} Model Training ===")
    
    # 모델 이름 생성
    model_name = generate_model_name(model_type, args)
    print(f"{model_name_upper} 모델 이름: {model_name}")
    
    # 데이터셋 생성
    train_dataset = TacrolimusDataset(train_data, train_static, is_training=True, 
                                      model_type=model_type, time_window=args.time_window, 
                                      use_timeseries=args.use_timeseries, max_seq_len=args.max_seq_len)
    val_dataset = TacrolimusDataset(val_data, val_static, train_dataset.scaler, 
                                    is_training=False, model_type=model_type, 
                                    time_window=args.time_window, use_timeseries=args.use_timeseries,
                                    max_seq_len=args.max_seq_len)
    test_dataset = TacrolimusDataset(test_data, test_static, train_dataset.scaler, 
                                     is_training=False, model_type=model_type, 
                                     time_window=args.time_window, use_timeseries=args.use_timeseries,
                                     max_seq_len=args.max_seq_len)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 및 트레이너 생성
    if args.use_timeseries:
        # 시계열 모델: 모델 타입에 따라 입력 차원 다름
        if model_type == 'am':
            # AM 모델: 각 시점마다 4개 피처 (prev_dose_pm, today_dose_am, today_tdm, today_dose_pm)
            input_dim = 4
        else:  # pm
            # PM 모델: 각 시점마다 3개 피처 (prev_dose_pm, today_dose_am, today_tdm)
            input_dim = 3
        static_dim = len(train_static.columns) - 1 if args.use_static_features and train_static is not None else 0
        model = create_lstm_model(input_dim, args.hidden_dim, num_layers=args.num_layers,
                                 use_static=args.use_static_features, static_dim=static_dim, 
                                 rnn_type=args.rnn_type)
        trainer = create_lstm_trainer(model, device, args.learning_rate, model_name, model_type, checkpoint_dir="checkpoints")
        
        # 입력 차원 정보 출력 (샘플 확인)
        sample_batch, _ = next(iter(train_loader))
        if isinstance(sample_batch, (tuple, list)) and len(sample_batch) == 2:
            sample_features, _ = sample_batch
            seq_len = sample_features.shape[1]
            print(f"{model_name_upper} {args.rnn_type.upper()} 모델 입력 차원: {input_dim} (각 시점), 시퀀스 길이: {seq_len}, 레이어 수: {args.num_layers}")
        else:
            print(f"{model_name_upper} {args.rnn_type.upper()} 모델 입력 차원: {input_dim} (각 시점), 레이어 수: {args.num_layers}")
    else:
        # Tabular 모델
        model = TacrolimusRegressorModel(
            model_type=args.regressor_type,
            random_state=args.seed
        )
        trainer = create_regressor_trainer(model, model_name, model_type, checkpoint_dir="checkpoints")
        
        # 입력 차원 정보 출력
        sample_features, _ = next(iter(train_loader))
        input_dim = sample_features.shape[1]
        print(f"{model_name_upper} Tabular 모델 입력 차원: {input_dim} (time_window={args.time_window})")
    
    # 체크포인트 경로 확인
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pth' if args.use_timeseries else f'{model_name}.pkl')
    checkpoint_exists = os.path.exists(checkpoint_path)
    
    if checkpoint_exists:
        print(f"체크포인트 발견: {checkpoint_path}")
        print("기존 체크포인트를 로드하고 테스트만 수행합니다.")
        skip_training = True
    else:
        print(f"체크포인트 없음: {checkpoint_path}")
        print("새로 학습을 시작합니다.")
        skip_training = False
    
    # 체크포인트가 있으면 로드, 없으면 학습
    if skip_training:
        loaded = trainer.load_checkpoint()
        if not loaded:
            print(f"Warning: 체크포인트를 로드할 수 없습니다. 학습을 시작합니다.")
            trainer.train(train_loader, val_loader, args.epochs, args.patience)
    else:
        # 학습
        trainer.train(train_loader, val_loader, args.epochs, args.patience)
    
    # Scaler 저장 (Tabular 모델인 경우)
    if not args.use_timeseries and train_dataset.scaler is not None:
        scaler_path = os.path.join(checkpoint_dir, f'{model_name}_scaler.pkl')
        joblib.dump(train_dataset.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    # 테스트 평가 (예측값과 실제값 포함)
    test_loss, test_metrics, (test_predictions, test_actuals) = trainer.validate(test_loader, return_predictions=True)
    print(f"\n{model_name_upper} Model Test Results:")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    
    # 실험 결과 저장
    save_experiment_results(model_type, model_name, args, test_metrics, test_loss, results_dir="experiment_results")
    
    # 시각화
    visualize_predictions(test_data, test_predictions, test_actuals, 
                        model_type, model_name, max_day=8, test_dataset=test_dataset,
                        show_std=getattr(args, 'show_std', False), visualization_dir=visualization_dir,
                        use_static_features=args.use_static_features)
    
    return model_name


def main():
    parser = argparse.ArgumentParser(description='타크로리무스 투여량 예측 모델')
    
    # 데이터 경로
    parser.add_argument('--dose_data', type=str, 
                       default='data/extracted_df_Tac_dose_tdm.csv',
                       help='투여량 데이터 경로')
    parser.add_argument('--static_data', type=str,
                       default='data/2017_2022_registry.xlsx',
                       help='정적 특성 데이터 경로')
    
    # 모델 설정
    parser.add_argument('--use_timeseries', action='store_true',
                       help='시계열 모델 사용 여부')
    parser.add_argument('--use_static_features', action='store_true',
                       help='정적 특성 사용 여부')
    parser.add_argument('--model_type', type=str, choices=['pm', 'am', 'both'],
                       default='both', help='학습할 모델 타입')
    parser.add_argument('--regressor_type', type=str, 
                       choices=['random_forest', 'gradient_boosting', 'svr', 'ridge', 'lasso', 'elastic_net', 'knn', 'decision_tree', 'catboost', 'xgboost'],
                       default='random_forest', help='Regressor 모델 타입 (시계열 모델 사용 시 무시)')
    parser.add_argument('--time_window', type=int, default=1,
                       help='시계열 윈도우 크기 (과거 며칠의 데이터를 사용할지, tabular 모델용)')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='은닉층 차원')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='RNN 레이어 수 (LSTM/GRU)')
    parser.add_argument('--rnn_type', type=str, choices=['lstm', 'gru'], default='lstm',
                       help='RNN 타입 (lstm 또는 gru)')
    parser.add_argument('--max_seq_len', type=int, default=23,
                       help='최대 시퀀스 길이')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # 데이터 분할
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='테스트 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='검증 데이터 비율')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (cpu/cuda/auto)')
    parser.add_argument('--show_std', action='store_true',
                       help='Show standard deviation as shaded area in visualization')
    
    args = parser.parse_args()
    
    # 시드 설정 (재현성 보장)
    set_random_seed(args.seed)
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 데이터 로딩
    dose_data, static_data = load_data(args.dose_data, args.static_data)
    
    # 환자 단위 분할
    train_patients, val_patients, test_patients = patient_wise_split(
        dose_data, args.test_ratio, args.val_ratio
    )
    
    # 데이터셋 준비
    train_data = dose_data[dose_data["patient_id"].isin(train_patients)]
    val_data = dose_data[dose_data["patient_id"].isin(val_patients)]
    test_data = dose_data[dose_data["patient_id"].isin(test_patients)]
    
    # 정적 특성 준비
    train_static = static_data[static_data["patient_id"].isin(train_patients)] if args.use_static_features else None
    val_static = static_data[static_data["patient_id"].isin(val_patients)] if args.use_static_features else None
    test_static = static_data[static_data["patient_id"].isin(test_patients)] if args.use_static_features else None
    
    # 시계열 데이터 처리: 원본 데이터 저장 (시계열이 아닌 경우 사용)
    original_train_data = train_data
    original_val_data = val_data
    original_test_data = test_data
    
    # 모델 학습 및 평가
    checkpoint_dir = "checkpoints"
    visualization_dir = "visualizations"
    
    # 폴더 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    trained_models = {}
    
    if args.model_type in ['pm', 'both']:
        # PM 모델용 데이터 준비
        if args.use_timeseries:
            print("Preparing sequence data for PM model...")
            pm_train_data = prepare_sequence_data(original_train_data, train_patients, model_type='pm', max_seq_len=args.max_seq_len, include_day_number=False)
            pm_val_data = prepare_sequence_data(original_val_data, val_patients, model_type='pm', max_seq_len=args.max_seq_len, include_day_number=False)
            pm_test_data = prepare_sequence_data(original_test_data, test_patients, model_type='pm', max_seq_len=8, include_day_number=True)
        else:
            pm_train_data = original_train_data
            pm_val_data = original_val_data
            pm_test_data = original_test_data
        
        trained_models['pm'] = train_and_evaluate_model(
            'pm', pm_train_data, pm_val_data, pm_test_data,
            train_static, val_static, test_static, args, device,
            checkpoint_dir=checkpoint_dir, visualization_dir=visualization_dir
        )
    
    if args.model_type in ['am', 'both']:
        # AM 모델용 데이터 준비
        if args.use_timeseries:
            print("Preparing sequence data for AM model...")
            am_train_data = prepare_sequence_data(original_train_data, train_patients, model_type='am', max_seq_len=args.max_seq_len, include_day_number=False)
            am_val_data = prepare_sequence_data(original_val_data, val_patients, model_type='am', max_seq_len=args.max_seq_len, include_day_number=False)
            # NOTE: 테스트 데이터는 8일 이상의 데이터는 테스트에 사용하지 않음
            am_test_data = prepare_sequence_data(original_test_data, test_patients, model_type='am', max_seq_len=8, include_day_number=True)
        else:
            am_train_data = original_train_data
            am_val_data = original_val_data
            am_test_data = original_test_data
        
        trained_models['am'] = train_and_evaluate_model(
            'am', am_train_data, am_val_data, am_test_data,
            train_static, val_static, test_static, args, device,
            checkpoint_dir=checkpoint_dir, visualization_dir=visualization_dir
        )
    
    # 학습 완료 메시지 및 저장된 모델 파일 출력
    print("\nTraining completed!")
    print("저장된 모델 파일:")
    for model_type, model_name in trained_models.items():
        model_name_upper = model_type.upper()
        if args.use_timeseries:
            print(f"  - {model_name_upper} LSTM 모델: {model_name}.pth")
        else:
            print(f"  - {model_name_upper} {args.regressor_type} 모델: {model_name}.pkl")


if __name__ == "__main__":
    main()