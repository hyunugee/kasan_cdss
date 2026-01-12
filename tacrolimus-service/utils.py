#!/usr/bin/env python3
"""
유틸리티 함수 모음
재현성, 모델 이름 생성, 시각화, 데이터 로딩 등의 공통 기능
"""

import os
import random
import json
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백엔드 사용

from torch.utils.data import Dataset


def set_random_seed(seed: int = 42) -> None:
    """
    재현성을 위한 난수 시드 설정
    
    Args:
        seed: 난수 시드 값
    """
    # Python 내장 random 모듈
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch 재현성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 환경 변수 설정 (일부 라이브러리용)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


def generate_model_name(model_type: str, args) -> str:
    """모델 이름 생성"""
    name_parts = [model_type]
    
    if args.use_timeseries:
        rnn_type = getattr(args, 'rnn_type', 'lstm')
        name_parts.append(rnn_type)
    elif hasattr(args, 'regressor_type') and args.regressor_type:
        name_parts.append(args.regressor_type)
    else:
        name_parts.append("mlp")
    
    if args.use_static_features:
        name_parts.append("static")
    
    name_parts.extend([
        f"ep{args.epochs}",
        f"bs{args.batch_size}",
        f"lr{args.learning_rate}",
        f"hd{args.hidden_dim}",
        f"tw{args.time_window}",
        f"seed{args.seed}"
    ])
    
    # num_layers 추가 (시계열 모델인 경우)
    if args.use_timeseries and hasattr(args, 'num_layers'):
        name_parts.append(f"nl{args.num_layers}")
    
    # max_seq_len 추가 (시계열 모델인 경우)
    if args.use_timeseries and hasattr(args, 'max_seq_len'):
        name_parts.append(f"msl{args.max_seq_len}")
    
    return "_".join(name_parts)


def visualize_predictions(test_data: pd.DataFrame, predictions: np.ndarray, 
                          actuals: np.ndarray, model_type: str, model_name: str,
                          max_day: int = 8, test_dataset: Optional[Dataset] = None,
                          show_std: bool = False, visualization_dir: str = "visualizations",
                          use_static_features: bool = False, dr2_data: Optional[np.ndarray] = None,
                          is_external_test: bool = False):
    """
    테스트 데이터의 예측값과 실제값을 시각화
    
    Args:
        test_data: 원본 테스트 데이터 (day_number 포함)
        predictions: 예측값 배열
        actuals: 실제값 배열
        model_type: 'pm' 또는 'am'
        model_name: 모델 이름
        max_day: 최대 시각화할 day_number
        test_dataset: 테스트 데이터셋 (선택사항, 필터링된 데이터에 접근용)
        show_std: 표준편차를 에러바로 표시할지 여부
        use_static_features: 정적 특성 사용 여부
        dr2_data: Clinician2 (DR2) 데이터 (선택사항, external test용)
        is_external_test: external test 여부 (True이면 "Actual (Clinician1)"로 표시)
    """
    # External test인지에 따라 레이블 결정
    actual_label = 'Actual (Clinician1)' if is_external_test else 'Actual (Clinician)'
    # 타겟 컬럼명 결정
    if model_type == 'pm':
        target_col = 'today_dose_pm'
    else:  # am
        target_col = 'next_dose_am'
    
    # test_dataset의 data는 이미 필터링되어 있고 predictions와 순서가 일치
    dataset_df = test_dataset.data.reset_index(drop=True)
    
    # day_number가 있는지 확인 (시계열 모델은 day_number가 없음)
    if 'day_number' in dataset_df.columns:
        # Tabular 모델: day_number 기반 시각화
        plot_data = pd.DataFrame({
            'day_number': dataset_df['day_number'].values,
            'predicted': predictions,
            'actual': actuals
        })
        
        # DR2 데이터가 있으면 추가
        if dr2_data is not None:
            plot_data['clinician2'] = dr2_data
        
        # day_number가 8 이하인 데이터만 필터링
        plot_data = plot_data[plot_data['day_number'] <= max_day].copy()
        
        if len(plot_data) == 0:
            print(f"Warning: day_number <= {max_day}인 데이터가 없습니다.")
            return
        
        # day_number별로 그룹화하여 평균 및 표준편차 계산 (같은 day_number의 여러 환자/시퀀스)
        if show_std:
            agg_dict = {
                'predicted': ['mean', 'std'],
                'actual': ['mean', 'std']
            }
            if dr2_data is not None:
                agg_dict['clinician2'] = ['mean', 'std']
        else:
            agg_dict = {
                'predicted': 'mean',
                'actual': 'mean'
            }
            if dr2_data is not None:
                agg_dict['clinician2'] = 'mean'
        
        plot_df = plot_data.groupby('day_number').agg(agg_dict).reset_index()
        
        # 컬럼명 정리 (MultiIndex 제거)
        if show_std:
            if dr2_data is not None:
                plot_df.columns = ['day_number', 'predicted_mean', 'predicted_std', 'actual_mean', 'actual_std', 'clinician2_mean', 'clinician2_std']
                clinician2_std = plot_df['clinician2_std'].fillna(0).values
            else:
                plot_df.columns = ['day_number', 'predicted_mean', 'predicted_std', 'actual_mean', 'actual_std']
                clinician2_std = None
            predicted_std = plot_df['predicted_std'].fillna(0).values
            actual_std = plot_df['actual_std'].fillna(0).values
            # 평균값만 사용 (CSV 저장용)
            plot_df['predicted'] = plot_df['predicted_mean']
            plot_df['actual'] = plot_df['actual_mean']
            if dr2_data is not None:
                plot_df['clinician2'] = plot_df['clinician2_mean']
        else:
            if dr2_data is not None:
                plot_df.columns = ['day_number', 'predicted', 'actual', 'clinician2']
            else:
                plot_df.columns = ['day_number', 'predicted', 'actual']
            predicted_std = None
            actual_std = None
            clinician2_std = None
        
        # CSV 저장용 데이터 생성 (day1~day8 컬럼, Actual/ML Predicted/Clinician2 행)
        csv_data = {}
        for day in range(1, max_day + 1):
            day_data = plot_df[plot_df['day_number'] == day]
            if len(day_data) > 0:
                row_data = [
                    day_data['actual'].values[0] if len(day_data['actual'].values) > 0 else None,
                    day_data['predicted'].values[0] if len(day_data['predicted'].values) > 0 else None
                ]
                if dr2_data is not None:
                    row_data.append(day_data['clinician2'].values[0] if 'clinician2' in day_data.columns and len(day_data['clinician2'].values) > 0 else None)
                csv_data[f'day{day}'] = row_data
            else:
                csv_data[f'day{day}'] = [None, None, None] if dr2_data is not None else [None, None]
        
        # CSV index 레이블 (external test일 때는 Clinician1로 표시)
        if dr2_data is not None:
            index_labels = [actual_label, 'ML Predicted', 'Clinician2']
        else:
            index_labels = [actual_label, 'ML Predicted']
        csv_df = pd.DataFrame(csv_data, index=index_labels).round(2)
        
        # CSV 저장
        os.makedirs(visualization_dir, exist_ok=True)
        csv_output_path = os.path.join(visualization_dir, f'{model_name}_mean_by_day.csv')
        csv_df.to_csv(csv_output_path, encoding='utf-8-sig')
        print(f"평균값 CSV가 {csv_output_path}에 저장되었습니다.")
        
        # 시각화
        # 기존 방식 (단일 플롯) - show_std 여부와 관계없이 동일한 형식 사용
        plt.figure(figsize=(10, 6))
        
        # show_std가 True이면 error bar 추가
        if show_std:
            plt.errorbar(plot_df['day_number'], plot_df['actual'], yerr=actual_std, 
                        fmt='o-', label=actual_label, linewidth=2, markersize=8, 
                        color='#2E86AB', capsize=5, capthick=2, elinewidth=1.5)
            plt.errorbar(plot_df['day_number'], plot_df['predicted'], yerr=predicted_std, 
                        fmt='s-', label='ML Predicted', linewidth=2, markersize=8, 
                        color='#A23B72', capsize=5, capthick=2, elinewidth=1.5)
            
            # DR2 데이터가 있으면 추가
            if dr2_data is not None and 'clinician2' in plot_df.columns:
                plt.errorbar(plot_df['day_number'], plot_df['clinician2'], yerr=clinician2_std, 
                            fmt='^-', label='Clinician2', linewidth=2, markersize=8, 
                            color='#F18F01', capsize=5, capthick=2, elinewidth=1.5)
        else:
            plt.plot(plot_df['day_number'], plot_df['actual'], 'o-', label=actual_label, 
                    linewidth=2, markersize=8, color='#2E86AB')
            plt.plot(plot_df['day_number'], plot_df['predicted'], 's-', label='ML Predicted', 
                    linewidth=2, markersize=8, color='#A23B72')
            
            # DR2 데이터가 있으면 추가
            if dr2_data is not None and 'clinician2' in plot_df.columns:
                plt.plot(plot_df['day_number'], plot_df['clinician2'], '^-', label='Clinician2', 
                        linewidth=2, markersize=8, color='#F18F01')
        
        plt.xlabel('Day Number', fontsize=12)
        plt.ylabel('Mean tacrolimus dose (mg)', fontsize=12)
        if model_type == 'am':
            title = 'Mean tacrolimus morning dose (AM) predictions'
        else:  # pm
            title = 'Mean tacrolimus evening dose (PM) predictions'
        if use_static_features:
            title += ' (with Static Features)'
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.5, max_day + 0.5)
        plt.xticks(range(1, max_day + 1))
        plt.ylim(1.25, 3.25)
        plt.yticks(np.arange(1.25, 3.50, 0.25))
    else:
        # 시계열 모델: 환자당 하나의 예측값만 있음 (scatter plot 사용)
        plot_data = pd.DataFrame({
            'predicted': predictions,
            'actual': actuals
        })
        
        # DR2 데이터가 있으면 추가
        if dr2_data is not None:
            plot_data['clinician2'] = dr2_data
        
        # 시각화
        plt.figure(figsize=(10, 6))
        
        # Scatter plot - ML Predicted
        plt.scatter(plot_data['actual'], plot_data['predicted'], 
                   alpha=0.6, s=50, color='#A23B72', marker='s', label='ML Predicted')
        
        # Scatter plot - Clinician2 (DR2)
        if dr2_data is not None:
            plt.scatter(plot_data['actual'], plot_data['clinician2'], 
                       alpha=0.6, s=50, color='#F18F01', marker='^', label='Clinician2')
        
        # Perfect prediction line (y=x)
        min_val = plot_data['actual'].min()
        max_val = plot_data['actual'].max()
        if dr2_data is not None:
            min_val = min(min_val, plot_data['clinician2'].min())
            max_val = max(max_val, plot_data['clinician2'].max())
        min_val = min(min_val, plot_data['predicted'].min())
        max_val = max(max_val, plot_data['predicted'].max())
        
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction', alpha=0.8)
        
        plt.xlabel(actual_label, fontsize=12)
        plt.ylabel('Predicted Dose (mg)', fontsize=12)
        if model_type == 'am':
            title = 'Mean tacrolimus morning dose (AM) predictions'
        else:  # pm
            title = 'Mean tacrolimus evening dose (PM) predictions'
        if use_static_features:
            title += ' (with Static Features)'
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # CSV 저장 (시계열 모델)
        csv_data = {
            'actual': plot_data['actual'].values,
            'predicted': plot_data['predicted'].values
        }
        if dr2_data is not None:
            csv_data['clinician2'] = plot_data['clinician2'].values
        
        csv_df = pd.DataFrame(csv_data).round(2)
        os.makedirs(visualization_dir, exist_ok=True)
        csv_output_path = os.path.join(visualization_dir, f'{model_name}_predictions.csv')
        csv_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
        print(f"예측값 CSV가 {csv_output_path}에 저장되었습니다.")
    
    # 그래프 저장
    os.makedirs(visualization_dir, exist_ok=True)
    filename_suffix = '_std' if show_std else ''
    output_path = os.path.join(visualization_dir, f'{model_name}_predictions_vs_actual{filename_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 그래프가 {output_path}로 저장되었습니다.")


def load_data(data_path: str, static_path: str, dose_cap: float = 7.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터 로딩
    
    Args:
        data_path: 투여량 데이터 경로
        static_path: 정적 특성 데이터 경로
        dose_cap: dose 컬럼의 최대값 임계값 (이 값 초과인 환자 제거)
    
    Returns:
        dose_data, static_data
    """
    print("Loading data...")
    
    # 투여량 데이터
    dose_data = pd.read_csv(data_path)
    dose_data["patient_id"] = dose_data["j"].apply(lambda x: x.split(".")[0])
    dose_data = dose_data.drop(columns=['j'])
    
    # 컬럼명 매핑 (실제 데이터 컬럼명 → 코드에서 사용하는 컬럼명)
    column_mapping = {
        'before_pm_str': 'prev_dose_pm',
        'today_am': 'today_dose_am', 
        'today_tdm': 'today_tdm',
        'y1_pm': 'today_dose_pm',
        'y2_am': 'next_dose_am'
    }
    dose_data = dose_data.rename(columns=column_mapping)
    
    # dose 컬럼에서 임계값 초과인 값이 있는 환자 제거
    dose_cols = ["prev_dose_pm", "today_dose_am", "today_dose_pm", "next_dose_am"]
    outlier_mask = (dose_data[dose_cols] > dose_cap).any(axis=1)
    outlier_patients = dose_data[outlier_mask]["patient_id"].unique()
    dose_data = dose_data[~dose_data["patient_id"].isin(outlier_patients)].copy()
    
    if len(outlier_patients) > 0:
        print(f"Removed {len(outlier_patients)} patients with dose values > {dose_cap}")
    
    # 정적 특성 데이터
    static_data = pd.read_excel(static_path)
    static_data = static_data[['NO','AGE','Sex','Bwt','Ht','BMI','Cause','HD_type','HD_duration','DM','HTN']]
    static_data["patient_id"] = static_data["NO"].astype(str)
    static_data = static_data.drop(columns=['NO'])
    
    # 환자 ID 매칭
    common_patients = set(dose_data["patient_id"].unique()) & set(static_data["patient_id"].unique())
    dose_data = dose_data[dose_data["patient_id"].isin(common_patients)]
    static_data = static_data[static_data["patient_id"].isin(common_patients)]
    
    # 환자당 row 수 계산
    patient_row_counts = dose_data.groupby('patient_id').size()
    
    # 4개 미만인 환자 제거
    patients_too_few = patient_row_counts[patient_row_counts < 4].index.tolist()
    if len(patients_too_few) > 0:
        dose_data = dose_data[~dose_data["patient_id"].isin(patients_too_few)].copy()
        static_data = static_data[~static_data["patient_id"].isin(patients_too_few)].copy()
        print(f"Removed {len(patients_too_few)} patients with < 4 rows")
    
    # 23개 초과인 환자 제거
    patient_row_counts = dose_data.groupby('patient_id').size()
    patients_too_many = patient_row_counts[patient_row_counts > 23].index.tolist()
    if len(patients_too_many) > 0:
        dose_data = dose_data[~dose_data["patient_id"].isin(patients_too_many)].copy()
        static_data = static_data[~static_data["patient_id"].isin(patients_too_many)].copy()
        print(f"Removed {len(patients_too_many)} patients with > 23 rows")
    
    # 각 환자별로 day_number 컬럼 추가 (인덱스 순서 기준으로 첫 번째 날, 두 번째 날...)
    dose_data = dose_data.sort_index().reset_index(drop=True)
    dose_data['day_number'] = dose_data.groupby('patient_id').cumcount() + 1
    
    print(f"Loaded {len(dose_data)} dose records from {len(dose_data['patient_id'].unique())} patients")
    
    return dose_data, static_data


def patient_wise_split(data: pd.DataFrame, test_ratio: float = 0.2, 
                       val_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    """환자 단위 데이터 분할"""
    patients = data["patient_id"].unique()
    n_patients = len(patients)
    
    # 랜덤 셔플
    np.random.shuffle(patients)
    
    # 분할
    n_test = int(n_patients * test_ratio)
    n_val = int(n_patients * val_ratio)
    
    test_patients = patients[:n_test]
    val_patients = patients[n_test:n_test + n_val]
    train_patients = patients[n_test + n_val:]
    
    print(f"Patient split - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    return train_patients, val_patients, test_patients


def save_experiment_results(model_type: str, model_name: str, args: Any, 
                           test_metrics: Dict[str, float], test_loss: float,
                           results_dir: str = "experiment_results") -> str:
    """
    실험 결과를 JSON 파일로 저장
    
    Args:
        model_type: 모델 타입 ('pm' 또는 'am')
        model_name: 모델 이름
        args: argparse 인자 객체
        test_metrics: 테스트 메트릭 (MAE, RMSE, R² 등)
        test_loss: 테스트 loss
        results_dir: 결과 저장 디렉토리
    
    Returns:
        저장된 JSON 파일 경로
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # 하이퍼파라미터 수집
    hyperparams = {
        'model_type': model_type,
        'model_name': model_name,
        'use_timeseries': getattr(args, 'use_timeseries', False),
        'use_static_features': getattr(args, 'use_static_features', False),
        'epochs': getattr(args, 'epochs', 100),
        'batch_size': getattr(args, 'batch_size', 32),
        'learning_rate': getattr(args, 'learning_rate', 0.001),
        'hidden_dim': getattr(args, 'hidden_dim', 64),
        'patience': getattr(args, 'patience', 10),
        'seed': getattr(args, 'seed', 42),
    }
    
    # 시계열 모델인 경우 추가 파라미터
    if getattr(args, 'use_timeseries', False):
        hyperparams.update({
            'num_layers': getattr(args, 'num_layers', 2),
            'rnn_type': getattr(args, 'rnn_type', 'lstm'),
            'max_seq_len': getattr(args, 'max_seq_len', 23),
        })
    else:
        # Tabular 모델인 경우
        hyperparams.update({
            'regressor_type': getattr(args, 'regressor_type', 'random_forest'),
            'time_window': getattr(args, 'time_window', 1),
        })
    
    # 실험 결과 (소수점 셋째자리에서 반올림)
    results = {
        'test_loss': round(float(test_loss), 3),
        'test_mae': round(float(test_metrics.get('mae', 0.0)), 3),
        'test_rmse': round(float(test_metrics.get('rmse', 0.0)), 3),
        'test_r2': round(float(test_metrics.get('r2', 0.0)), 3),
        'test_mape': round(float(test_metrics.get('mape', 0.0)), 3),
    }
    
    # 전체 데이터 구조
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparams,
        'results': results,
    }
    
    # 파일명 생성 (모델 이름 기반)
    filename = f"{model_name}_results.json"
    filepath = os.path.join(results_dir, filename)
    
    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"실험 결과가 {filepath}에 저장되었습니다.")
    
    return filepath

