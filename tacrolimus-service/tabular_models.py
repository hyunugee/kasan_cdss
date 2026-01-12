#!/usr/bin/env python3
"""
타크로리무스 투여량 예측 테이블 모델
다양한 Regressor 모델들 구현
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple


class TacrolimusRegressorModel:
    """타크로리무스 투여량 예측 Regressor 모델 (PM/AM 통합)"""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        # CatBoost와 XGBoost는 NaN을 직접 처리하고 스케일링이 필수는 아니므로 선택적 사용
        self.use_scaler = model_type not in ['catboost', 'xgboost']
        if self.use_scaler:
            self.scaler = StandardScaler()
            self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
    
    def _create_model(self, model_type: str, **kwargs):
        """모델 생성"""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42)
            ),
            'svr': SVR(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale')
            ),
            'ridge': Ridge(
                alpha=kwargs.get('alpha', 1.0),
                random_state=kwargs.get('random_state', 42)
            ),
            'lasso': Lasso(
                alpha=kwargs.get('alpha', 1.0),
                random_state=kwargs.get('random_state', 42)
            ),
            'elastic_net': ElasticNet(
                alpha=kwargs.get('alpha', 1.0),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                random_state=kwargs.get('random_state', 42)
            ),
            'knn': KNeighborsRegressor(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'uniform')
            ),
            'decision_tree': DecisionTreeRegressor(
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            ),
            'catboost': CatBoostRegressor(
                iterations=kwargs.get('iterations', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                depth=kwargs.get('depth', 6),
                random_seed=kwargs.get('random_state', 42),
                verbose=False
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=kwargs.get('random_state', 42),
                verbosity=0
            )
        }
        return models.get(model_type, models['random_forest'])
    
    def fit(self, X, y):
        """모델 학습"""
        if self.use_scaler:
            # 다른 모델: NaN을 중앙값으로 채우고 스케일링
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
        # CatBoost/XGBoost: NaN을 직접 처리 (스케일링 없음)
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X):
        """예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        if self.use_scaler:
            # 다른 모델: NaN을 중앙값으로 채우고 스케일링
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
        # CatBoost/XGBoost: NaN을 직접 처리 (스케일링 없음)
        return self.model.predict(X)


class TacrolimusRegressorTrainer:
    """타크로리무스 Regressor 모델 학습 클래스"""
    
    def __init__(self, model, model_name: str = "model", model_type: str = "pm",
                 checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type  # "pm" 또는 "am"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pkl')
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """전체 학습 과정"""
        # 데이터 준비
        X_train, y_train = self._prepare_data(train_loader)
        X_val, y_val = self._prepare_data(val_loader)
        
        # 모델 학습
        print(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        
        # 검증
        val_loss, val_metrics = self.validate(val_loader)
        
        print(f"Training completed!")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}, Val R²: {val_metrics['r2']:.4f}")
        
        # 모델 저장
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        joblib.dump(self.model, self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")
        
        return {
            'train_losses': [val_loss],  # Regressor는 한 번만 학습
            'val_losses': [val_loss]
        }
    
    def _prepare_data(self, dataloader: DataLoader):
        """데이터로더에서 데이터 추출"""
        X_list = []
        y_list = []
        
        for features, targets in dataloader:
            X_list.append(features.numpy())
            y_list.append(targets.numpy())
        
        X = np.vstack(X_list)
        # 타겟은 이미 단일 값으로 제공됨
        y = np.hstack([target.squeeze() for target in y_list])
        
        return X, y
    
    def validate(self, dataloader: DataLoader, return_predictions: bool = False):
        """검증"""
        X_val, y_val = self._prepare_data(dataloader)
        
        # 예측
        y_pred = self.model.predict(X_val)
        
        # 메트릭 계산
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        # MAPE 계산 (0으로 나누는 경우 처리)
        non_zero_mask = y_val != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_val[non_zero_mask] - y_pred[non_zero_mask]) / y_val[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        if return_predictions:
            return mse, metrics, (y_pred, y_val)
        return mse, metrics
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            self.model = joblib.load(self.checkpoint_path)
            print(f"Loaded checkpoint from {self.checkpoint_path}")
            return True
        return False


def create_regressor_model(model_type: str, **kwargs):
    """Regressor 모델 생성 헬퍼 함수"""
    return TacrolimusRegressorModel(model_type, **kwargs)


def create_regressor_trainer(model, model_name: str = "regressor_model", model_type: str = "pm", checkpoint_dir: str = "checkpoints"):
    """Regressor 트레이너 생성 헬퍼 함수"""
    return TacrolimusRegressorTrainer(model, model_name, model_type, checkpoint_dir=checkpoint_dir)
