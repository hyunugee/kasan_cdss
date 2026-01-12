#!/usr/bin/env python3
"""
타크로리무스 투여량 예측 시계열 모델
LSTM 기반 시계열 모델 구현
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np


class RNNModel(nn.Module):
    """시계열 RNN 모델 (LSTM/GRU)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, use_static: bool = False, static_dim: int = 0,
                 rnn_type: str = 'lstm'):
        super(RNNModel, self).__init__()
        
        self.use_static = use_static
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # RNN 레이어 (LSTM 또는 GRU)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers, 
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers, 
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'lstm' or 'gru'")
        
        # 정적 특성 처리
        if use_static and static_dim > 0:
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.final_dim = hidden_dim + hidden_dim // 2
        else:
            self.final_dim = hidden_dim
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(self.final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, lengths=None, static_features=None):
        """
        Args:
            x: 입력 시퀀스 (batch_size, seq_len, input_dim)
            lengths: 실제 시퀀스 길이 (batch_size,), None이면 패딩 마스킹 안 함
            static_features: 정적 특성 (batch_size, static_dim)
        """
        # 패딩 마스킹 처리
        if lengths is not None:
            # pack_padded_sequence: 패딩된 부분을 제외하고 RNN 처리
            # lengths는 내림차순 정렬되어 있어야 함
            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            rnn_out_packed, _ = self.rnn(x_packed)
            # pad_packed_sequence: 다시 패딩된 형태로 변환
            rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True)
            
            # 실제 마지막 시점의 출력 추출
            # 각 시퀀스의 실제 마지막 위치에서 출력 가져오기
            batch_size = x.size(0)
            # lengths는 CPU tensor이므로 GPU로 변환 필요
            lengths_gpu = lengths.to(rnn_out.device)
            rnn_features = rnn_out[torch.arange(batch_size, device=rnn_out.device), lengths_gpu - 1, :]
        else:
            # 길이 정보가 없으면 기존 방식 (마지막 시점 사용)
            rnn_out, _ = self.rnn(x)
            rnn_features = rnn_out[:, -1, :]  # 마지막 시점의 출력
        
        # 정적 특성 결합
        if self.use_static and static_features is not None:
            static_out = self.static_net(static_features)
            combined = torch.cat([rnn_features, static_out], dim=1)
        else:
            combined = rnn_features
        
        return self.output_layer(combined)


# 하위 호환성을 위한 별칭
LSTMModel = RNNModel

class TacrolimusLSTMTrainer:
    """타크로리무스 LSTM 모델 학습 클래스"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 model_name: str = "model", model_type: str = "pm",
                 checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.model_type = model_type  # "pm" 또는 "am"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        
        for batch_data, targets in tqdm(dataloader, desc="Training"):
            # batch_data는 (features, sequence_length, static_features) 튜플 또는 리스트 형태
            # DataLoader가 튜플을 배치로 묶을 때 리스트로 변환될 수 있음
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 3:
                features, seq_lengths, static_features = batch_data
                features = features.to(self.device)
                # pack_padded_sequence는 lengths가 CPU tensor여야 함
                seq_lengths = seq_lengths.cpu()
                # 더미 텐서([0.0])인 경우 None으로 처리 (정적 특성이 없는 경우)
                # 배치로 묶이면 shape는 (batch_size, static_dim)
                # 정적 특성이 있으면 static_dim > 1, 없으면 더미 텐서는 static_dim = 1
                if static_features is not None and static_features.shape[-1] > 1:
                    static_features = static_features.to(self.device)
                else:
                    static_features = None
            elif isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                # 하위 호환성: (features, static_features) 형태
                features, static_features = batch_data
                features = features.to(self.device)
                seq_lengths = None
                if static_features is not None and static_features.shape[-1] > 1:
                    static_features = static_features.to(self.device)
                else:
                    static_features = None
            else:
                # features만 있는 경우
                features = batch_data.to(self.device)
                seq_lengths = None
                static_features = None
            
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(features, lengths=seq_lengths, static_features=static_features)
            # 타겟은 이미 단일 값으로 제공됨
            loss = self.criterion(pred.squeeze(), targets.squeeze())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader, return_predictions: bool = False):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data, targets in tqdm(dataloader, desc="Validation"):
                # batch_data는 (features, sequence_length, static_features) 튜플 또는 리스트 형태
                # DataLoader가 튜플을 배치로 묶을 때 리스트로 변환될 수 있음
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 3:
                    features, seq_lengths, static_features = batch_data
                    features = features.to(self.device)
                    # pack_padded_sequence는 lengths가 CPU tensor여야 함
                    seq_lengths = seq_lengths.cpu()
                    # 더미 텐서([0.0])인 경우 None으로 처리 (정적 특성이 없는 경우)
                    # 배치로 묶이면 shape는 (batch_size, static_dim)
                    # 정적 특성이 있으면 static_dim > 1, 없으면 더미 텐서는 static_dim = 1
                    if static_features is not None and static_features.shape[-1] > 1:
                        static_features = static_features.to(self.device)
                    else:
                        static_features = None
                elif isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    # 하위 호환성: (features, static_features) 형태
                    features, static_features = batch_data
                    features = features.to(self.device)
                    seq_lengths = None
                    if static_features is not None and static_features.shape[-1] > 1:
                        static_features = static_features.to(self.device)
                    else:
                        static_features = None
                else:
                    # features만 있는 경우
                    features = batch_data.to(self.device)
                    seq_lengths = None
                    static_features = None
                
                targets = targets.to(self.device)
                
                pred = self.model(features, lengths=seq_lengths, static_features=static_features)
                # 타겟은 이미 단일 값으로 제공됨
                loss = self.criterion(pred.squeeze(), targets.squeeze())
                
                all_preds.extend(pred.squeeze().cpu().numpy())
                all_targets.extend(targets.squeeze().cpu().numpy())
                
                total_loss += loss.item()
        
        # 메트릭 계산
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_preds)
        
        # MAPE 계산 (0으로 나누는 경우 처리)
        non_zero_mask = all_targets != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((all_targets[non_zero_mask] - all_preds[non_zero_mask]) / all_targets[non_zero_mask])) * 100
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
            return total_loss / len(dataloader), metrics, (all_preds, all_targets)
        return total_loss / len(dataloader), metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 10) -> Dict[str, List[float]]:
        """전체 학습 과정"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 학습
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f}, Val R²: {val_metrics['r2']:.4f}")
            print("-" * 50)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 모델 저장
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Loaded checkpoint from {self.checkpoint_path}")
            return True
        return False


def create_lstm_model(input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                     use_static: bool = False, static_dim: int = 0, rnn_type: str = 'lstm') -> RNNModel:
    """RNN 모델 생성 헬퍼 함수 (LSTM 또는 GRU)"""
    return RNNModel(input_dim, hidden_dim, num_layers=num_layers, 
                   use_static=use_static, static_dim=static_dim, rnn_type=rnn_type)


def create_lstm_trainer(model: nn.Module, device: torch.device, 
                       learning_rate: float = 0.001, model_name: str = "lstm_model", 
                       model_type: str = "pm", checkpoint_dir: str = "checkpoints") -> TacrolimusLSTMTrainer:
    """LSTM 트레이너 생성 헬퍼 함수"""
    return TacrolimusLSTMTrainer(model, device, learning_rate, model_name=model_name, model_type=model_type, checkpoint_dir=checkpoint_dir)
