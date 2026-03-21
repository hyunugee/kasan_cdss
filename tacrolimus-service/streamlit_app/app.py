import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
import re
import sys
import base64
import gspread
from google.oauth2.service_account import Credentials
import json

# 상위 디렉토리의 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeseries_models import RNNModel

# 페이지 설정
st.set_page_config(
    page_title="신장이식 환자 FK 레벨 추적기",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets 설정
SPREADSHEET_NAME = "tacrolimus_tdm_data"
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# 모델 경로 설정 (Streamlit Cloud와 로컬 모두 지원)
# 현재 파일의 위치를 기준으로 프로젝트 루트 찾기
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def get_gspread_client():
    """Google Sheets 클라이언트 초기화"""
    try:
        # Streamlit Cloud secrets에서 인증 정보 가져오기 시도
        try:
            if hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
                credentials_dict = dict(st.secrets["gcp_service_account"])
                credentials = Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=SCOPES
                )
            else:
                raise KeyError("Secrets not found, trying local file")
        except (KeyError, FileNotFoundError, AttributeError):
            # 로컬 개발 환경: JSON 파일에서 읽기
            json_path = os.path.join(os.path.dirname(__file__), 'service_account.json')
            if not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"service_account.json 파일을 찾을 수 없습니다: {json_path}\n"
                    "로컬 개발: service_account.json 파일을 streamlit_app/ 디렉토리에 배치하세요.\n"
                    "Streamlit Cloud: Secrets 설정에서 gcp_service_account를 구성하세요."
                )
            credentials = Credentials.from_service_account_file(
                json_path,
                scopes=SCOPES
            )
        
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Google Sheets 연결 오류: {str(e)}")
        return None

def get_or_create_spreadsheet():
    """스프레드시트 가져오기 또는 생성"""
    try:
        client = get_gspread_client()
        if not client:
            return None
        
        try:
            # 기존 스프레드시트 열기
            spreadsheet = client.open(SPREADSHEET_NAME)
        except gspread.SpreadsheetNotFound:
            # 스프레드시트 생성
            spreadsheet = client.create(SPREADSHEET_NAME)
            st.success(f"새 스프레드시트 '{SPREADSHEET_NAME}' 생성됨")
        
        return spreadsheet
    except Exception as e:
        st.error(f"스프레드시트 접근 오류: {str(e)}")
        return None

def get_or_create_worksheet(patient_id):
    """환자별 워크시트 가져오기 또는 생성"""
    try:
        spreadsheet = get_or_create_spreadsheet()
        if not spreadsheet:
            return None
        
        try:
            # 기존 워크시트 열기
            worksheet = spreadsheet.worksheet(patient_id)
        except gspread.WorksheetNotFound:
            # 새 워크시트 생성
            worksheet = spreadsheet.add_worksheet(title=patient_id, rows=100, cols=10)
            
            # 헤더 설정
            headers = ['Day', '전날 오후 FK용량', '당일 오전 FK용량', 'FK TDM', '당일 오후 FK용량']
            worksheet.update('A1:E1', [headers])

            # Day 1-8 초기화
            days_data = [[i, None, None, None, None] for i in range(1, 9)]
            worksheet.update('A2:E9', days_data)
        
        return worksheet
    except Exception as e:
        st.error(f"워크시트 접근 오류: {str(e)}")
        return None

def load_patient_data_from_sheets(patient_id):
    """Google Sheets에서 환자 데이터 로드"""
    try:
        worksheet = get_or_create_worksheet(patient_id)
        if not worksheet:
            return {}
        
        # 모든 데이터 가져오기
        all_data = worksheet.get_all_values()
        
        if len(all_data) < 2:
            return {}
        
        # DataFrame으로 변환
        headers = all_data[0]
        data = all_data[1:]
        df = pd.DataFrame(data, columns=headers)
        
        # Day 컬럼을 숫자로 변환
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
        
        # 빈 문자열을 None으로 변환
        for col in ['전날 오후 FK용량', '당일 오전 FK용량', 'FK TDM', '당일 오후 FK용량']:
            if col in df.columns:
                df[col] = df[col].replace('', None)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # table_data 형식으로 변환
        table_data = {}
        for idx, row in df.iterrows():
            day = int(row['Day']) if not pd.isna(row['Day']) else None
            if day:
                table_data[day] = {
                    '전날 오후 FK용량': row['전날 오후 FK용량'] if not pd.isna(row['전날 오후 FK용량']) else None,
                    '당일 오전 FK용량': row['당일 오전 FK용량'] if not pd.isna(row['당일 오전 FK용량']) else None,
                    'FK TDM': row['FK TDM'] if not pd.isna(row['FK TDM']) else None,
                    '당일 오후 FK용량': row['당일 오후 FK용량'] if '당일 오후 FK용량' in df.columns and not pd.isna(row['당일 오후 FK용량']) else None
                }
        
        return table_data
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return {}

def save_data_to_sheets(patient_id, day_index, prev_pm_dose, am_dose, fk_tdm, pm_prediction=None, am_prediction=None):
    """Google Sheets에 데이터 저장"""
    try:
        worksheet = get_or_create_worksheet(patient_id)
        if not worksheet:
            return False
        
        # 행 번호 계산 (헤더가 1행, 데이터는 2행부터)
        row_num = day_index + 1
        
        # 현재 날짜 데이터 업데이트
        worksheet.update(f'B{row_num}', [[prev_pm_dose if prev_pm_dose else '']])
        worksheet.update(f'C{row_num}', [[am_dose if am_dose else '']])
        worksheet.update(f'D{row_num}', [[fk_tdm if fk_tdm else '']])

        # 당일 오후 FK용량 (PM 예측값) 저장
        if pm_prediction is not None:
            worksheet.update(f'E{row_num}', [[pm_prediction]])

        # 다음날 예측값 저장 (day_index < 8인 경우)
        if pm_prediction is not None and am_prediction is not None and day_index < 8:
            next_row_num = day_index + 2
            worksheet.update(f'B{next_row_num}', [[pm_prediction]])  # 다음날 전날 오후 FK용량
            worksheet.update(f'C{next_row_num}', [[am_prediction]])  # 다음날 당일 오전 FK용량
        
        return True
    except Exception as e:
        st.error(f"데이터 저장 오류: {str(e)}")
        return False

def clear_patient_data_in_sheets(patient_id):
    """Google Sheets에서 환자 데이터 초기화"""
    try:
        worksheet = get_or_create_worksheet(patient_id)
        if not worksheet:
            return False
        
        # Day 2-9행의 B, C, D, E 컬럼 초기화 (헤더 제외)
        clear_data = [['', '', '', ''] for _ in range(8)]
        worksheet.update('B2:E9', clear_data)
        
        return True
    except Exception as e:
        st.error(f"데이터 초기화 오류: {str(e)}")
        return False

def round_prediction(value):
    """예측값을 0.5 단위로 반올림"""
    if value < 0:
        return 0
    else:
        rounded_value = round(value * 2) / 2
        return rounded_value

def parse_model_filename(filename):
    """모델 파일명에서 파라미터 추출"""
    # 예: am_lstm_static_ep100_bs32_lr0.001_hd64_tw1_seed42_nl2_msl23.pth
    # 또는: pm_gru_ep100_bs32_lr0.005_hd128_tw1_seed42_nl3_msl10.pth
    
    use_static = 'static' in filename
    
    # rnn_type 추출 (lstm 또는 gru)
    if '_lstm_' in filename or filename.startswith('am_lstm') or filename.startswith('pm_lstm'):
        rnn_type = 'lstm'
    elif '_gru_' in filename or filename.startswith('am_gru') or filename.startswith('pm_gru'):
        rnn_type = 'gru'
    else:
        rnn_type = 'lstm'  # 기본값
    
    # hidden_dim 추출 (hd64, hd128 등)
    hd_match = re.search(r'hd(\d+)', filename)
    hidden_dim = int(hd_match.group(1)) if hd_match else 64
    
    # num_layers 추출 (nl2 등)
    nl_match = re.search(r'nl(\d+)', filename)
    num_layers = int(nl_match.group(1)) if nl_match else 2
    
    # max_seq_len 추출 (msl23, msl10 등)
    msl_match = re.search(r'msl(\d+)', filename)
    max_seq_len = int(msl_match.group(1)) if msl_match else 10
    
    return {
        'use_static': use_static,
        'rnn_type': rnn_type,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len
    }

@st.cache_resource
def load_lstm_model(model_type, use_static, checkpoint_dir):
    """RNN 모델 로드 (LSTM 또는 GRU)"""
    try:
        # 모델 파일 찾기 - pm_ 또는 am_으로 시작하는 모든 .pth 파일 검색
        all_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        model_files = [f for f in all_files 
                      if f.startswith(f'{model_type}_lstm') or f.startswith(f'{model_type}_gru')]
        
        if not model_files:
            return None, None, f"{model_type.upper()} 모델 파일을 찾을 수 없습니다."
        
        # static 여부에 따라 필터링
        if use_static:
            model_files = [f for f in model_files if 'static' in f]
        else:
            model_files = [f for f in model_files if 'static' not in f]
        
        if not model_files:
            return None, None, f"{model_type.upper()} {'static' if use_static else 'non-static'} 모델 파일을 찾을 수 없습니다."
        
        model_file = model_files[0]  # 첫 번째 매칭 파일 사용
        model_path = os.path.join(checkpoint_dir, model_file)
        
        # 파라미터 추출
        params = parse_model_filename(model_file)
        rnn_type = params.get('rnn_type', 'lstm')  # 파일명에서 추출한 rnn_type 사용
        
        # 모델 구조 생성
        if model_type == 'pm':
            input_dim = 3  # prev_dose_pm, today_dose_am, today_tdm
        else:  # am
            input_dim = 4  # prev_dose_pm, today_dose_am, today_tdm, today_dose_pm
        
        # Static features: AGE, Sex, Bwt, Ht, BMI, Cause, HD_type, HD_duration, DM, HTN (총 10개)
        static_dim = 10 if use_static else 0
        
        model = RNNModel(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            use_static=use_static,
            static_dim=static_dim,
            rnn_type=rnn_type  # 파일명에서 추출한 rnn_type 사용
        )
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, params, None
        
    except Exception as e:
        return None, None, f"모델 로드 중 오류 발생: {str(e)}"

def prepare_sequence_for_prediction(patient_data, day_index, model_type, max_seq_len, pm_prediction=None):
    """예측을 위한 시퀀스 데이터 준비"""
    # 환자의 과거 데이터를 시퀀스로 준비
    sequence = []
    
    # day_index까지의 데이터 수집 (과거 데이터만)
    # 피처 순서: prev_dose_pm, today_dose_am, today_tdm (PM 모델)
    #           prev_dose_pm, today_dose_am, today_tdm, today_dose_pm (AM 모델)
    for day in range(1, day_index):
        day_data = patient_data.get(day, {})
        
        # prev_dose_pm: 전날 오후 용량 (각 day의 '전날 오후 FK용량' 필드에서 가져옴)
        prev_dose_pm = day_data.get('전날 오후 FK용량', 0) or 0
        # today_dose_am: 당일 오전 용량
        today_dose_am = day_data.get('당일 오전 FK용량', 0) or 0
        # today_tdm: 당일 TDM
        today_tdm = day_data.get('FK TDM', 0) or 0
        
        # 과거 데이터가 있으면 추가 (순서 유지: prev_dose_pm, today_dose_am, today_tdm)
        if today_dose_am > 0 or today_tdm > 0:
            if model_type == 'pm':
                # PM 모델: 3개 피처 (prev_dose_pm, today_dose_am, today_tdm)
                sequence.append([prev_dose_pm, today_dose_am, today_tdm])
            else:  # am
                # AM 모델: 4개 피처 (prev_dose_pm, today_dose_am, today_tdm, today_dose_pm)
                today_dose_pm = day_data.get('당일 오후 FK용량', 0) or 0
                sequence.append([prev_dose_pm, today_dose_am, today_tdm, today_dose_pm])
    
    # 예측할 날짜의 데이터 추가
    day_data = patient_data.get(day_index, {})
    # prev_dose_pm: 전날 오후 용량 (각 day의 '전날 오후 FK용량' 필드에서 가져옴)
    prev_dose_pm = day_data.get('전날 오후 FK용량', 0) or 0
    # today_dose_am: 당일 오전 용량
    today_dose_am = day_data.get('당일 오전 FK용량', 0) or 0
    # today_tdm: 당일 TDM
    today_tdm = day_data.get('FK TDM', 0) or 0
    
    if model_type == 'pm':
        # PM 모델: 3개 피처 (prev_dose_pm, today_dose_am, today_tdm)
        sequence.append([prev_dose_pm, today_dose_am, today_tdm])
    else:  # am
        # AM 모델: 4개 피처 (prev_dose_pm, today_dose_am, today_tdm, today_dose_pm)
        # PM 예측값을 today_dose_pm으로 사용
        today_dose_pm = pm_prediction if pm_prediction is not None else day_data.get('당일 오후 FK용량', 0)
        sequence.append([prev_dose_pm, today_dose_am, today_tdm, today_dose_pm])
    
    if not sequence:
        return None, None
    
    # numpy array로 변환
    sequence_array = np.array(sequence, dtype=np.float32)
    seq_len = len(sequence_array)
    
    # 패딩 (max_seq_len까지)
    if seq_len < max_seq_len:
        padding = np.zeros((max_seq_len - seq_len, sequence_array.shape[1]), dtype=np.float32)
        sequence_array = np.vstack([sequence_array, padding])
    
    # (1, max_seq_len, input_dim) 형태로 변환
    sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(device)
    # pack_padded_sequence는 lengths가 CPU tensor여야 함
    seq_length = torch.LongTensor([seq_len]).cpu()
    
    return sequence_tensor, seq_length

def load_or_create_patient_data(patient_id, patient_name, patient_sex=None):
    """환자 데이터를 로드하거나 새로 생성 (Google Sheets)"""
    table_data = load_patient_data_from_sheets(patient_id)
    
    # 데이터가 없으면 초기화된 딕셔너리 반환
    if not table_data:
        days = list(range(1, 9))
        table_data = {day: {'전날 오후 FK용량': None, '당일 오전 FK용량': None, 'FK TDM': None, '당일 오후 FK용량': None} for day in days}
    
    return table_data

def predict_dose(patient_id, day_index, previous_evening_dose, current_morning_dose, current_fk_tdm, 
                 static_features=None, patient_data=None):
    """용량 예측 수행 (LSTM 모델 사용)"""
    try:
        # 환자 정보 입력 여부 확인 (모든 static feature가 입력되었는지)
        use_static = static_features is not None and len(static_features) == 10
        
        # 모델 로드
        pm_model, pm_params, error = load_lstm_model('pm', use_static, CHECKPOINT_DIR)
        if error:
            return None, None, error
        
        am_model, am_params, error = load_lstm_model('am', use_static, CHECKPOINT_DIR)
        if error:
            return None, None, error
        
        # PM 모델 예측을 위한 시퀀스 준비
        pm_sequence, pm_seq_len = prepare_sequence_for_prediction(
            patient_data, day_index, 'pm', pm_params['max_seq_len']
        )
        
        if pm_sequence is None:
            return None, None, "시퀀스 데이터를 준비할 수 없습니다."
        
        # Static feature 준비 (10개: AGE, Sex, Bwt, Ht, BMI, Cause, HD_type, HD_duration, DM, HTN)
        static_feature = None
        if use_static:
            # static_features는 딕셔너리 형태로 전달됨
            static_values = [
                static_features.get('AGE', 0.0),
                static_features.get('Sex', 0.0),  # 0: 남성, 1: 여성
                static_features.get('Bwt', 0.0),
                static_features.get('Ht', 0.0),
                static_features.get('BMI', 0.0),
                static_features.get('Cause', 0.0),
                static_features.get('HD_type', 0.0),
                static_features.get('HD_duration', 0.0),
                static_features.get('DM', 0.0),
                static_features.get('HTN', 0.0)
            ]
            static_feature = torch.FloatTensor([static_values]).to(device)
        
        # PM 모델 예측: 입력 [prev_dose_pm, today_dose_am, today_tdm] → 출력 today_dose_pm (당일 오후 용량)
        with torch.no_grad():
            pm_prediction = pm_model(pm_sequence, lengths=pm_seq_len, static_features=static_feature)
            pm_prediction = pm_prediction.cpu().item()
        
        pm_prediction = round_prediction(pm_prediction)
        # pm_prediction은 당일 오후 FK용량 (today_dose_pm)
        
        # AM 모델 예측을 위한 시퀀스 준비 (PM 예측값 포함)
        # AM 모델 입력: [prev_dose_pm, today_dose_am, today_tdm, today_dose_pm]
        am_sequence, am_seq_len = prepare_sequence_for_prediction(
            patient_data, day_index, 'am', am_params['max_seq_len'], pm_prediction=pm_prediction
        )
        
        if am_sequence is None:
            return None, None, "AM 모델 시퀀스 데이터를 준비할 수 없습니다."
        
        # AM 모델 예측: 입력 [prev_dose_pm, today_dose_am, today_tdm, today_dose_pm] → 출력 next_dose_am (다음날 오전 용량)
        with torch.no_grad():
            am_prediction = am_model(am_sequence, lengths=am_seq_len, static_features=static_feature)
            am_prediction = am_prediction.cpu().item()
        
        am_prediction = round_prediction(am_prediction)
        # am_prediction은 다음날 오전 FK용량 (next_dose_am)
        
        # TDM 기반 방향성 강제
        # TDM < 7: 전날 대비 최소 0.5mg 증량
        # TDM 7~7.5: 감량 금지 (유지 이상)
        # TDM 7.5~9: 모델 예측 그대로 (제약 없음)
        # TDM >= 9: 전날 대비 최소 0.5mg 감량
        if current_fk_tdm < 7:
            pm_prediction = max(pm_prediction, previous_evening_dose + 0.5)
            am_prediction = max(am_prediction, current_morning_dose + 0.5)
        elif current_fk_tdm < 7.5:
            pm_prediction = max(pm_prediction, previous_evening_dose)
            am_prediction = max(am_prediction, current_morning_dose)
        elif current_fk_tdm >= 9:
            pm_prediction = min(pm_prediction, previous_evening_dose - 0.5)
            am_prediction = min(am_prediction, current_morning_dose - 0.5)

        # 저용량 보정 (0~4일차)
        if day_index <= 4:
            if previous_evening_dose <= 1 or current_morning_dose <= 1:
                pm_prediction += 0.5
                am_prediction += 0.5

        # 1일차 PM 감산
        if day_index == 1:
            pm_prediction -= 0.5
        
        # Google Sheets에 데이터 저장
        save_success = save_data_to_sheets(
            patient_id, 
            day_index, 
            previous_evening_dose, 
            current_morning_dose, 
            current_fk_tdm,
            pm_prediction, 
            am_prediction
        )
        
        if not save_success:
            return pm_prediction, am_prediction, "데이터 저장 중 오류 발생 (예측은 완료됨)"
        
        return pm_prediction, am_prediction, None
        
    except Exception as e:
        return None, None, f"예측 중 오류 발생: {str(e)}"

# 메인 앱
def main():
    # Google Sheets 연결 상태 확인
    try:
        client = get_gspread_client()
        if client:
            # 연결 성공 - 조용히 진행
            pass
        else:
            st.error("⚠️ Google Sheets 연결 실패. 데이터 저장이 불가능합니다.")
            st.info("💡 로컬 개발: `service_account.json` 파일 확인\n\n💡 Streamlit Cloud: Secrets 설정 확인")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Google Sheets 초기화 오류: {str(e)}")
        st.stop()
    
    # 헤더
    # 로고 이미지를 base64로 인코딩
    logo_path = os.path.join(os.path.dirname(__file__), "static", "mark.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 32px; vertical-align: middle; margin-right: 15px;">'
    else:
        logo_html = '🏥'
    
    st.markdown(f"""
    <div style="background-color: #00274d; color: white; padding: 15px 20px; border-radius: 5px; margin-bottom: 30px; display: flex; align-items: center;">
        {logo_html}
        <span style="font-size: 28px; font-weight: bold;">ASAN MEDICAL</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = ''
    if 'patient_name' not in st.session_state:
        st.session_state.patient_name = ''
    if 'static_features' not in st.session_state:
        st.session_state.static_features = {}
    if 'table_data' not in st.session_state:
        st.session_state.table_data = None
    if 'patient_data_loaded' not in st.session_state:
        st.session_state.patient_data_loaded = False
    
    # 환자 정보 입력 섹션
    st.subheader("Patient information")
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("Patient ID", value=st.session_state.patient_id, key="input_patient_id")
    with col2:
        patient_name = st.text_input("Patient Name", value=st.session_state.patient_name, key="input_patient_name")
    
    # Static Features 입력 섹션
    st.markdown("### Patient Features (Optional)")
    st.caption(
        "If you provide all patient features, a patient-specific time-series model is used; "
        "otherwise a general time-series model is used."
    )
    
    # 첫 번째 줄: 5개 컬럼
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=150, value=int(st.session_state.static_features.get('AGE', 0)) if st.session_state.static_features.get('AGE') else None, key="input_age", step=1)
    
    with col2:
        sex = st.selectbox("Sex", ["", "Male", "Female"], 
                          index=0 if not st.session_state.static_features.get('Sex') else 
                          (1 if st.session_state.static_features.get('Sex') == 0 else 2),
                          key="input_sex")
    
    with col3:
        bwt = st.number_input("Body Weight (kg)", min_value=0.0, value=float(st.session_state.static_features.get('Bwt', 0)) if st.session_state.static_features.get('Bwt') else None, key="input_bwt", step=0.1, format="%.1f")
    
    with col4:
        ht = st.number_input("Height (cm)", min_value=0.0, value=float(st.session_state.static_features.get('Ht', 0)) if st.session_state.static_features.get('Ht') else None, key="input_ht", step=0.1, format="%.1f")
    
    with col5:
        bmi = st.number_input("BMI", min_value=0.0, value=float(st.session_state.static_features.get('BMI', 0)) if st.session_state.static_features.get('BMI') else None, key="input_bmi", step=0.1, format="%.1f")
    
    # 두 번째 줄: 5개 컬럼
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        cause_options = ["", "HTN", "DM", "GN", "IgA", "FSGS", "PCKD", "Unknown", "etc."]
        cause_values = [None, 0, 1, 2, 3, 4, 5, 6, 7]
        cause_index = 0
        if st.session_state.static_features.get('Cause') is not None:
            cause_val = int(st.session_state.static_features.get('Cause'))
            if cause_val in cause_values:
                cause_index = cause_values.index(cause_val)
        cause = st.selectbox("Cause", cause_options, index=cause_index, key="input_cause")
    
    with col7:
        hd_type_options = ["", "Preemptive", "HD", "CAPD", "HD+PD"]
        hd_type_values = [None, 0, 1, 2, 3]
        hd_type_index = 0
        if st.session_state.static_features.get('HD_type') is not None:
            hd_type_val = int(st.session_state.static_features.get('HD_type'))
            if hd_type_val in hd_type_values:
                hd_type_index = hd_type_values.index(hd_type_val)
        hd_type = st.selectbox("Hemodialysis Type", hd_type_options, index=hd_type_index, key="input_hd_type")
    
    with col8:
        hd_duration = st.number_input("Hemodialysis Duration (months)", min_value=0.0, value=float(st.session_state.static_features.get('HD_duration', 0)) if st.session_state.static_features.get('HD_duration') else None, key="input_hd_duration", step=0.1, format="%.1f")
    
    with col9:
        dm_options = ["", "No", "Yes"]
        dm_values = [None, 0, 1]
        dm_index = 0
        if st.session_state.static_features.get('DM') is not None:
            dm_val = int(st.session_state.static_features.get('DM'))
            if dm_val in dm_values:
                dm_index = dm_values.index(dm_val)
        dm = st.selectbox("Diabetes Mellitus", dm_options, index=dm_index, key="input_dm")
    
    with col10:
        htn_options = ["", "No", "Yes"]
        htn_values = [None, 0, 1]
        htn_index = 0
        if st.session_state.static_features.get('HTN') is not None:
            htn_val = int(st.session_state.static_features.get('HTN'))
            if htn_val in htn_values:
                htn_index = htn_values.index(htn_val)
        htn = st.selectbox("Hypertension", htn_options, index=htn_index, key="input_htn")
    
    # Static features 수집
    static_features_dict = {}
    if age is not None:
        static_features_dict['AGE'] = float(age)
    if sex:
        static_features_dict['Sex'] = 1.0 if sex == "Female" else 0.0
    if bwt is not None:
        static_features_dict['Bwt'] = float(bwt)
    if ht is not None:
        static_features_dict['Ht'] = float(ht)
    if bmi is not None:
        static_features_dict['BMI'] = float(bmi)
    if cause and cause != "":
        cause_val = cause_values[cause_options.index(cause)]
        if cause_val is not None:
            static_features_dict['Cause'] = float(cause_val)
    if hd_type and hd_type != "":
        hd_type_val = hd_type_values[hd_type_options.index(hd_type)]
        if hd_type_val is not None:
            static_features_dict['HD_type'] = float(hd_type_val)
    if hd_duration is not None:
        static_features_dict['HD_duration'] = float(hd_duration)
    if dm and dm != "":
        dm_val = dm_values[dm_options.index(dm)]
        if dm_val is not None:
            static_features_dict['DM'] = float(dm_val)
    if htn and htn != "":
        htn_val = htn_values[htn_options.index(htn)]
        if htn_val is not None:
            static_features_dict['HTN'] = float(htn_val)
    
    # 모델 선택 정보 표시
    use_static = len(static_features_dict) == 10
    if use_static:
        st.success(
            "✅ All patient features are provided: using a patient-specific time-series model."
        )
    elif len(static_features_dict) > 0:
        st.warning(
            f"⚠️ Only {len(static_features_dict)}/10 patient features are provided: using a general time-series model."
        )
    else:
        st.info("ℹ️ No patient features provided.")
    
    if st.button("Confirm", type="primary", use_container_width=True):
        if not patient_id or not patient_name:
            st.error("Please enter patient ID and name.")
        else:
            st.session_state.patient_id = patient_id
            st.session_state.patient_name = patient_name
            st.session_state.static_features = static_features_dict if use_static else {}
            
            # 환자 데이터 로드
            with st.spinner("Loading patient data..."):
                table_data = load_or_create_patient_data(patient_id, patient_name, "")
                st.session_state.table_data = table_data
                st.session_state.patient_data_loaded = True
            
            static_info = (
                f"Patient Features: {len(static_features_dict)}/10"
                if static_features_dict
                else "Patient Features: None"
            )
            st.success(
                f"✅ Patient ID: {patient_id} / Patient name: {patient_name} / {static_info}"
            )
            st.rerun()
    
    # 환자 정보가 입력된 경우 테이블 표시
    if st.session_state.patient_id and st.session_state.table_data is not None:
        st.markdown("---")
        st.subheader("FK dose prediction")
        
        # 환자 정보 표시
        use_static = len(st.session_state.static_features) == 10
        model_type_display = (
            "Patient-specific time-series model"
            if use_static
            else "General time-series model"
        )
        
        col_info, col_flush = st.columns([4, 1])
        with col_info:
            st.info(
                f"**Current patient**: {st.session_state.patient_id} - {st.session_state.patient_name} "
                f"| **Model**: {model_type_display}"
            )
        with col_flush:
            if st.button("🗑️ Reset data", type="secondary", use_container_width=True):
                # 모든 데이터 초기화
                days_to_reset = list(range(1, 9))
                for day in days_to_reset:
                    st.session_state.table_data[day] = {
                        '전날 오후 FK용량': None,
                        '당일 오전 FK용량': None,
                        'FK TDM': None,
                        '당일 오후 FK용량': None  # 예측 결과 저장용 (Excel에는 저장 안 함)
                    }
                    # 예측 결과도 초기화
                    if f"predicted_pm_{day}" in st.session_state:
                        del st.session_state[f"predicted_pm_{day}"]
                    if f"predicted_am_{day}" in st.session_state:
                        del st.session_state[f"predicted_am_{day}"]
                    # 입력 칸의 세션 상태도 초기화 (number_input의 key 값)
                    st.session_state[f"prev_pm_{day}"] = 0.0
                    st.session_state[f"am_{day}"] = 0.0
                    st.session_state[f"tdm_{day}"] = 0.0
                
                # Google Sheets도 초기화
                clear_success = clear_patient_data_in_sheets(st.session_state.patient_id)
                if not clear_success:
                    st.warning("⚠️ Failed to reset data in Google Sheets.")
                
                st.success("✅ All data have been reset.")
                st.rerun()
        
        days = list(range(1, 9))
        
        # 테이블 형태로 데이터 표시 및 입력
        for day in days:
            day_data = st.session_state.table_data.get(day, {})
            day_label = f"Day {day}"
            
            with st.expander(f"{day_label}", expanded=(day <= 3)):
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1.5])
                
                with col1:
                    st.write("**전날 오후 FK용량**")
                    prev_pm_key = f"prev_pm_{day}"
                    # 세션 상태 우선, 없으면 table_data에서 가져오기
                    if prev_pm_key not in st.session_state:
                        prev_pm_default = day_data.get('전날 오후 FK용량')
                        st.session_state[prev_pm_key] = float(prev_pm_default) if prev_pm_default is not None else 0.0
                    
                    prev_pm_value = st.number_input(
                        "Previous PM dose (mg)",
                        min_value=0.0,
                        step=0.05,
                        key=prev_pm_key,
                        format="%.2f"
                    )
                
                with col2:
                    st.write("**당일 오전 FK용량**")
                    am_key = f"am_{day}"
                    # 세션 상태 우선, 없으면 table_data에서 가져오기
                    if am_key not in st.session_state:
                        am_default = float(day_data.get('당일 오전 FK용량', 0)) if day_data.get('당일 오전 FK용량') is not None else 0.0
                        st.session_state[am_key] = am_default
                    
                    am_value = st.number_input(
                        "Today AM dose (mg)",
                        min_value=0.0,
                        step=0.05,
                        key=am_key,
                        format="%.2f"
                    )
                
                with col3:
                    st.write("**FK TDM**")
                    tdm_key = f"tdm_{day}"
                    # 세션 상태 우선, 없으면 table_data에서 가져오기
                    if tdm_key not in st.session_state:
                        tdm_default = float(day_data.get('FK TDM', 0)) if day_data.get('FK TDM') is not None else 0.0
                        st.session_state[tdm_key] = tdm_default
                    
                    tdm_value = st.number_input(
                        "FK TDM level",
                        min_value=0.0,
                        step=0.1,
                        key=tdm_key,
                        format="%.1f"
                    )
                    # TDM 값이 변경되면 세션 상태 업데이트 (Google Sheets는 예측 시 저장됨)
                    if tdm_value > 0 and tdm_value != day_data.get('FK TDM', 0):
                        st.session_state.table_data[day]['FK TDM'] = tdm_value
                
                with col4:
                    st.write("**Prediction results**")
                    if f"predicted_pm_{day}" in st.session_state:
                        st.success(
                            f"Today PM: **{st.session_state[f'predicted_pm_{day}']:.2f}** mg"
                        )
                    else:
                        st.write("_No prediction yet_")
                    
                    if f"predicted_am_{day+1}" in st.session_state and day < 8:
                        st.success(
                            f"Next day AM: **{st.session_state[f'predicted_am_{day+1}']:.2f}** mg"
                        )
                
                with col5:
                    st.write("**Run prediction**")
                    if st.button("🔮 Predict", key=f"predict_btn_{day}", use_container_width=True):
                        # 전날 오후 용량 (입력된 값 사용)
                        previous_evening_dose = prev_pm_value
                        
                        # 당일 오전 용량
                        current_morning_dose = am_value
                        
                        # 당일 FK TDM
                        current_fk_tdm = tdm_value
                        
                        # 입력값 검증 (전날 오후 용량, 당일 오전 용량, FK TDM 모두 필수)
                        if previous_evening_dose == 0 or current_morning_dose == 0 or current_fk_tdm == 0:
                            st.error(
                                "Please enter previous PM dose, today AM dose, and FK TDM."
                            )
                            st.stop()

                        # 이전 Day 예측 완료 여부 검증
                        if day > 1:
                            missing_days = []
                            for prev_day in range(1, day):
                                prev_data = st.session_state.table_data.get(prev_day, {})
                                if prev_data.get('당일 오후 FK용량') is None:
                                    missing_days.append(prev_day)
                            if missing_days:
                                missing_str = ", ".join([f"Day {d}" for d in missing_days])
                                st.error(f"Please complete prediction for {missing_str} first.")
                                st.stop()

                        # 입력값 저장 (예측 전에 현재 입력값 저장)
                        st.session_state.table_data[day]['전날 오후 FK용량'] = previous_evening_dose
                        st.session_state.table_data[day]['당일 오전 FK용량'] = current_morning_dose
                        
                        # 예측 수행
                        with st.spinner("Predicting..."):
                            pm_pred, am_pred, error = predict_dose(
                                st.session_state.patient_id,
                                day,
                                previous_evening_dose,
                                current_morning_dose,
                                current_fk_tdm,
                                st.session_state.static_features if len(st.session_state.static_features) == 10 else None,
                                st.session_state.table_data
                            )
                        
                        if error:
                            st.error(f"❌ Prediction error: {error}")
                        else:
                            # 예측 결과 저장 및 표시
                            # pm_pred: 당일 오후 FK용량 (today_dose_pm) → day의 '당일 오후 FK용량'에 저장
                            st.session_state[f"predicted_pm_{day}"] = pm_pred
                            st.session_state.table_data[day]['당일 오후 FK용량'] = pm_pred
                            
                            # am_pred: 다음날 오전 FK용량 (next_dose_am) → day+1의 '당일 오전 FK용량'에 저장
                            # pm_pred: 다음날 전날 오후 FK용량 → day+1의 '전날 오후 FK용량'에 저장
                            if day < 8:
                                st.session_state[f"predicted_am_{day+1}"] = am_pred
                                st.session_state.table_data[day+1]['당일 오전 FK용량'] = am_pred
                                st.session_state.table_data[day+1]['전날 오후 FK용량'] = pm_pred
                                # 입력 칸의 세션 상태도 업데이트 (number_input의 key 값)
                                st.session_state[f"prev_pm_{day+1}"] = pm_pred
                                st.session_state[f"am_{day+1}"] = am_pred
                            
                            st.success("✅ Prediction completed!")
                            st.rerun()
        
        # 전체 데이터 요약 테이블
        st.markdown("---")
        st.subheader("📊 Summary table")
        
        summary_data = []
        for day in days:
            day_data = st.session_state.table_data.get(day, {})
            # 전날 오후 FK용량 가져오기 (각 day의 '전날 오후 FK용량' 필드에서)
            prev_pm_dose = day_data.get('전날 오후 FK용량', '')
            
            summary_data.append({
                'Day': f"Day {day}",
                '전날 오후 FK용량 (Previous PM FK dose)': prev_pm_dose if prev_pm_dose is not None else '',
                '당일 오전 FK용량 (Today AM FK dose)': day_data.get('당일 오전 FK용량', ''),
                'FK TDM (FK trough level)': day_data.get('FK TDM', '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()