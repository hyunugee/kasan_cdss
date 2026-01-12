# Streamlit FK 레벨 추적기

신장이식 환자의 Tacrolimus (FK) 투여량 및 TDM 레벨을 추적하고 예측하는 Streamlit 웹 애플리케이션입니다.

## 기능

- 환자 정보 등록 및 관리
- FK TDM 레벨 입력 및 저장
- FK 용량 (오전/오후) 입력 및 관리
- PyTorch LSTM 모델을 사용한 용량 예측
  - 오후 용량 예측 (PM 모델)
  - 다음날 오전 용량 예측 (AM 모델)
- **Google Sheets를 통한 데이터 영구 저장 및 다중 기기 접근**
  - 여러 환자 데이터 동시 관리
  - 화면을 나갔다 들어와도 데이터 유지
  - 병원 직원 간 데이터 공유 가능

## 빠른 시작

### 1. Google Sheets 설정
먼저 Google Sheets 연동을 설정해야 합니다. 자세한 가이드는 [GOOGLE_SHEETS_SETUP.md](./GOOGLE_SHEETS_SETUP.md)를 참조하세요.

**요약:**
1. Google Cloud Console에서 프로젝트 생성 및 API 활성화
2. 서비스 계정 생성 및 JSON 키 다운로드
3. Google Sheets 생성 (`tacrolimus_tdm_data`)
4. 서비스 계정과 공유

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 인증 설정

**로컬 개발:**
```bash
# 다운로드한 서비스 계정 JSON 파일을 복사
cp ~/Downloads/your-service-account-key.json streamlit_app/service_account.json
```

**Streamlit Cloud:**
- Streamlit Cloud의 Secrets 설정에 서비스 계정 정보 입력
- 자세한 내용은 [GOOGLE_SHEETS_SETUP.md](./GOOGLE_SHEETS_SETUP.md) 참조

## 모델 파일 준비

애플리케이션을 실행하기 전에 다음 PyTorch LSTM 모델 파일들이 `checkpoints/` 디렉토리에 있어야 합니다:

### 환자 정보 입력 시 사용 (Static 모델):
- `am_lstm_static_*.pth` (AM 모델 - 환자 정보 포함)
- `pm_lstm_static_*.pth` (PM 모델 - 환자 정보 포함)

### 환자 정보 미입력 시 사용 (일반 모델):
- `am_lstm_*.pth` (AM 모델 - 환자 정보 없음)
- `pm_lstm_*.pth` (PM 모델 - 환자 정보 없음)

**참고**: 모델 파일명에서 파라미터(hidden_dim, num_layers, max_seq_len 등)를 자동으로 추출합니다.

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용할 수 있습니다.

## 사용 방법

1. **환자 정보 입력**
   - 환자 ID, 이름, 성별을 입력하고 "확인" 버튼을 클릭합니다.
   - 기존 환자 데이터가 있으면 자동으로 로드됩니다.

2. **데이터 입력**
   - 각 일차별로 FK TDM 값과 FK 용량 (오전/오후)을 입력합니다.
   - TDM 값은 자동으로 Excel 파일에 저장됩니다.

3. **예측 실행**
   - 각 일차의 "예측" 버튼을 클릭하여 용량을 예측합니다.
   - 예측에 필요한 데이터:
     - 전날 오후 용량 (1일차 제외)
     - 당일 오전 용량
     - 당일 FK TDM 값
   - 예측 결과는 자동으로 Excel 파일에 저장됩니다.

## 데이터 저장

모든 데이터는 **Google Sheets**에 실시간으로 저장됩니다:
- 스프레드시트 이름: `tacrolimus_tdm_data`
- 각 환자는 별도의 시트(워크시트)로 관리
- 자동 백업 및 버전 관리
- 여러 기기에서 동시 접근 가능
- 병원 직원 간 데이터 공유 가능

**데이터 구조:**
```
| Day | 전날 오후 FK용량 | 당일 오전 FK용량 | FK TDM |
|-----|----------------|----------------|--------|
| 1   | 2.0            | 2.0            | 5.5    |
| 2   | 2.5            | 2.5            | 7.2    |
```

## 모델 선택 로직

- **모든 Static Features 입력 시**: 다음 10개의 Static Features가 모두 입력되면 `*_lstm_static_*.pth` 모델을 사용합니다.
  - AGE (나이)
  - Sex (성별: 남성=0, 여성=1)
  - Bwt (체중, kg)
  - Ht (키, cm)
  - BMI
  - Cause
  - HD_type
  - HD_duration
  - DM (0 또는 1)
  - HTN (0 또는 1)
- **Static Features 미입력 또는 일부만 입력 시**: 10개가 모두 입력되지 않으면 `*_lstm_*.pth` (static 없는) 모델을 사용합니다.

## 주의사항

- 모델 파일 경로는 `app.py`의 `CHECKPOINT_DIR` 변수에서 설정할 수 있습니다.
- Google Sheets 이름은 `app.py`의 `SPREADSHEET_NAME` 변수에서 설정할 수 있습니다 (기본값: `tacrolimus_tdm_data`).
- 예측 모델은 PyTorch LSTM 모델(.pth 파일)이어야 합니다.
- 모델 파일명에서 파라미터를 자동으로 추출하므로, 파일명 형식을 유지해야 합니다.
- `service_account.json` 파일은 Git에 커밋하지 마세요 (이미 `.gitignore`에 추가됨).

## 보안

- Google Sheets 서비스 계정 키는 안전하게 보관하세요
- Streamlit Cloud Secrets는 암호화되어 저장됩니다
- Google Sheets 접근 권한을 필요한 사람에게만 부여하세요

## 문제 해결

문제가 발생하면 [GOOGLE_SHEETS_SETUP.md](./GOOGLE_SHEETS_SETUP.md)의 "문제 해결" 섹션을 참조하세요.

### 주요 확인 사항:
1. Google Sheets API 활성화 확인
2. 서비스 계정 JSON 파일 위치 확인
3. Google Sheets에 서비스 계정 공유 확인
4. 스프레드시트 이름 정확성 확인

## 배포

### Streamlit Cloud
1. GitHub에 푸시 (service_account.json 제외)
2. Streamlit Cloud에서 앱 생성
3. Secrets 설정에 서비스 계정 정보 입력
4. 배포

자세한 내용은 [GOOGLE_SHEETS_SETUP.md](./GOOGLE_SHEETS_SETUP.md)를 참조하세요.

