# Google Sheets 연동 설정 가이드

이 가이드는 Tacrolimus 용량 예측 앱을 Google Sheets와 연동하는 방법을 설명합니다.

## 📋 목차
1. [Google Cloud 설정](#1-google-cloud-설정)
2. [Google Sheets 생성](#2-google-sheets-생성)
3. [로컬 개발 환경 설정](#3-로컬-개발-환경-설정)
4. [Streamlit Cloud 배포](#4-streamlit-cloud-배포)
5. [테스트](#5-테스트)

---

## 1. Google Cloud 설정

### 1.1 프로젝트 생성
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. **새 프로젝트** 생성
   - 프로젝트 이름: `tacrolimus-tdm` (또는 원하는 이름)
   - 프로젝트 생성 후 선택

### 1.2 Google Sheets API 활성화
1. 왼쪽 메뉴 → **API 및 서비스** → **라이브러리**
2. "Google Sheets API" 검색
3. **사용 설정** 클릭

### 1.3 Google Drive API 활성화
1. "Google Drive API" 검색
2. **사용 설정** 클릭

### 1.4 서비스 계정 생성
1. 왼쪽 메뉴 → **IAM 및 관리자** → **서비스 계정**
2. **서비스 계정 만들기** 클릭
3. 서비스 계정 세부정보:
   - 이름: `streamlit-app`
   - 설명: `Streamlit app for TDM data management`
4. **만들기 및 계속하기** 클릭
5. 역할 선택: **편집자** (또는 **소유자**)
6. **완료** 클릭

### 1.5 서비스 계정 키 생성
1. 생성된 서비스 계정 클릭
2. 상단 **키** 탭 선택
3. **키 추가** → **새 키 만들기**
4. 키 유형: **JSON** 선택
5. **만들기** 클릭 → JSON 파일 자동 다운로드
   - ⚠️ **중요**: 이 파일을 안전하게 보관하세요!
   - 파일명: `service_account.json`

---

## 2. Google Sheets 생성

### 2.1 스프레드시트 생성
1. [Google Sheets](https://sheets.google.com/) 접속
2. **새 스프레드시트 만들기**
3. 이름: `tacrolimus_tdm_data`
   - ⚠️ **주의**: 이름이 정확히 일치해야 합니다!

### 2.2 서비스 계정과 공유
1. 스프레드시트 우측 상단 **공유** 클릭
2. 다운로드한 JSON 파일에서 `client_email` 복사
   - 예: `streamlit-app@tacrolimus-tdm.iam.gserviceaccount.com`
3. 해당 이메일 입력 후 **편집자** 권한 부여
4. **완료** 클릭

---

## 3. 로컬 개발 환경 설정

### 3.1 서비스 계정 파일 배치
다운로드한 JSON 파일을 `streamlit_app/` 디렉토리에 `service_account.json`으로 저장:

```bash
cd streamlit_app/
# JSON 파일을 service_account.json으로 복사
cp ~/Downloads/your-project-xxxxx.json service_account.json
```

### 3.2 .gitignore 확인
`.gitignore`에 다음이 포함되어 있는지 확인:

```
# Google Sheets credentials
streamlit_app/service_account.json
streamlit_app/.streamlit/secrets.toml
```

### 3.3 Streamlit secrets 설정 (선택사항)
로컬에서 secrets를 사용하려면:

1. `.streamlit/secrets.toml` 파일 생성:
```bash
cd streamlit_app/
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. `service_account.json` 내용을 `secrets.toml`에 복사:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
# ... (JSON 파일의 모든 필드)
```

### 3.4 패키지 설치
```bash
pip install -r requirements.txt
```

### 3.5 앱 실행
```bash
cd streamlit_app/
streamlit run app.py
```

---

## 4. Streamlit Cloud 배포

### 4.1 GitHub에 푸시
```bash
git add .
git commit -m "Add Google Sheets integration"
git push origin main
```

⚠️ **주의**: `service_account.json` 파일이 푸시되지 않았는지 확인!

### 4.2 Streamlit Cloud 설정
1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. **New app** 클릭
3. GitHub repository 선택
4. Main file path: `streamlit_app/app.py`
5. **Advanced settings** 클릭

### 4.3 Secrets 설정
**Secrets** 섹션에 서비스 계정 JSON 내용 입력:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
```

⚠️ **주의**: 
- `private_key`의 줄바꿈은 `\n`으로 표현
- 모든 필드는 큰따옴표로 감싸기

### 4.4 배포
**Deploy!** 클릭 → 앱이 자동으로 빌드 및 배포됩니다.

---

## 5. 테스트

### 5.1 환자 등록 테스트
1. 앱 접속
2. 사이드바에서 환자 정보 입력
   - 환자 ID: `TEST001`
   - 이름: `테스트환자`
3. **등록** 클릭

### 5.2 Google Sheets 확인
1. `tacrolimus_tdm_data` 스프레드시트 열기
2. 하단에 `TEST001` 시트가 생성되었는지 확인
3. 헤더: `Day`, `전날 오후 FK용량`, `당일 오전 FK용량`, `FK TDM`

### 5.3 데이터 입력 및 예측
1. Day 1에 데이터 입력:
   - 전날 오후 FK용량: `2.0`
   - 당일 오전 FK용량: `2.0`
   - FK TDM: `5.5`
2. **예측** 버튼 클릭
3. Google Sheets에 자동 저장 확인

### 5.4 데이터 지속성 테스트
1. 브라우저 새로고침 또는 다른 기기에서 접속
2. `TEST001` 환자 선택
3. 이전에 입력한 데이터가 로드되는지 확인

---

## 🔒 보안 권장사항

1. **서비스 계정 키 보안**
   - JSON 파일을 Git에 절대 커밋하지 마세요
   - 파일 권한: `chmod 600 service_account.json`

2. **Google Sheets 접근 제한**
   - 필요한 사람에게만 공유
   - 서비스 계정 외에는 **보기 전용** 권장

3. **Streamlit Cloud Secrets**
   - Secrets는 암호화되어 저장됨
   - Team members만 접근 가능

---

## 🐛 문제 해결

### 문제 1: "Google Sheets 연결 오류"
**원인**: 서비스 계정 인증 실패

**해결**:
1. `service_account.json` 파일이 올바른 위치에 있는지 확인
2. Streamlit Cloud에서 Secrets가 올바르게 설정되었는지 확인
3. JSON 형식이 올바른지 확인 (줄바꿈 `\n` 확인)

### 문제 2: "스프레드시트 접근 오류"
**원인**: 서비스 계정이 스프레드시트에 접근 권한이 없음

**해결**:
1. Google Sheets에서 서비스 계정 이메일과 공유했는지 확인
2. **편집자** 권한이 부여되었는지 확인

### 문제 3: "WorksheetNotFound"
**원인**: 스프레드시트 이름이 일치하지 않음

**해결**:
1. 스프레드시트 이름이 정확히 `tacrolimus_tdm_data`인지 확인
2. `app.py`의 `SPREADSHEET_NAME` 변수 확인

### 문제 4: API 할당량 초과
**원인**: 무료 할당량 초과 (분당 300회)

**해결**:
1. 할당량은 자동으로 리셋됨 (1분 대기)
2. 캐싱 추가 고려

---

## 📊 데이터 구조

각 환자는 별도의 시트로 관리됩니다:

### 시트 이름
- 환자 ID (예: `20250101`, `TEST001`)

### 컬럼 구조
| Day | 전날 오후 FK용량 | 당일 오전 FK용량 | FK TDM |
|-----|----------------|----------------|--------|
| 1   | 2.0            | 2.0            | 5.5    |
| 2   | 2.5            | 2.5            | 7.2    |
| ... | ...            | ...            | ...    |
| 8   | 3.0            | 3.0            | 8.5    |

---

## 💡 추가 기능

### 여러 병원에서 사용
각 병원별로 별도의 스프레드시트 생성:
- 병원 A: `tacrolimus_tdm_data_hospital_a`
- 병원 B: `tacrolimus_tdm_data_hospital_b`

`app.py`에서 `SPREADSHEET_NAME` 변경 또는 환경 변수로 관리

### 데이터 백업
Google Sheets의 **버전 기록** 활용:
1. 파일 → 버전 기록 → 버전 기록 보기
2. 특정 시점으로 복원 가능

### 데이터 분석
Google Sheets에서 직접:
- 차트 생성
- 피벗 테이블
- Google Data Studio 연동

---

## 📞 지원

문제가 발생하면:
1. 로그 확인: Streamlit Cloud → Manage app → Logs
2. GitHub Issues 등록
3. 담당자에게 문의

---

**마지막 업데이트**: 2025-11-19
**버전**: 1.0

