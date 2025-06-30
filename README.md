# Beauty Recommendation API

AI 기반 뷰티 상품 추천 시스템 - FastAPI 서버

## 📋 프로젝트 개요

사용자의 뷰티 프로필(피부타입, 고민, 알러지 등)을 분석하여 맞춤형 화장품을 추천하는 AI 시스템입니다.

### 주요 기능

- Two-Tower 임베딩 기반 상품 추천
- OCR을 통한 상품 정보 추출
- KoSBERT 모델을 활용한 한국어 자연어 처리
- Faiss 벡터 검색 엔진 연동

## 🏗️ 프로젝트 구조

```
Ururu/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 애플리케이션 진입점
│   ├── api/                    # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── recommendation.py   # 추천 관련 API (예정)
│   │   └── health.py          # 헬스체크 API (예정)
│   ├── core/                   # 핵심 설정
│   │   ├── __init__.py
│   │   ├── config.py          # 환경 설정 (예정)
│   │   └── database.py        # DB 연결 (예정)
│   ├── models/                 # 데이터베이스 모델
│   │   └── __init__.py
│   └── services/               # 비즈니스 로직
│       ├── __init__.py
│       ├── embedding_service.py    # 임베딩 처리 (예정)
│       ├── recommendation_service.py # 추천 로직 (예정)
│       └── ocr_service.py          # OCR 처리 (예정)
├── requirements.txt            # Python 패키지 의존성
├── .gitignore                 # Git 무시 파일 목록
└── README.md                  # 프로젝트 설명서
```

## 🚀 시작하기

### 1. 저장소 클론

```bash
git clone <repository-url>
cd Ururu
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. 패키지 설치

```bash
# requirements.txt에 명시된 패키지들 설치
pip install -r requirements.txt
```

**requirements.txt 사용법:**

- `pip install -r requirements.txt`: 파일에 명시된 모든 패키지를 정확한 버전으로 설치
- `pip freeze > requirements.txt`: 현재 설치된 패키지들의 버전을 파일로 저장
- 새로운 패키지 추가 시: 패키지 설치 후 `pip freeze > requirements.txt`로 업데이트

### 4. 서버 실행

```bash
# 개발 서버 실행 (코드 변경 시 자동 재시작)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. API 확인

- **서버 주소**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs (Swagger UI)
- **헬스체크**: http://localhost:8000/health

## 🛠️ 기술 스택

### 현재 구현된 기술

- **FastAPI**: 고성능 웹 프레임워크
- **Uvicorn**: ASGI 서버
- **Pydantic**: 데이터 검증 및 설정 관리

### 예정된 기술 스택

- **KoSBERT**: 한국어 문장 임베딩
- **Faiss**: 벡터 유사도 검색
- **OCR**: 상품 이미지 텍스트 추출
- **AWS**: 클라우드 인프라 (EC2, S3)

## 📝 개발 가이드

### 새로운 API 엔드포인트 추가

1. `app/api/` 폴더에 새로운 Python 파일 생성
2. FastAPI 라우터 정의
3. `app/main.py`에서 라우터 등록

### 환경변수 설정

추후 `app/core/config.py`에서 환경변수를 관리할 예정입니다.
