FROM python:3.11-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/embeddings \
            /app/data/faiss_index \
            /app/data/model_cache \
            /app/data/products

# Config 디렉토리 생성 (Private Config 리포지토리 volume mount용)
RUN mkdir -p /app/config

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 포트 노출
EXPOSE 8000

# Health check 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/recommendations/health || exit 1

# 실행 스크립트에 실행 권한 부여
RUN chmod +x /app/scripts/run_*.py

# 기본 명령어 (docker-compose에서 오버라이드됨)
CMD ["python", "scripts/run_prod.py"]
