# 베이스 이미지
FROM python:3.11-slim as base

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 환경 설정
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 개발 환경 스테이지
FROM base as development

# 개발용 추가 도구
RUN pip install --no-cache-dir pytest pytest-asyncio black isort flake8

# 애플리케이션 코드 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p /app/logs && \
    chmod 755 /app/logs

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 개발 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 운영 환경 스테이지
FROM base as production

# 운영용 사용자 생성
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 애플리케이션 코드 복사
COPY --chown=appuser:appgroup . .

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/logs && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app/logs

# 캐시 디렉토리 생성
RUN mkdir -p /app/.cache && \
    chown appuser:appgroup /app/.cache

# 비root 사용자로 전환
USER appuser

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 포트 노출
EXPOSE 8000

# 운영 서버 실행 (Gunicorn 사용)
CMD ["sh", "-c", "gunicorn app.main:app -w ${GUNICORN_WORKERS:-4} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --access-logfile /app/logs/access.log --error-logfile /app/logs/error.log --log-level info"]
