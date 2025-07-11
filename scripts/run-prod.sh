#!/bin/bash

echo "🚀 운영환경에서 Ururu AI 서비스 시작"
echo "환경: Production"
echo "Spring Boot 연동: 활성화"
echo "데이터: 실제 데이터베이스 사용"

export ENVIRONMENT=production
export BUILD_TARGET=production

# 환경 파일 존재 확인
if [ ! -f ".env.production" ]; then
    echo "❌ .env.production 파일을 찾을 수 없습니다."
    echo "💡 Config 레포지토리에서 환경 파일을 가져와주세요."
    exit 1
fi

# Spring Boot 연결 확인
echo "🔍 Spring Boot 서버 연결 확인 중..."
if curl -f --connect-timeout 5 http://localhost:8080/health 2>/dev/null; then
    echo "✅ Spring Boot 서버 연결 확인됨"
else
    echo "⚠️  Spring Boot 서버에 연결할 수 없습니다."
    echo "   Docker 환경에서는 host.docker.internal:8080으로 연결을 시도합니다."
fi

docker compose up --build

echo "✅ 운영환경 서비스가 시작되었습니다."
echo "🌐 AI 서비스 접근: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "🔗 Spring Boot 연동: 활성화됨"
