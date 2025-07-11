#!/bin/bash

echo "🚀 개발환경에서 Ururu AI 서비스 시작"
echo "환경: Development"
echo "Spring Boot 연동: 비활성화"
echo "데이터: Mock 데이터 사용"

export ENVIRONMENT=development
export BUILD_TARGET=development

# 환경 파일 존재 확인
if [ ! -f ".env.development" ]; then
    echo "❌ .env.development 파일을 찾을 수 없습니다."
    echo "💡 Config 레포지토리에서 환경 파일을 가져와주세요."
    exit 1
fi

docker compose -f docker-compose.development.yml up --build

echo "✅ 개발환경 서비스가 시작되었습니다."
echo "🌐 AI 서비스 접근: http://localhost:8001"
echo "📚 API 문서: http://localhost:8001/docs"
