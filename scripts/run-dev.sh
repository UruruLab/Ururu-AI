#!/bin/bash

echo "🚀 개발환경에서 Ururu AI 서비스 시작"
echo "환경: Development"
echo "Spring Boot 연동: 비활성화"
echo "데이터: Mock 데이터 사용"

export ENVIRONMENT=development
export BUILD_TARGET=development

docker compose -f docker-compose.development.yml up --build

echo "✅ 개발환경 서비스가 시작되었습니다."
echo "🌐 AI 서비스 접근: http://localhost:8001"
echo "📚 API 문서: http://localhost:8001/docs"
