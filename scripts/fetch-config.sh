#!/bin/bash

set -e

# 환경 변수 확인
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN 환경 변수가 설정되지 않았습니다."
    exit 1
fi

if [ -z "$CONFIG_REPO_URL" ]; then
    echo "❌ CONFIG_REPO_URL 환경 변수가 설정되지 않았습니다."
    exit 1
fi

ENVIRONMENT=${ENVIRONMENT:-development}
CONFIG_DIR="/config"
TEMP_DIR="/tmp/ururu-config"

echo "🔄 Private Config 리포지토리에서 설정 파일을 가져옵니다..."
echo "📂 환경: $ENVIRONMENT"
echo "🔗 리포지토리: $CONFIG_REPO_URL"

# 기존 임시 디렉토리 정리
rm -rf "$TEMP_DIR"

# GitHub Token을 사용하여 Private 리포지토리 클론
REPO_URL_WITH_TOKEN=$(echo "$CONFIG_REPO_URL" | sed "s|https://|https://$GITHUB_TOKEN@|")

echo "📥 Config 리포지토리 클론 중..."
git clone "$REPO_URL_WITH_TOKEN" "$TEMP_DIR"

# Config 디렉토리 생성
mkdir -p "$CONFIG_DIR"

# 환경별 설정 파일 복사
if [ -f "$TEMP_DIR/.env.$ENVIRONMENT" ]; then
    echo "✅ .env.$ENVIRONMENT 파일을 복사합니다."
    cp "$TEMP_DIR/.env.$ENVIRONMENT" "$CONFIG_DIR/.env.$ENVIRONMENT"
else
    echo "⚠️  .env.$ENVIRONMENT 파일이 없습니다. 기본 .env 파일을 사용합니다."
    if [ -f "$TEMP_DIR/.env" ]; then
        cp "$TEMP_DIR/.env" "$CONFIG_DIR/.env.$ENVIRONMENT"
    else
        echo "❌ 기본 .env 파일도 없습니다."
        exit 1
    fi
fi

# 공통 설정 파일들 복사
if [ -f "$TEMP_DIR/nginx.conf" ]; then
    echo "✅ nginx.conf 파일을 복사합니다."
    mkdir -p "$CONFIG_DIR/nginx"
    cp "$TEMP_DIR/nginx.conf" "$CONFIG_DIR/nginx/"
fi

if [ -f "$TEMP_DIR/fluent-bit.conf" ]; then
    echo "✅ fluent-bit.conf 파일을 복사합니다."
    mkdir -p "$CONFIG_DIR/fluent-bit"
    cp "$TEMP_DIR/fluent-bit.conf" "$CONFIG_DIR/fluent-bit/"
fi

# SSL 인증서 복사 (운영 환경)
if [ "$ENVIRONMENT" = "production" ] && [ -d "$TEMP_DIR/ssl" ]; then
    echo "✅ SSL 인증서를 복사합니다."
    mkdir -p "$CONFIG_DIR/ssl"
    cp -r "$TEMP_DIR/ssl/"* "$CONFIG_DIR/ssl/"
fi

# 환경별 추가 설정 파일 복사
if [ -d "$TEMP_DIR/$ENVIRONMENT" ]; then
    echo "✅ $ENVIRONMENT 환경별 설정을 복사합니다."
    cp -r "$TEMP_DIR/$ENVIRONMENT/"* "$CONFIG_DIR/"
fi

# 권한 설정
chmod -R 644 "$CONFIG_DIR"
find "$CONFIG_DIR" -type d -exec chmod 755 {} \;

# 임시 디렉토리 정리
rm -rf "$TEMP_DIR"

echo "🎉 Config 파일 동기화가 완료되었습니다!"
echo "📁 복사된 파일 목록:"
ls -la "$CONFIG_DIR"

# 환경 변수 파일 검증
if [ -f "$CONFIG_DIR/.env.$ENVIRONMENT" ]; then
    echo "✅ 환경 설정 파일 검증 완료: .env.$ENVIRONMENT"
    # 민감한 정보 제외하고 설정 확인
    echo "📋 주요 설정 변수:"
    grep -E "^(ENVIRONMENT|DEBUG|LOG_LEVEL|SPRING_BOOT_BASE_URL)" "$CONFIG_DIR/.env.$ENVIRONMENT" | head -5 || true
else
    echo "❌ 환경 설정 파일이 없습니다: .env.$ENVIRONMENT"
    exit 1
fi

echo "🚀 Config 동기화가 성공적으로 완료되었습니다!"
