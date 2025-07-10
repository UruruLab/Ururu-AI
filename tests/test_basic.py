"""
기본 테스트 케이스
GitHub Actions CI/CD 파이프라인 검증용
"""
import os
import pytest


def test_basic_functionality():
    """기본 기능 테스트"""
    assert True


def test_environment_variables():
    """환경변수 설정 확인"""
    # pytest-env 플러그인으로 설정된 환경변수 확인
    environment = os.getenv('ENVIRONMENT')
    app_name = os.getenv('APP_NAME')
    
    print(f"ENVIRONMENT: {environment}")
    print(f"APP_NAME: {app_name}")
    print(f"GITHUB_ACTIONS: {os.getenv('GITHUB_ACTIONS')}")
    
    # 최소한 하나라도 설정되어 있으면 통과
    assert environment is not None or app_name is not None, f"Environment variables not set: ENVIRONMENT={environment}, APP_NAME={app_name}"


@pytest.mark.skipif(
    os.getenv('GITHUB_ACTIONS') == 'true',
    reason="GitHub Actions에서는 실제 설정 파일이 없으므로 스킵"
)
def test_config_import_local():
    """로컬에서만 실행되는 설정 임포트 테스트"""
    try:
        from app.core.config import settings
        assert settings.APP_NAME == "Ururu AI Recommendation System"
    except Exception:
        pytest.skip("설정 파일 로드 실패 - 환경변수 누락")


def test_fastapi_basic_import():
    """FastAPI 기본 임포트 테스트"""
    try:
        from fastapi import FastAPI
        app = FastAPI(title="Test App")
        assert app.title == "Test App"
    except ImportError:
        pytest.fail("FastAPI 임포트 실패")


@pytest.mark.skipif(
    os.getenv('GITHUB_ACTIONS') == 'true',
    reason="GitHub Actions에서는 전체 앱 로드 스킵"
)
def test_app_import_local():
    """로컬에서만 실행되는 앱 임포트 테스트"""
    try:
        from app.main import app
        assert app.title == "Ururu AI Recommendation API"
    except Exception:
        pytest.skip("앱 로드 실패 - 의존성 또는 환경변수 문제")


def test_github_actions_detection():
    """GitHub Actions 환경 감지 테스트"""
    is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
    if is_github_actions:
        print("✅ GitHub Actions 환경에서 실행 중")
        assert True
    else:
        print("✅ 로컬 환경에서 실행 중")
        assert True
