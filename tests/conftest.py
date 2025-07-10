"""
테스트 설정 파일
pytest 실행 전 환경변수를 설정합니다.
"""
import os
import pytest


def pytest_configure():
    """pytest 설정 초기화"""
    # 환경변수가 설정되지 않았다면 기본값 설정
    test_env_vars = {
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'LOG_LEVEL': 'INFO',
        'APP_NAME': 'Ururu AI Recommendation System',
        'VERSION': '1.0.0',
        'EMBEDDING_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
        'EMBEDDING_DIMENSION': '384',
        'MAX_SEQUENCE_LENGTH': '512',
        'PRODUCT_EMBEDDING_BATCH_SIZE': '50',
        'FAISS_INDEX_TYPE': 'IndexFlatIP',
        'TOP_K_RECOMMENDATIONS': '40',
        'MIN_SIMILARITY_THRESHOLD': '0.3',
        'MAX_SIMILARITY_THRESHOLD': '1.0',
        'TEXT_MAX_LENGTH': '500',
        'PRICE_RANGE_LOW': '10000',
        'PRICE_RANGE_MID_LOW': '30000',
        'PRICE_RANGE_MID': '50000',
        'PRICE_RANGE_MID_HIGH': '100000',
        'MAX_KEY_INGREDIENTS': '5',
        'MAX_BENEFITS': '5',
        'MAX_TARGET_CONCERNS': '5',
        'MAX_WORKERS': '2',
        'CACHE_TTL_SECONDS': '3600',
        'BATCH_PROCESSING_DELAY': '0.1',
        'DB_HOST': 'localhost',
        'DB_PORT': '3306',
        'DB_USERNAME': 'test_user',
        'DB_PASSWORD': 'test_password',
        'DB_NAME': 'test_db',
        'DB_CHARSET': 'utf8mb4',
        'AWS_REGION': 'ap-northeast-2',
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'S3_BUCKET_NAME': 'test-bucket'
    }
    
    for key, value in test_env_vars.items():
        if not os.getenv(key):
            os.environ[key] = value
    
    print(f"✅ 테스트 환경변수 설정 완료 (ENVIRONMENT: {os.getenv('ENVIRONMENT')})")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """각 테스트 실행 전 환경 확인"""
    # GitHub Actions 환경인지 확인
    is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
    
    if is_github_actions:
        print("🤖 GitHub Actions 환경에서 테스트 실행 중")
    else:
        print("🖥️ 로컬 환경에서 테스트 실행 중")
    
    yield
    
    # 테스트 후 정리 (필요시)
    pass
