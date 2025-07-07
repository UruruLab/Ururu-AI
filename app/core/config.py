# app/core/config.py 완전 수정 버전
from pydantic_settings import BaseSettings
from typing import List, Dict, Optional
from pathlib import Path
import os

class Settings(BaseSettings):
    # 서버 설정 (고정값)
    APP_NAME: str = "Ururu AI Recommendation System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 환경 설정 (환경변수에서만 가져옴)
    ENVIRONMENT: str
    
    # AI 모델 설정 (환경변수에서만 가져옴)
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_DIMENSION: int
    MAX_SEQUENCE_LENGTH: int
    
    # Product Tower 설정 (환경변수에서만 가져옴)
    PRODUCT_EMBEDDING_BATCH_SIZE: int
    FAISS_INDEX_TYPE: str
    TOP_K_RECOMMENDATIONS: int
    MIN_SIMILARITY_THRESHOLD: float
    MAX_SIMILARITY_THRESHOLD: float
    
    # 텍스트 전처리 설정 (환경변수에서만 가져옴)
    TEXT_MAX_LENGTH: int
    
    # 가격대 분류 기준 (환경변수에서만 가져옴)
    PRICE_RANGE_LOW: int
    PRICE_RANGE_MID_LOW: int
    PRICE_RANGE_MID: int
    PRICE_RANGE_MID_HIGH: int
    
    # 추출 제한 설정 (환경변수에서만 가져옴)
    MAX_KEY_INGREDIENTS: int
    MAX_BENEFITS: int
    MAX_TARGET_CONCERNS: int
    
    # 성능 및 리소스 설정 (환경변수에서만 가져옴)
    MAX_WORKERS: int
    CACHE_TTL_SECONDS: int
    BATCH_PROCESSING_DELAY: float
    
    # Spring Boot 연동 설정 (환경변수에서만 가져옴)
    SPRING_BOOT_BASE_URL: str = "http://localhost:8080"
    HTTP_TIMEOUT: float = 30.0
    HTTP_RETRY_COUNT: int = 3
    HTTP_RETRY_DELAY: float = 1.0
    
    # 로깅 설정 (환경변수에서만 가져옴)
    LOG_LEVEL: str
    LOG_FORMAT: str

    # 상품 카테고리 설정 (고정값 - 비즈니스 로직)
    MAIN_CATEGORIES: List[str] = [
        "스킨케어", "메이크업", "클렌징", "마스크팩", 
        "선케어", "향수", "헤어케어", "바디케어"
    ]
    
    # 텍스트 전처리 설정 (고정값 - 한국어 특성)
    KOREAN_STOPWORDS: List[str] = [
        "입니다", "있습니다", "합니다", "됩니다", "있는", "위한", "같은", "이런", "그런"
    ]
    
    # 뷰티 용어 정규화 매핑 (고정값 - 도메인 지식)
    BEAUTY_TERMS_MAPPING: Dict[str, str] = {
        "보습": "보습 수분공급",
        "수분": "수분 보습", 
        "탄력": "탄력 안티에이징",
        "미백": "미백 브라이트닝",
        "주름": "주름개선 안티에이징",
        "트러블": "트러블케어 진정",
        "민감": "민감성피부 순한",
        "건성": "건성피부 보습",
        "지성": "지성피부 유분조절",
        "복합성": "복합성피부",
        "각질": "각질제거 엑스폴리에이션",
        "모공": "모공케어 모공축소",
        "여드름": "여드름 트러블케어",
        "기미": "기미 미백",
        "잡티": "잡티 브라이트닝"
    }

    # 데이터베이스 설정 (환경변수에서만 가져옴)
    DB_HOST: str
    DB_PORT: int = 3306
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_CHARSET: str = "utf8mb4"
    
    # 데이터베이스 연결 풀 설정
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 0
    DB_POOL_RECYCLE: int = 300
    DB_POOL_PRE_PING: bool = True
    
    # AWS 설정 (S3 연동용)
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_BUCKET_NAME: str
    
    # Vector DB 설정
    VECTOR_INDEX_PATH: str = "data/faiss_index"
    VECTOR_BACKUP_S3_PREFIX: str = "vector-indices"
    
    # 파일 경로 설정 (동적 계산)
    @property
    def BASE_DIR(self) -> str:
        return str(Path(__file__).parent.parent.parent)
    
    @property
    def DATA_DIR(self) -> str:
        return str(Path(self.BASE_DIR) / "data")
    
    @property
    def EMBEDDINGS_SAVE_PATH(self) -> str:
        return str(Path(self.BASE_DIR) / "data" / "embeddings")
    
    @property
    def FAISS_INDEX_PATH(self) -> str:
        return str(Path(self.BASE_DIR) / "data" / "faiss_index")
    
    @property
    def PRODUCT_DATA_PATH(self) -> str:
        return str(Path(self.BASE_DIR) / "data" / "products")
    
    @property
    def MODEL_CACHE_PATH(self) -> str:
        return str(Path(self.BASE_DIR) / "data" / "model_cache")
    
    @property
    def database_url(self) -> str:
        """동기 데이터베이스 URL"""
        return f"mysql+pymysql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset={self.DB_CHARSET}"
    
    @property
    def async_database_url(self) -> str:
        """비동기 데이터베이스 URL"""
        return f"mysql+aiomysql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset={self.DB_CHARSET}"
    
    def is_database_configured(self) -> bool:
        """실제 데이터베이스 설정 여부 확인"""
        return (
            self.DB_HOST != "localhost" and 
            self.DB_USERNAME != "root" and
            self.DB_PASSWORD != "password" and
            len(self.DB_PASSWORD) > 5  
        )
    
    @staticmethod
    def parse_comma_separated_list(value: str) -> List[str]:
        """콤마로 구분된 문자열을 리스트로 변환"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return []
    
    def ensure_directories(self):
        """필요한 디렉토리들을 생성"""
        directories = [
            self.DATA_DIR,
            self.EMBEDDINGS_SAVE_PATH,
            self.FAISS_INDEX_PATH,
            self.PRODUCT_DATA_PATH,
            self.MODEL_CACHE_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, any]:
        """현재 모델 설정 정보 반환"""
        return {
            "model_name": self.EMBEDDING_MODEL_NAME,
            "dimension": self.EMBEDDING_DIMENSION,
            "max_length": self.MAX_SEQUENCE_LENGTH,
            "batch_size": self.PRODUCT_EMBEDDING_BATCH_SIZE,
            "environment": self.ENVIRONMENT
        }
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property 
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    class Config:
        # 🔧 올바른 환경변수 파일 경로 설정
        env_file = os.getenv("ENV_FILE_PATH", ".env.development")  # 실제 파일명과 일치
        env_file_encoding = "utf-8"
        case_sensitive = True


# 🔧 안전한 설정 초기화
try:
    settings = Settings()
    settings.ensure_directories()
    print(f"✅ 설정 로드 완료 (환경: {settings.ENVIRONMENT})")
    
    if settings.is_database_configured():
        print(f"✅ 데이터베이스 설정: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    else:
        print("⚠️  기본 데이터베이스 설정 사용 중 (.env.development 파일 확인 필요)")
        
except Exception as e:
    print(f"❌ 설정 로드 실패: {e}")
    print("🔧 환경변수 파일(.env.development)을 확인하세요")
    # 에러 발생 시 앱 종료 (개발 중에는 문제를 명확히 해결하는 것이 중요)
    raise