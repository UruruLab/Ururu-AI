from pydantic_settings import BaseSettings
from typing import List, Dict
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
    
    # 로깅 설정 (환경변수에서만 가져옴)
    LOG_LEVEL: str
    LOG_FORMAT: str
    
    # 브랜드 분류 (환경변수에서만 가져옴)
    PREMIUM_BRANDS: str  # 콤마 구분 문자열
    DRUGSTORE_BRANDS: str  # 콤마 구분 문자열
    KOREAN_BRANDS: str  # 콤마 구분 문자열
    
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
    
    @staticmethod
    def parse_comma_separated_list(value: str) -> List[str]:
        """콤마로 구분된 문자열을 리스트로 변환"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return []
    
    def get_premium_brands(self) -> List[str]:
        """프리미엄 브랜드 리스트 반환"""
        return self.parse_comma_separated_list(self.PREMIUM_BRANDS)
    
    def get_drugstore_brands(self) -> List[str]:
        """드럭스토어 브랜드 리스트 반환"""
        return self.parse_comma_separated_list(self.DRUGSTORE_BRANDS)
    
    def get_korean_brands(self) -> List[str]:
        """한국 브랜드 리스트 반환"""
        return self.parse_comma_separated_list(self.KOREAN_BRANDS)
    
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
        # 환경변수 파일 경로 설정 (Docker에서 오버라이드 가능)
        env_file = os.getenv("ENV_FILE_PATH", ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# 애플리케이션 시작 시 디렉토리 생성
settings.ensure_directories()
