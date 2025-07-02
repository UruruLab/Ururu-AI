from pydantic import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    # 서버 설정
    APP_NAME: str = "Ururu AI Recommendation System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # AI 모델 설정
    EMBEDDING_MODEL_NAME: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    EMBEDDING_DIMENSION: int = 768
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Product Tower 설정
    PRODUCT_EMBEDDING_BATCH_SIZE: int = 32
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner Product for cosine similarity
    TOP_K_RECOMMENDATIONS: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.3
    MAX_SIMILARITY_THRESHOLD: float = 1.0
    
    # 상품 카테고리 설정
    MAIN_CATEGORIES: List[str] = [
        "스킨케어", "메이크업", "클렌징", "마스크팩", 
        "선케어", "향수", "헤어케어", "바디케어"
    ]
    
    # 텍스트 전처리 설정
    KOREAN_STOPWORDS: List[str] = [
        "입니다", "있습니다", "합니다", "됩니다", "있는", "위한", "같은", "이런", "그런"
    ]
    
    # 파일 경로 설정
    BASE_DIR: str = str(Path(__file__).parent.parent.parent)
    DATA_DIR: str = str(Path(BASE_DIR) / "data")
    EMBEDDINGS_SAVE_PATH: str = str(Path(BASE_DIR) / "data" / "embeddings")
    FAISS_INDEX_PATH: str = str(Path(BASE_DIR) / "data" / "faiss_index")
    PRODUCT_DATA_PATH: str = str(Path(BASE_DIR) / "data" / "products")
    MODEL_CACHE_PATH: str = str(Path(BASE_DIR) / "data" / "model_cache")
    
    # 성능 및 리소스 설정
    MAX_WORKERS: int = 4
    CACHE_TTL_SECONDS: int = 3600  # 1시간
    BATCH_PROCESSING_DELAY: float = 0.1  # 배치 처리 간 딜레이(초)
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# 애플리케이션 시작 시 디렉토리 생성
settings.ensure_directories()
