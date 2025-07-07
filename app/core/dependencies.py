from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.services.embedding_service import EmbeddingService
from app.services.user_tower_service import UserTowerService
from app.services.product_tower_service import ProductTowerService
from app.services.product_converter import ProductConverter


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    """임베딩 모델 의존성 주입 (레거시 호환성)"""
    service = get_embedding_service()
    return service.model

def get_user_tower_service() -> UserTowerService:
    """User Tower 서비스 의존성 주입"""
    return UserTowerService(get_embedding_service())

def get_product_tower_service() -> ProductTowerService:
    """Product Tower 서비스 의존성 주입"""
    return ProductTowerService(get_embedding_service())

def get_product_converter() -> ProductConverter:
    """Product Converter 서비스 의존성 주입"""
    return ProductConverter()
