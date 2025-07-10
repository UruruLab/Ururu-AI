from functools import lru_cache
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.services.embedding_service import EmbeddingService
from app.services.user_tower_service import UserTowerService
from app.services.product_tower_service import ProductTowerService
from app.services.product_converter import ProductConverter
from app.services.faiss_service import FaissVectorStore
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스들
_vector_store: Optional[FaissVectorStore] = None
_recommendation_service: Optional[RecommendationService] = None


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

async def get_vector_store() -> FaissVectorStore:
    """벡터 저장소 의존성 주입"""
    global _vector_store
    if _vector_store is None:
        try:
            _vector_store = FaissVectorStore()
            logger.info("✅ Faiss 벡터 저장소 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Faiss 벡터 저장소 초기화 실패: {e}")
            raise
    return _vector_store

async def get_recommendation_service() -> RecommendationService:
    """프로필 기반 추천 서비스 의존성 주입"""
    global _recommendation_service
    
    if _recommendation_service is None:
        try:
            # 필요한 서비스들 초기화
            vector_store = await get_vector_store()
            embedding_service = get_embedding_service()
            product_tower_service = get_product_tower_service()
            user_tower_service = get_user_tower_service()
            
            # 프로필 기반 추천 서비스 생성
            _recommendation_service = RecommendationService(
                vector_store=vector_store,
                embedding_service=embedding_service,
                product_tower_service=product_tower_service,
                user_tower_service=user_tower_service
            )
            
            logger.info("✅ 프로필 기반 추천 서비스 초기화 완료")
                
        except Exception as e:
            logger.error(f"❌ 프로필 기반 추천 서비스 초기화 실패: {e}")
            raise
    
    return _recommendation_service