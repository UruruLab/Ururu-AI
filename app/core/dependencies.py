from functools import lru_cache
from app.services.embedding_service import EmbeddingService
from app.services.user_tower_service import UserTowerService
from app.services.product_tower_service import ProductTowerService

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

def get_user_tower_service() -> UserTowerService:
    """User Tower 서비스 의존성 주입"""
    return UserTowerService(get_embedding_service())


def get_product_tower_service() -> ProductTowerService:
    """Product Tower 서비스 의존성 주입"""
    return ProductTowerService(get_embedding_service())
