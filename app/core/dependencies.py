from app.services.user_tower_service import UserTowerService
from app.services.product_tower_service import ProductTowerService


def get_user_tower_service() -> UserTowerService:
    """User Tower 서비스 의존성 주입"""
    return UserTowerService()


def get_product_tower_service() -> ProductTowerService:
    """Product Tower 서비스 의존성 주입"""
    return ProductTowerService()
