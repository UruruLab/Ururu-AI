"""
데이터베이스 모델 통합 관리
- 기존 Pydantic 모델과 분리
- SQLAlchemy 모델만 여기서 관리
"""
from .base import Base
from .spring_models import (
    DBProduct, DBProductOption, DBMember, DBBeautyProfile,
    DBProductCategory, DBProductNotice, DBProductTag, DBCategory
)
from .ai_models import (
    DBProductEmbedding, DBRecommendationResult, DBVectorIndex, DBUserEmbedding
)

# 그룹별 리스트
SPRING_MODELS = [
    DBProduct, DBProductOption, DBMember, DBBeautyProfile,
    DBProductCategory, DBProductNotice, DBProductTag, DBCategory
]

AI_MODELS = [
    DBProductEmbedding, DBRecommendationResult, DBVectorIndex, DBUserEmbedding
]

ALL_MODELS = SPRING_MODELS + AI_MODELS

__all__ = [
    "Base",
    # Spring Boot 모델들
    "DBProduct", "DBProductOption", "DBMember", "DBBeautyProfile",
    "DBProductCategory", "DBProductNotice", "DBProductTag", "DBCategory"
    # AI 모델들
    "DBProductEmbedding", "DBRecommendationResult", "DBVectorIndex", "DBUserEmbedding",
    # 그룹
    "SPRING_MODELS", "AI_MODELS", "ALL_MODELS"
]