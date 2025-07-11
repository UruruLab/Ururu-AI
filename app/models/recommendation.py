from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from app.models.user import BeautyProfile
from app.models.product import Product, ProductCategory


class ProfileBasedRecommendationRequest(BaseModel):
    """BeautyProfile 기반 상품 추천 요청"""
    beauty_profile: BeautyProfile = Field(..., description="사용자 뷰티 프로필")
    top_k: int = Field(default=10, ge=1, le=50, description="추천할 상품 개수")
    exclude_categories: Optional[List[ProductCategory]] = Field(default=None, description="제외할 카테고리")
    include_categories: Optional[List[ProductCategory]] = Field(default=None, description="포함할 카테고리")
    min_similarity: Optional[float] = Field(default=0.3, ge=0, le=1, description="최소 유사도 임계값")
    use_price_filter: bool = Field(default=True, description="프로필의 가격 범위 사용 여부")


class RecommendedProduct(BaseModel):
    """추천된 상품 정보"""
    product: Product
    similarity_score: float = Field(..., ge=0, le=1, description="벡터 유사도 점수")
    profile_match_score: float = Field(..., ge=0, le=1, description="프로필 매칭 점수")
    final_score: float = Field(..., ge=0, le=1, description="최종 추천 점수")
    recommendation_reason: str = Field(..., description="추천 이유")
    matched_features: List[str] = Field(default_factory=list, description="매칭된 특성")
    confidence_score: float = Field(..., ge=0, le=1, description="추천 신뢰도")
    confidence_level: str = Field(..., description="신뢰도 수준 (high/medium/low)")


class ProfileBasedRecommendationResponse(BaseModel):
    """BeautyProfile 기반 상품 추천 응답"""
    recommendations: List[RecommendedProduct]
    total_count: int
    processing_time_ms: float
    user_profile_summary: str = Field(..., description="사용자 프로필 요약")
    request_info: ProfileBasedRecommendationRequest
    
    class Config:
        from_attributes = True


class RecommendationMetadata(BaseModel):
    """추천 시스템 메타데이터"""
    algorithm_version: str = Field(..., description="알고리즘 버전")
    model_name: str = Field(..., description="사용된 모델명")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")
    total_candidates: int = Field(..., description="후보 상품 총 개수")
    confidence_threshold: float = Field(0.3, description="신뢰도 임계값")
    timestamp: datetime = Field(default_factory=datetime.now, description="추천 생성 시간")
    recommendation_method: str = Field(default="profile_based", description="추천 방식")
    error: Optional[str] = Field(None, description="오류 메시지")