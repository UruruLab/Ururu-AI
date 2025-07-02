from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ProductStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETE = "DELETE"


class ProductCategory(str, Enum):
    SKINCARE = "스킨케어"
    MAKEUP = "메이크업"
    CLEANSING = "클렌징"
    MASK_PACK = "마스크팩"
    SUNCARE = "선케어"
    PERFUME = "향수"
    HAIRCARE = "헤어케어"
    BODYCARE = "바디케어"


class ProductBase(BaseModel):
    name: str = Field(..., description="상품명(시리즈명)")
    brand: str = Field(..., description="브랜드명")
    description: str = Field(..., description="상품 설명")
    ingredients: Optional[str] = Field(None, description="성분 정보")
    category_main: ProductCategory = Field(..., description="상위 카테고리")
    category_sub: str = Field(..., description="하위 카테고리")
    base_price: float = Field(..., ge=0, description="기본 가격")


class ProductCreate(ProductBase):
    pass


class ProductOption(BaseModel):
    id: int
    option_name: str = Field(..., description="옵션명")
    option_price: float = Field(..., ge=0, description="옵션 가격")
    image_url: Optional[str] = Field(None, description="옵션 이미지 URL")
    additional_info: Optional[str] = Field(None, description="옵션 관련 추가 정보")
    is_active: bool = Field(default=True, description="활성화 상태")


class Product(ProductBase):
    id: int
    status: ProductStatus = Field(default=ProductStatus.ACTIVE, description="상품 상태")
    created_at: datetime
    updated_at: Optional[datetime] = None
    options: List[ProductOption] = Field(default_factory=list, description="상품 옵션 목록")
    
    class Config:
        from_attributes = True


class ProductEmbedding(BaseModel):
    """상품 임베딩 정보"""
    id: int
    product_id: int
    embedding_vector: List[float] = Field(..., description="임베딩 벡터 (768차원)")
    text_content: str = Field(..., description="임베딩에 사용된 텍스트 내용")
    model_version: str = Field(default="KoSBERT-v1", description="사용된 모델 버전")
    embedding_dimension: int = Field(default=768, description="임베딩 벡터 차원")
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ProductProfile(BaseModel):
    """상품 프로필 - User의 BeautyProfile과 대응"""
    product_type: ProductCategory
    skin_compatibility: List[str] = Field(..., description="적합한 피부 타입")
    key_ingredients: List[str] = Field(..., description="주요 성분")
    benefits: List[str] = Field(..., description="제품 효능/혜택")
    price_range: str = Field(..., description="가격대")
    target_concerns: List[str] = Field(..., description="타겟 피부 고민")
    brand_positioning: str = Field(..., description="브랜드 포지셔닝")


class ProductRecommendationRequest(BaseModel):
    """상품 추천 요청"""
    user_diagnosis: str = Field(..., description="사용자 진단 결과 텍스트")
    top_k: int = Field(default=10, ge=1, le=50, description="추천할 상품 개수")
    exclude_categories: Optional[List[ProductCategory]] = Field(default=None, description="제외할 카테고리")
    include_categories: Optional[List[ProductCategory]] = Field(default=None, description="포함할 카테고리")
    min_similarity: Optional[float] = Field(default=0.3, ge=0, le=1, description="최소 유사도 임계값")
    max_price: Optional[float] = Field(default=None, ge=0, description="최대 가격 필터")


class RecommendedProduct(BaseModel):
    """추천된 상품 정보"""
    product: Product
    similarity_score: float = Field(..., ge=0, le=1, description="유사도 점수")
    recommendation_reason: str = Field(..., description="추천 이유")
    matched_keywords: List[str] = Field(default_factory=list, description="매칭된 키워드")
    confidence_score: float = Field(..., ge=0, le=1, description="추천 신뢰도")


class ProductRecommendationResponse(BaseModel):
    """상품 추천 응답"""
    recommendations: List[RecommendedProduct]
    total_count: int
    processing_time_ms: float
    request_info: ProductRecommendationRequest
    
    class Config:
        from_attributes = True
