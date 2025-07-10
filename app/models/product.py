from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from enum import Enum
from decimal import Decimal


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
    description: str = Field(..., description="상품 설명")
    ingredients: Optional[str] = Field(None, description="성분 정보")
    category_main: ProductCategory = Field(..., description="상위 카테고리")
    category_sub: str = Field(..., description="하위 카테고리")
    base_price: Decimal = Field(..., ge=0, description="기본 가격")


class ProductOption(BaseModel):
    id: int
    option_name: str = Field(..., description="옵션명")
    option_price: Decimal = Field(..., ge=0, description="옵션 가격")
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
    embedding_vector: List[float] = Field(..., description="임베딩 벡터")
    text_content: str = Field(..., description="임베딩에 사용된 텍스트 내용")
    model_version: str = Field(default="KoSBERT-v1", description="사용된 모델 버전")
    embedding_dimension: int = Field(default=768, description="임베딩 벡터 차원")
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @validator('embedding_vector')
    def validate_embedding_dimension(cls, v, values):
        expected_dim = values.get('embedding_dimension', 768)
        if len(v) != expected_dim:
            raise ValueError(f"임베딩 벡터 차원이 일치하지 않습니다. 예상: {expected_dim}, 실제: {len(v)}")
        return v
    
    class Config:
        from_attributes = True

class RecommendedProduct(BaseModel):
    """추천된 상품 정보"""
    product: Product
    similarity_score: float = Field(..., ge=0, le=1, description="유사도 점수")
    recommendation_reason: str = Field(..., description="추천 이유")
    matched_keywords: List[str] = Field(default_factory=list, description="매칭된 키워드")
    confidence_score: float = Field(..., ge=0, le=1, description="추천 신뢰도")