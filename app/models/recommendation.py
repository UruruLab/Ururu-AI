from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RecommendationRequest(BaseModel):
    """추천 요청 모델"""
    member_id: int = Field(..., description="회원 ID")
    category_id: Optional[str] = Field(None, description="카테고리 필터")
    limit: int = Field(10, ge=1, le=50, description="추천 개수 (1-50)")
    include_groupbuys: bool = Field(True, description="공동구매 포함 여부")
    filters: Optional[Dict[str, Any]] = Field(None, description="추가 필터 조건")


class RecommendationItem(BaseModel):
    """추천 아이템 모델"""
    item_id: int = Field(..., description="상품/공구 ID")
    item_type: str = Field(..., description="아이템 타입 (product, groupbuy)")
    title: str = Field(..., description="상품/공구 제목")
    description: Optional[str] = Field(None, description="상품 설명")
    price: Optional[int] = Field(None, description="가격")
    discount_rate: Optional[float] = Field(None, description="할인율")
    image_url: Optional[str] = Field(None, description="대표 이미지 URL")
    category_names: List[str] = Field(default_factory=list, description="카테고리명 리스트")
    brand_name: Optional[str] = Field(None, description="브랜드명")
    
    # 추천 관련 메타데이터
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="유사도 점수")
    confidence: float = Field(..., ge=0.0, le=1.0, description="추천 신뢰도")
    reason: Optional[str] = Field(None, description="추천 이유")
    recommendation_id: str = Field(..., description="추천 결과 고유 ID")
    
    # 공동구매 전용 필드
    groupbuy_status: Optional[str] = Field(None, description="공구 상태 (OPEN, CLOSED 등)")
    current_participants: Optional[int] = Field(None, description="현재 참여자 수")
    min_participants: Optional[int] = Field(None, description="최소 참여자 수")
    end_date: Optional[datetime] = Field(None, description="공구 종료일")


class RecommendationMetadata(BaseModel):
    """추천 메타데이터 모델"""
    algorithm_version: str = Field(..., description="알고리즘 버전")
    model_name: str = Field(..., description="사용된 모델명")
    processing_time_ms: int = Field(0, description="처리 시간 (밀리초)")
    total_candidates: int = Field(0, description="후보 아이템 총 개수")
    confidence_threshold: float = Field(0.7, description="신뢰도 임계값")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="추천 생성 시간 (UTC)")
    error: Optional[str] = Field(None, description="오류 메시지")


class RecommendationResponse(BaseModel):
    """추천 응답 모델"""
    member_id: int = Field(..., description="회원 ID")
    recommendations: List[RecommendationItem] = Field(..., description="추천 아이템 리스트")
    metadata: RecommendationMetadata = Field(..., description="추천 메타데이터")


class BeautyProfile(BaseModel):
    """뷰티 프로필 모델 (팀원이 DB 연동 시 사용할 인터페이스)"""
    member_id: int = Field(..., description="회원 ID")
    skin_type: Optional[str] = Field(None, description="피부 타입")
    skin_concerns: List[str] = Field(default_factory=list, description="피부 고민 리스트")
    allergic_ingredients: List[str] = Field(default_factory=list, description="알레르기 성분")
    preferred_categories: List[str] = Field(default_factory=list, description="선호 카테고리")
    age_range: Optional[str] = Field(None, description="연령대")
    additional_info: Optional[str] = Field(None, description="추가 정보")
    
    def to_text(self) -> str:
        """프로필을 임베딩용 텍스트로 변환"""
        parts = []
        
        if self.skin_type:
            parts.append(f"피부타입: {self.skin_type}")
            
        if self.skin_concerns:
            parts.append(f"피부고민: {', '.join(self.skin_concerns)}")
            
        if self.allergic_ingredients:
            parts.append(f"알레르기성분: {', '.join(self.allergic_ingredients)}")
            
        if self.preferred_categories:
            parts.append(f"관심카테고리: {', '.join(self.preferred_categories)}")
            
        if self.age_range:
            parts.append(f"연령대: {self.age_range}")
            
        if self.additional_info:
            parts.append(f"추가정보: {self.additional_info}")
        
        return " | ".join(parts)


class ProductInfo(BaseModel):
    """상품 정보 모델 (팀원이 DB 연동 시 사용할 인터페이스)"""
    product_id: int = Field(..., description="상품 ID")
    name: str = Field(..., description="상품명")
    description: Optional[str] = Field(None, description="상품 설명")
    brand_name: Optional[str] = Field(None, description="브랜드명")
    categories: List[str] = Field(default_factory=list, description="카테고리 리스트")
    ingredients: Optional[str] = Field(None, description="성분 정보")
    effects: Optional[str] = Field(None, description="효과/효능")
    price: Optional[int] = Field(None, description="가격")
    
    def to_text(self) -> str:
        """상품 정보를 임베딩용 텍스트로 변환"""
        parts = [f"상품명: {self.name}"]
        
        if self.brand_name:
            parts.append(f"브랜드: {self.brand_name}")
            
        if self.categories:
            parts.append(f"카테고리: {', '.join(self.categories)}")
            
        if self.description:
            parts.append(f"설명: {self.description}")
            
        if self.ingredients:
            parts.append(f"성분: {self.ingredients}")
            
        if self.effects:
            parts.append(f"효과: {self.effects}")
        
        return " | ".join(parts)


class GroupbuyInfo(BaseModel):
    """공동구매 정보 모델 (팀원이 DB 연동 시 사용할 인터페이스)"""
    groupbuy_id: int = Field(..., description="공동구매 ID")
    title: str = Field(..., description="공구 제목")
    description: Optional[str] = Field(None, description="공구 설명")
    product_info: ProductInfo = Field(..., description="연관 상품 정보")
    discount_rate: float = Field(..., ge=0, description="할인율")
    current_participants: int = Field(..., ge=0, description="현재 참여자 수")
    min_participants: int = Field(..., gt=0, description="최소 참여자 수")
    max_participants: Optional[int] = Field(None, gt=0, description="최대 참여자 수")
    status: str = Field(..., description="공구 상태")
    end_date: datetime = Field(..., description="공구 종료일")
    
    def to_text(self) -> str:
        """공동구매 정보를 임베딩용 텍스트로 변환"""
        parts = [
            f"공동구매: {self.title}",
            f"할인율: {self.discount_rate}%"
        ]
        
        if self.description:
            parts.append(f"설명: {self.description}")
            
        # 연관 상품 정보 포함
        parts.append(self.product_info.to_text())
        
        return " | ".join(parts)
