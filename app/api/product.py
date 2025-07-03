from fastapi import APIRouter, HTTPException, Depends
import logging
import time

from app.models.product import (
    Product, ProductCreate, ProductRecommendationRequest, 
    ProductRecommendationResponse, RecommendedProduct,
    ProductCategory, ProductProfile
)
from app.services.product_tower_service import ProductTowerService
from app.core.dependencies import get_product_tower_service


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["products"])


@router.post("/", response_model=Product)
async def create_product(
    product_data: ProductCreate,
    product_service: ProductTowerService = Depends(get_product_tower_service)
):
    """상품 생성"""
    try:
        # 실제 구현에서는 DB에 저장
        # 여기서는 임시로 Product 객체 반환
        from datetime import datetime
        
        product = Product(
            id=1,  # 임시 ID
            **product_data.dict(),
            created_at=datetime.now()
        )
        
        logger.info(f"상품 생성 완료: {product.name}")
        return product
        
    except Exception as e:
        logger.error(f"상품 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="상품 생성에 실패했습니다.")


@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: int):
    """상품 조회"""
    try:
        # 실제 구현에서는 DB에서 조회
        # 임시 데이터 반환
        from datetime import datetime
        
        product = Product(
            id=product_id,
            name="히알루론산 수분 크림",
            brand="라로슈포제",
            description="건성 피부를 위한 깊은 보습 크림입니다. 히알루론산과 세라마이드가 함유되어 24시간 지속되는 수분 공급을 제공합니다.",
            ingredients="히알루론산, 세라마이드, 글리세린, 니아신아마이드",
            category_main=ProductCategory.SKINCARE,
            category_sub="모이스처라이저",
            base_price=35000.0,
            created_at=datetime.now(),
            options=[]
        )
        
        return product
        
    except Exception as e:
        logger.error(f"상품 조회 실패: {e}")
        raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")


@router.post("/recommend", response_model=ProductRecommendationResponse)
async def recommend_products(
    request: ProductRecommendationRequest,
    product_service: ProductTowerService = Depends(get_product_tower_service)
):
    """상품 추천 API - User Tower와 연동하는 핵심 엔드포인트"""
    start_time = time.time()
    
    try:
        logger.info(f"상품 추천 요청: {request.user_diagnosis[:50]}...")
        
        # 실제 구현에서는 임베딩 기반 추천 로직 수행
        # 여기서는 임시 데이터로 응답
        from datetime import datetime
        
        # 샘플 상품들
        sample_products = [
            Product(
                id=1,
                name="히알루론산 수분 크림",
                brand="라로슈포제",
                description="건성 피부를 위한 깊은 보습 크림입니다. 히알루론산과 세라마이드 함유.",
                ingredients="히알루론산, 세라마이드, 글리세린",
                category_main=ProductCategory.SKINCARE,
                category_sub="모이스처라이저",
                base_price=35000.0,
                created_at=datetime.now(),
                options=[]
            ),
            Product(
                id=2,
                name="센텔라 진정 토너",
                brand="코스알엑스",
                description="민감성 피부를 위한 진정 토너입니다. 센텔라 아시아티카 추출물 함유.",
                ingredients="센텔라 아시아티카, 판테놀, 나이아신아마이드",
                category_main=ProductCategory.SKINCARE,
                category_sub="토너",
                base_price=22000.0,
                created_at=datetime.now(),
                options=[]
            )
        ]
        
        # 추천 상품 생성
        recommendations = []
        for i, product in enumerate(sample_products[:request.top_k]):
            # 간단한 키워드 매칭 시뮬레이션
            matched_keywords = []
            if "건성" in request.user_diagnosis or "수분" in request.user_diagnosis:
                if "수분" in product.description or "보습" in product.description:
                    matched_keywords.extend(["수분", "보습"])
            
            if "민감" in request.user_diagnosis:
                if "민감" in product.description or "진정" in product.description:
                    matched_keywords.extend(["민감성", "진정"])
            
            # 유사도 점수 시뮬레이션 (실제로는 임베딩 기반 계산)
            similarity_score = 0.85 - (i * 0.1)  # 간단한 감소 패턴
            
            # 추천 이유 생성
            reason = product_service.generate_recommendation_reason(
                product, matched_keywords, similarity_score
            )
            
            recommended_product = RecommendedProduct(
                product=product,
                similarity_score=similarity_score,
                recommendation_reason=reason,
                matched_keywords=matched_keywords,
                confidence_score=similarity_score * 0.9
            )
            recommendations.append(recommended_product)
        
        # 처리 시간 계산
        processing_time = (time.time() - start_time) * 1000
        
        response = ProductRecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            request_info=request
        )
        
        logger.info(f"상품 추천 완료: {len(recommendations)}개 상품, {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"상품 추천 실패: {e}")
        raise HTTPException(status_code=500, detail="상품 추천에 실패했습니다.")


@router.get("/{product_id}/profile", response_model=ProductProfile)
async def get_product_profile(
    product_id: int,
    product_service: ProductTowerService = Depends(get_product_tower_service)
):
    """상품 프로필 조회"""
    try:
        # 실제 구현에서는 DB에서 상품 조회
        from datetime import datetime
        
        product = Product(
            id=product_id,
            name="히알루론산 수분 크림",
            brand="라로슈포제",
            description="건성 피부를 위한 깊은 보습 크림입니다. 히알루론산과 세라마이드가 함유되어 24시간 지속되는 수분 공급을 제공합니다.",
            ingredients="히알루론산, 세라마이드, 글리세린, 니아신아마이드",
            category_main=ProductCategory.SKINCARE,
            category_sub="모이스처라이저",
            base_price=35000.0,
            created_at=datetime.now(),
            options=[]
        )
        
        # 상품 프로필 추출
        profile = product_service.extract_product_profile(product)
        
        logger.info(f"상품 프로필 조회 완료: {product.name}")
        return profile
        
    except Exception as e:
        logger.error(f"상품 프로필 조회 실패: {e}")
        raise HTTPException(status_code=404, detail="상품 프로필을 찾을 수 없습니다.")


@router.post("/embedding/generate")
async def generate_product_embedding(
    product_id: int,
    product_service: ProductTowerService = Depends(get_product_tower_service)
):
    """상품 임베딩 생성"""
    try:
        # 실제 구현에서는 DB에서 상품 조회 후 임베딩 생성
        logger.info(f"상품 {product_id} 임베딩 생성 시작")
        
        # 임시로 성공 응답
        return {
            "success": True,
            "message": f"상품 {product_id}의 임베딩이 생성되었습니다.",
            "embedding_dimension": 768,
            "model_version": "KoSBERT-v1"
        }
        
    except Exception as e:
        logger.error(f"상품 임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="임베딩 생성에 실패했습니다.")
