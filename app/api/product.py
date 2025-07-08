from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import logging
import time
from typing import List

from app.models.product import (
    Product, ProductRecommendationRequest, 
    ProductRecommendationResponse, RecommendedProduct,
    ProductCategory
)
from app.services.recommendation_service import RecommendationService
from app.services.product_tower_service import ProductTowerService
from app.services.product_converter import ProductConverter, get_product_converter
from app.core.dependencies import get_recommendation_service
from app.core.dependencies import get_product_tower_service
from app.models.database import DBProduct
from app.core.database import get_async_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["products"])


@router.post("/recommend", response_model=ProductRecommendationResponse)
async def recommend_products(
    request: ProductRecommendationRequest,
    product_service: ProductTowerService = Depends(get_product_tower_service),
    converter: ProductConverter = Depends(get_product_converter),
    db: AsyncSession = Depends(get_async_db)
):
    """상품 추천 API - User Tower와 연동하는 핵심 엔드포인트"""
    start_time = time.time()
    
    try:
        logger.info(f"상품 추천 요청: {request.user_diagnosis[:50]}...")
        
        stmt = (
            select(DBProduct)
            .options(selectinload(DBProduct.product_options))
            .where(DBProduct.status == "ACTIVE")
            .limit(request.top_k * 2)  # 여유분 조회
        )        
        result = await db.execute(stmt)
        db_products = result.scalars().all()

        if not db_products:
            return ProductRecommendationResponse(
                recommendations=[],
                total_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                request_info=request
            )
        
        # 상품들을 Pydantic 모델로 변환
        products = []
        for db_product in db_products[:request.top_k]:
            try:
                product = await converter.db_to_pydantic(db, db_product)
                products.append(product)
            except Exception as e:
                logger.error(f"상품 변환 실패: {db_product.id}, 에러: {e}")
                continue
        
        # 추천 상품 생성
        recommendations = []
        for i, product in enumerate(products):
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


@router.post("/embedding/generate",
             summary="상품 임베딩 생성",
             description="상품 ID를 기반으로 상품 임베딩을 생성합니다. ")
async def generate_product_embedding(
    product_id: int,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    converter: ProductConverter = Depends(get_product_converter),
    db: AsyncSession = Depends(get_async_db)
):
    """상품 임베딩 생성"""
    try:
        logger.info(f"상품 {product_id} 임베딩 생성 시작")
        
        # DB에서 상품 조회 (옵션과 함께)
        stmt = (
            select(DBProduct)
            .options(selectinload(DBProduct.product_options))
            .where(DBProduct.id == product_id)
        )
        result = await db.execute(stmt)
        db_product = result.scalar_one_or_none()
        
        if not db_product:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")
        
        product = await converter.db_to_pydantic(db, db_product)
        processed_text = recommendation_service.product_tower_service.preprocess_product_text(product)
        
        # 임베딩 벡터 생성
        embedding_vector = recommendation_service.embedding_service.encode_text(processed_text)
        
        embedding_dimension = len(embedding_vector)
        logger.info(f"임베딩 생성 완료 - 차원: {embedding_dimension}, 텍스트 길이: {len(processed_text)}")
        
        success = await recommendation_service.vector_store.add_embeddings([{
            "product_id": product_id,
            "embedding": embedding_vector,
            "metadata": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed_text": processed_text[:200]
            }
        }])

        if not success:
            raise HTTPException(status_code=500, detail="벡터 저장소 추가 실패")

        return {
            "success": True,
            "message": f"상품 {product_id}({product.name})의 임베딩이 생성되었습니다",
            "product_info": {
                "id": product.id,
                "name": product.name,
                "category": f"{product.category_main.value} > {product.category_sub}",
                "base_price": float(product.base_price)
            },
            "embedding_info": {
                "dimension": len(embedding_vector),
                "text_length": len(processed_text),
                "model_version": recommendation_service.embedding_service.get_model_info().get("model_name"),
                "vector_store_added": success,
                "sample_values": embedding_vector[:5]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"상품 임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}") from e
    
    
@router.get("/service/status",
            summary="상품 서비스 상태",
            description="상품 관리 서비스의 상태와 통계를 조회합니다")
async def get_product_service_status(
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """상품 서비스 상태 조회"""
    try:
        from sqlalchemy import func
        from app.models.database import DBProductOption
        
        # 기본 상품 통계
        active_products_stmt = select(func.count(DBProduct.id)).where(DBProduct.status == "ACTIVE")
        active_result = await db.execute(active_products_stmt)
        active_products = active_result.scalar() or 0
        
        total_products_stmt = select(func.count(DBProduct.id))
        total_result = await db.execute(total_products_stmt)
        total_products = total_result.scalar() or 0
        
        # 상품 옵션 통계
        active_options_stmt = select(func.count(DBProductOption.id)).where(DBProductOption.is_deleted == False)
        options_result = await db.execute(active_options_stmt)
        active_options = options_result.scalar() or 0
        
        # 추천 서비스 통계
        recommendation_stats = recommendation_service.get_recommendation_stats()
        
        return {
            "timestamp": "2025-01-01T00:00:00",
            "service_status": "healthy",
            "product_statistics": {
                "total_products": total_products,
                "active_products": active_products,
                "inactive_products": total_products - active_products,
                "active_options": active_options,
                "average_options_per_product": round(active_options / max(active_products, 1), 2)
            },
            "recommendation_integration": {
                "service_connected": True,
                "vector_store_ready": recommendation_stats["vector_store_stats"]["index_stats"]["status"] == "ready",
                "embedding_model": recommendation_stats["embedding_model"]["model_name"],
                "total_vectors": recommendation_stats["vector_store_stats"]["index_stats"]["total_vectors"]
            },
            "capabilities": {
                "individual_embedding_generation": True,
                "product_text_preprocessing": True,
                "category_mapping": True,
                "price_analysis": True
            },
            "api_endpoints": {
                "generate_embedding": "/products/embedding/generate",
                "service_status": "/products/service/status"
            },
            "note": "추천 기능은 /api/recommendations에서 제공됩니다"
        }
        
    except Exception as e:
        logger.error(f"상품 서비스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 상태 조회 실패: {str(e)}")