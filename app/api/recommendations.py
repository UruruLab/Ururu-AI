from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import logging
import time
from sqlalchemy import select 

from app.models.product import (
    ProductRecommendationRequest, 
    ProductRecommendationResponse, 
    RecommendedProduct
)
from app.services.recommendation_service import RecommendationService
from app.core.dependencies import get_recommendation_service
from app.core.database import get_async_db
from app.models.database import DBProduct
from app.clients.spring_client import get_spring_client, SpringBootClient

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ProductRecommendationResponse,
             summary="맞춤 상품 추천",
             description="""
             **핵심 추천 API** - 사용자 프로필을 기반으로 맞춤 상품을 추천합니다.
             
             ## 추천 프로세스:
             1. **사용자 프로필 분석**: 입력된 진단 텍스트를 AI가 이해
             2. **임베딩 변환**: 사용자 프로필을 벡터로 변환  
             3. **유사도 검색**: Faiss를 사용해 DB의 모든 상품과 유사도 계산
             4. **맞춤 필터링**: 가격대, 카테고리, 키워드 매칭 등 추가 조건 적용
             5. **최종 추천**: 가장 적합한 상품 40개(또는 요청 수량) 반환
             
             
             ## 입력 예시:
             ```json
             {
                 "user_diagnosis": "20대 건성 피부, 수분 부족으로 당김 현상이 심해요. 민감한 편이라 순한 제품 선호하고, 3만원 이하 예산입니다.",
                 "top_k": 40,
                 "max_price": 30000
             }
             ```
             """)
async def get_product_recommendations(
    request: ProductRecommendationRequest,
    background_tasks: BackgroundTasks,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """
    **메인 추천 API** - 사용자 프로필 기반 맞춤 상품 추천
    
    사용자의 피부 고민, 선호사항, 예산을 분석하여 
    가장 적합한 화장품을 AI가 선별해서 추천합니다.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🎯 상품 추천 요청: '{request.user_diagnosis[:50]}...'")
        logger.info(f"📊 요청 파라미터: top_k={request.top_k}, min_similarity={request.min_similarity}")
        
        # Faiss 벡터 검색 기반 추천 실행
        recommendation_results = await recommendation_service.recommend_products(request)
        
        if not recommendation_results:
            logger.warning("추천 결과가 없습니다")
            return ProductRecommendationResponse(
                recommendations=[],
                total_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                request_info=request
            )
        
        # 추천 결과를 RecommendedProduct 형태로 변환
        recommendations = []
        for result in recommendation_results:
            try:
                # recommendation_service에서 반환된 결과를 파싱
                product_details = await recommendation_service._get_product_details([result["product_id"]])
                
                if result["product_id"] in product_details:
                    product_info = product_details[result["product_id"]]
                    
                    recommended_product = RecommendedProduct(
                        product=product_info["product"],
                        similarity_score=result["similarity_score"],
                        recommendation_reason=result["recommendation_reason"],
                        matched_keywords=result["matched_keywords"],
                        confidence_score=result.get("confidence_score", result["final_score"])
                    )
                    recommendations.append(recommended_product)
                    
            except Exception as e:
                logger.error(f"추천 결과 변환 실패 (product_id: {result.get('product_id')}): {e}")
                continue
        
        # 처리 시간 계산
        processing_time = (time.time() - start_time) * 1000
        
        # 응답 생성
        response = ProductRecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            request_info=request
        )
        
        logger.info(f"✅ 추천 완료: {len(recommendations)}개 상품, {processing_time:.2f}ms")
        
        # 백그라운드에서 추천 결과 로깅 (선택적)
        background_tasks.add_task(
            _log_recommendation_result, 
            request.user_diagnosis[:100], 
            len(recommendations), 
            processing_time,
            [r.product.id for r in recommendations[:5]]  # 상위 5개 상품 ID만 로깅
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 상품 추천 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상품 추천에 실패했습니다: {str(e)}") from e



@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """추천 서비스 헬스체크"""
    try:
        return {
            "service": "recommendation-api",
            "status": "healthy",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        raise HTTPException(status_code=500, detail="서비스 상태 확인 실패")


@router.get("/spring-health")
async def check_spring_boot_connection(
    spring_client: SpringBootClient = Depends(get_spring_client)
) -> Dict[str, Any]:
    """Spring Boot 연결 상태 확인"""
    try:
        result = await spring_client.health_check()
        return {
            "spring_boot_connection": "healthy",
            "spring_boot_response": result
        }
    except Exception as e:
        logger.error(f"Spring Boot 연결 실패: {e}")
        return {
            "spring_boot_connection": "unhealthy",
            "error": str(e)
        }


@router.post("/test-spring-notify")
async def test_spring_notification(
    member_id: int,
    spring_client: SpringBootClient = Depends(get_spring_client)
) -> Dict[str, Any]:
    """Spring Boot 알림 전송 테스트"""
    try:
        success = await spring_client.notify_recommendation_request(
            member_id=member_id,
            request_type="test"
        )
        
        if success:
            return {"success": True, "message": f"회원 {member_id}에게 테스트 알림 전송 완료"}
        else:
            return {"success": False, "message": "알림 전송 실패"}
            
    except Exception as e:
        logger.error(f"테스트 알림 전송 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 실패: {str(e)}")
    
async def _log_recommendation_result(
    user_input: str, 
    result_count: int, 
    processing_time: float,
    top_product_ids: List[int]
):
    """추천 결과 로깅을 위한 백그라운드 태스크"""
    try:
        logger.info(f"📊 추천 로그: 입력='{user_input}', "
                   f"결과수={result_count}, 처리시간={processing_time:.2f}ms, "
                   f"상위상품={top_product_ids}")
        
        # 필요시 추천 성능 메트릭 수집이나 사용자 로그 저장
        
    except Exception as e:
        logger.error(f"추천 로그 저장 실패: {e}")


@router.post("/debug", 
             summary="디버깅용 추천 테스트",
             description="추천 시스템의 각 단계별 결과를 확인하는 디버깅 API")
async def debug_recommendation_process(
    request: ProductRecommendationRequest,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """추천 시스템 디버깅"""
    
    debug_info = {
        "request": request.dict(),
        "steps": {}
    }
    
    try:
        # 1단계: 벡터 검색
        user_embedding = recommendation_service.embedding_service.encode_text(request.user_diagnosis)
        raw_scores, product_ids = await recommendation_service.vector_store.search_vectors(
            user_embedding, 30
        )
        
        debug_info["steps"]["vector_search"] = {
            "total_results": len(product_ids),
            "product_ids": product_ids[:10],  # 처음 10개만
            "scores": raw_scores[:10]
        }
        
        # 2단계: 기본 상품 존재 여부 확인
        basic_stmt = (
            select(DBProduct)
            .where(DBProduct.id.in_(product_ids[:10]))
            .where(DBProduct.status == "ACTIVE")
        )
        basic_result = await db.execute(basic_stmt)
        basic_products = basic_result.scalars().all()
        
        debug_info["steps"]["basic_products"] = {
            "found_count": len(basic_products),
            "product_info": [{"id": p.id, "name": p.name} for p in basic_products]
        }
        
        # 3단계: 카테고리 정보 확인
        category_debug = await recommendation_service.debug_product_categories(product_ids[:5])
        debug_info["steps"]["category_info"] = category_debug
        
        # 4단계: 카테고리 필터 적용 테스트
        if request.include_categories or request.exclude_categories:
            filtered_details = await recommendation_service._get_filtered_product_details(
                product_ids[:10], 
                request.include_categories, 
                request.exclude_categories
            )
            
            debug_info["steps"]["category_filtering"] = {
                "filtered_count": len(filtered_details),
                "filtered_products": [
                    {
                        "id": pid, 
                        "category": details["product"].category_main.value
                    } 
                    for pid, details in filtered_details.items()
                ]
            }
        
        # 5단계: Fallback 테스트
        fallback_results = await recommendation_service._fallback_recommendation(request)
        debug_info["steps"]["fallback"] = {
            "result_count": len(fallback_results),
            "results": fallback_results[:3]  # 처음 3개만
        }
        
        return {
            "status": "debug_complete",
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"디버깅 실패: {e}")
        return {
            "status": "debug_failed",
            "error": str(e),
            "debug_info": debug_info
        }