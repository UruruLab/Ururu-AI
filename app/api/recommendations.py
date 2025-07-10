from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import logging
import time

from app.models.recommendation import (
    ProfileBasedRecommendationRequest,
    ProfileBasedRecommendationResponse,
    RecommendedProduct
)
from app.services.recommendation_service import RecommendationService
from app.core.dependencies import get_recommendation_service
from app.core.database import get_async_db
from app.clients.spring_client import get_spring_client, SpringBootClient

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=ProfileBasedRecommendationResponse,
             summary="프로필 기반 맞춤 상품 추천",
             description="""
             **메인 추천 API** - 구조화된 BeautyProfile을 기반으로 맞춤 상품을 추천합니다.
             
             ## 추천 프로세스:
             1. **사용자 프로필 임베딩**: BeautyProfile을 벡터로 변환
             2. **벡터 유사도 검색**: Faiss를 사용해 상품 임베딩과 유사도 계산
             3. **프로필 매칭**: 피부타입, 고민, 알레르기 등 세부 매칭
             4. **통합 점수 계산**: 벡터 유사도(70%) + 프로필 매칭(30%)
             5. **최종 추천**: 가장 적합한 상품들을 순위별로 반환
             
             ## 입력 예시:
             ```json
             {
                 "beauty_profile": {
                     "skin_type": "건성",
                     "skin_tone": "웜톤", 
                     "concerns": ["수분부족", "민감함"],
                     "has_allergy": true,
                     "allergies": ["파라벤"],
                     "interest_categories": ["스킨케어", "선케어"],
                     "min_price": 10000,
                     "max_price": 50000,
                     "additional_info": "민감한 편이라 순한 제품 선호"
                 },
                 "top_k": 10,
                 "include_categories": ["스킨케어"],
                 "use_price_filter": true
             }
             ```
             """)
async def get_recommendations(
    request: ProfileBasedRecommendationRequest,
    background_tasks: BackgroundTasks,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """
    **메인 추천 API** - BeautyProfile 구조화된 데이터 활용
    
    사용자의 피부타입, 고민, 알레르기 등 상세 프로필을 분석하여 
    벡터 유사도와 프로필 매칭을 통해 최적의 화장품을 추천합니다.
    """
    start_time = time.time()
    
    try:
        profile = request.beauty_profile
        logger.info(f"🎯 프로필 기반 추천 요청: {profile.skin_type.value} {profile.skin_tone.value}")
        logger.info(f"📊 요청 파라미터: top_k={request.top_k}, 고민={len(profile.concerns)}개")
        
        # 프로필 기반 벡터 유사도 추천 실행
        recommendation_results = await recommendation_service.recommend_products(request)
        
        if not recommendation_results:
            logger.warning("프로필 기반 추천 결과가 없습니다")
            return ProfileBasedRecommendationResponse(
                recommendations=[],
                total_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                user_profile_summary=recommendation_service.user_tower_service.profile_to_text(profile),
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
                        profile_match_score=result["profile_match_score"],
                        final_score=result["final_score"],
                        recommendation_reason=result["recommendation_reason"],
                        matched_features=result["matched_features"],
                        confidence_score=result.get("confidence_score", result["final_score"]),
                        confidence_level=result.get("confidence_level", "medium")
                    )
                    recommendations.append(recommended_product)
                    
            except Exception as e:
                logger.error(f"프로필 추천 결과 변환 실패 (product_id: {result.get('product_id')}): {e}")
                continue
        
        # 처리 시간 계산
        processing_time = (time.time() - start_time) * 1000
        
        # 프로필 요약 생성
        user_profile_summary = recommendation_service.user_tower_service.profile_to_text(profile)
        
        # 응답 생성
        response = ProfileBasedRecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            user_profile_summary=user_profile_summary,
            request_info=request
        )
        
        logger.info(f"✅ 프로필 기반 추천 완료: {len(recommendations)}개 상품, {processing_time:.2f}ms")
        
        # 백그라운드에서 추천 결과 로깅
        background_tasks.add_task(
            _log_recommendation_result, 
            f"{profile.skin_type.value} {profile.skin_tone.value}",
            len(profile.concerns),
            len(recommendations), 
            processing_time,
            [r.product.id for r in recommendations[:5]]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 프로필 기반 상품 추천 실패: {e}")
        raise HTTPException(status_code=500, detail=f"프로필 기반 상품 추천에 실패했습니다: {str(e)}") from e


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """추천 서비스 헬스체크"""
    try:
        return {
            "service": "profile-recommendation-api",
            "status": "healthy",
            "version": "4.0.0",
            "features": {
                "profile_based_recommendation": True,
                "vector_similarity": True,
                "profile_matching": True
            }
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
    profile_summary: str,
    concern_count: int,
    result_count: int, 
    processing_time: float,
    top_product_ids: List[int]
):
    """프로필 기반 추천 결과 로깅을 위한 백그라운드 태스크"""
    try:
        logger.info(f"📊 프로필 추천 로그: 프로필='{profile_summary}', "
                   f"고민수={concern_count}, 결과수={result_count}, "
                   f"처리시간={processing_time:.2f}ms, 상위상품={top_product_ids}")
        
    except Exception as e:
        logger.error(f"프로필 추천 로그 저장 실패: {e}")