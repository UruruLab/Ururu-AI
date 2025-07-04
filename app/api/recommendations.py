from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import logging

# TODO: 팀원이 구현할 모델들은 일단 주석처리
# from app.models.recommendation import (
#     RecommendationRequest,
#     RecommendationResponse,
#     RecommendationItem,
#     RecommendationMetadata
# )
# from app.services.recommendation_service import get_recommendation_service, RecommendationService
from app.clients.spring_client import get_spring_client, SpringBootClient

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])
logger = logging.getLogger(__name__)


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


# TODO: 추천 API는 팀원의 DB/임베딩 구현 후 활성화
# @router.post("/", response_model=RecommendationResponse)
# async def get_recommendations(
#     request: RecommendationRequest,
#     background_tasks: BackgroundTasks,
#     recommendation_service: RecommendationService = Depends(get_recommendation_service),
#     spring_client: SpringBootClient = Depends(get_spring_client)
# ) -> RecommendationResponse:
#     """메인 추천 API 엔드포인트 - 팀원 구현 대기"""
#     pass


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
