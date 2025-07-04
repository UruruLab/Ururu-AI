import httpx
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.core.config import settings


logger = logging.getLogger(__name__)


class SpringBootClient:
    """Spring Boot와의 인터페이스를 위한 HTTP 클라이언트
    
    역할:
    - AI 추천 결과를 Spring Boot에 전송
    - Spring Boot 서비스 상태 확인
    - 기타 연동 인터페이스
    
    Note: DB 조회는 팀원이 담당하는 직접 DB 연결로 처리
    """
    
    def __init__(self):
        self.base_url = getattr(settings, 'SPRING_BOOT_BASE_URL', 'http://localhost:8080')
        self.timeout = getattr(settings, 'HTTP_TIMEOUT', 30.0)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Ururu-AI/1.0'
            }
        )
        logger.info(f"Spring Boot 연동 클라이언트 초기화: {self.base_url}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Spring Boot 헬스체크"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Spring Boot 헬스체크 실패: {e}")
            raise
    
    async def send_recommendation_result(
            self,
            member_id: int,
            recommendations: List[Dict[str, Any]],
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """AI 추천 결과를 Spring Boot에 전송
        
        Args:
            member_id: 회원 ID
            recommendations: 추천 결과 리스트
            metadata: 추천 메타데이터 (알고리즘 정보 등)
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            payload = {
                "memberId": member_id,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            response = await self.client.post("/api/ai/recommendations", json=payload)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                logger.info(f"회원 {member_id}에 대한 추천 결과 전송 완료")
                return True
            logger.warning(f"추천 결과 전송 실패: {result.get('message')}")
            return False
                
        except Exception as e:
            logger.error(f"추천 결과 전송 실패: {e}")
            return False
    
    async def notify_recommendation_request(
            self,
            member_id: int,
            request_type: str = "manual"
    ) -> bool:
        """추천 요청 알림을 Spring Boot에 전송
        
        Args:
            member_id: 회원 ID
            request_type: 요청 타입 (manual, auto, batch 등)
            
        Returns:
            bool: 알림 성공 여부
        """
        try:
            payload = {
                "memberId": member_id,
                "requestType": request_type,
                "timestamp": datetime.now().isoformat()
            }
            
            response = await self.client.post("/api/ai/recommendation-requests", json=payload)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                logger.info(f"회원 {member_id}의 추천 요청 알림 전송 완료")
                return True
            logger.warning(f"추천 요청 알림 전송 실패: {result.get('message')}")
            return False
                
        except Exception as e:
            logger.error(f"추천 요청 알림 전송 실패: {e}")
            return False
    
    async def update_recommendation_feedback(
            self,
            member_id: int,
            recommendation_id: str,
            feedback_type: str,
            feedback_data: Dict[str, Any]
    ) -> bool:
        """추천 피드백 정보를 Spring Boot에 전송
        
        Args:
            member_id: 회원 ID
            recommendation_id: 추천 결과 ID
            feedback_type: 피드백 타입 (click, purchase, like, dislike 등)
            feedback_data: 피드백 상세 데이터
            
        Returns:
            bool: 피드백 전송 성공 여부
        """
        try:
            payload = {
                "memberId": member_id,
                "recommendationId": recommendation_id,
                "feedbackType": feedback_type,
                "feedbackData": feedback_data,
                "timestamp": datetime.now().isoformat()
            }
            
            response = await self.client.post("/api/ai/feedback", json=payload)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                logger.info(f"추천 피드백 전송 완료: {recommendation_id}")
                return True
            logger.warning(f"추천 피드백 전송 실패: {result.get('message')}")
            return False
                
        except Exception as e:
            logger.error(f"추천 피드백 전송 실패: {e}")
            return False
    
    async def get_ai_service_status(self) -> Dict[str, Any]:
        """AI 서비스 상태를 Spring Boot에 보고
        
        Returns:
            Dict: 서비스 상태 정보
        """
        try:
            payload = {
                "service": "ururu-ai",
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            response = await self.client.post("/api/ai/status", json=payload)
            response.raise_for_status()
            
            return response.json()
                
        except Exception as e:
            logger.error(f"AI 서비스 상태 보고 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# 전역 클라이언트 인스턴스
_spring_client: Optional[SpringBootClient] = None


async def get_spring_client() -> SpringBootClient:
    """Spring Boot 클라이언트 의존성 주입"""
    global _spring_client
    if _spring_client is None:
        _spring_client = SpringBootClient()
    return _spring_client


async def close_spring_client():
    """Spring Boot 클라이언트 종료"""
    global _spring_client
    if _spring_client:
        await _spring_client.close()
        _spring_client = None
