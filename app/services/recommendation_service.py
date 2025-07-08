import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.services.faiss_service import FaissVectorStore
from app.services.embedding_service import EmbeddingService
from app.services.product_tower_service import ProductTowerService
from app.models.product import ProductRecommendationRequest
from app.core.config import settings


logger = logging.getLogger(__name__)


class RecommendationService:
    """추천 비즈니스 로직을 담당하는 서비스"""

    def __init__(
            self,
            vector_store: FaissVectorStore,
            embedding_service: EmbeddingService,
            product_tower_service: ProductTowerService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.product_tower_service = product_tower_service

    async def recommend_products(
            self,
            request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """메인 상품 추천 로직 - 벡터 검색 + 비즈니스 로직"""

        logger.debug(f"상품 추천 시작: {request.user_diagnosis[:30]}...")

        try:
            user_embedding = self.embedding_service.encode_text(request.user_diagnosis)
            raw_scores, product_ids = await self.vector_store.search_vectors(
                user_embedding,
                request.top_k * 2
            )

            if not product_ids:
                logger.warning("추천할 상품이 없습니다.")
                return []
            
            recommendations = self._apply_recommendation_logic(
                raw_scores,
                product_ids,
                request
            )

            logger.debug(f"추천 완료! 추천 상품 수: {len(recommendations)}")
            return recommendations
        except Exception as e:
            logger.error(f"추천 처리 중 오류 발생: {e}")
            return await self._fallback_recommendations(request)
        

    def _apply_recommendation_logic(
            self,
            raw_scores: List[float],
            product_ids: List[int],
            request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """추천 알고리즘"""
        recommendations = []
        
        for i, (raw_score, product_id) in enumerate(zip(raw_scores, product_ids)):
            # 1. Faiss 점수를 유사도로 변환
            similarity_score = self._convert_faiss_score_to_similarity(raw_score)
            
            # 2. 최소 유사도 임계값 체크
            if similarity_score < (request.min_similarity or settings.MIN_SIMILARITY_THRESHOLD):
                continue
            
            # 3. 키워드 매칭 보정 점수 계산
            keyword_boost = self._calculate_keyword_boost(request.user_diagnosis, product_id)
            
            # 4. 최종 추천 점수 계산 (벡터 유사도 + 키워드 보정)
            final_score = similarity_score * 0.7 + keyword_boost * 0.3
            
            # 5. 다양성 보정 (동일 카테고리 너무 많이 추천되지 않도록)
            diversity_penalty = self._calculate_diversity_penalty(recommendations, product_id)
            final_score *= diversity_penalty
            
            # 6. 가격 필터 적용
            if self._passes_price_filter(product_id, request.max_price):
                recommendations.append({
                    "product_id": product_id,
                    "similarity_score": similarity_score,
                    "keyword_boost": keyword_boost,
                    "final_score": final_score,
                    "ranking_position": len(recommendations) + 1,
                    "recommendation_reason": self._generate_recommendation_reason(
                        similarity_score, keyword_boost, product_id
                    ),
                    "confidence_level": self._determine_confidence_level(final_score)
                })
        
        # 최종 점수로 정렬
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 요청된 개수만큼 반환
        return recommendations[:request.top_k]