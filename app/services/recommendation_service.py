import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, not_
from sqlalchemy.orm import selectinload

from app.services.faiss_service import FaissVectorStore
from app.services.embedding_service import EmbeddingService
from app.services.product_tower_service import ProductTowerService
from app.services.product_converter import ProductConverter
from app.models.product import ProductRecommendationRequest, Product, ProductCategory
from app.models.database import DBProduct, DBProductOption, DBCategory, DBProductCategory
from app.core.config import settings
from app.core.database import AsyncSessionLocal


logger = logging.getLogger(__name__)


class RecommendationService:
    """순수 임베딩 기반 추천 서비스 - AI 중심 접근법"""
    
    def __init__(
        self, 
        vector_store: FaissVectorStore,
        embedding_service: EmbeddingService,
        product_tower_service: ProductTowerService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.product_tower_service = product_tower_service
        self.product_converter = ProductConverter()
        logger.info("🎯 순수 임베딩 기반 추천 서비스 초기화 완료")
    
    async def recommend_products(
        self, 
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """순수 임베딩 기반 상품 추천 - AI가 모든 매칭 담당"""
        
        logger.info(f"🧠 AI 임베딩 기반 추천 시작: '{request.user_diagnosis[:30]}...'")
        logger.info(f"📂 필터: include={request.include_categories}, exclude={request.exclude_categories}")
        
        try:
            # 1. 사용자 진단을 임베딩으로 변환 (AI가 의미 이해)
            user_embedding = self.embedding_service.encode_text(request.user_diagnosis)
            logger.debug(f"🧠 사용자 임베딩 생성 완료: {len(user_embedding)}차원")
            
            # 2. Faiss 벡터 검색 (AI가 의미적 유사도 계산)
            search_multiplier = 3 if (request.include_categories or request.exclude_categories) else 2
            search_k = min(request.top_k * search_multiplier, 100)

            raw_scores, product_ids = await self.vector_store.search_vectors(
                user_embedding, 
                search_k 
            )
            
            if not product_ids:
                logger.warning("벡터 검색 결과 없음")
                return await self._fallback_recommendation(request)
            
            logger.info(f"🔍 벡터 검색 완료: {len(product_ids)}개 상품 (AI 유사도 기반)")
            
            # 3. 카테고리 필터링만 적용 (비즈니스 룰)
            product_details = await self._get_product_details(
                product_ids,
                request.include_categories,
                request.exclude_categories
            )

            if not product_details:
                logger.warning("카테고리 필터링 후 결과 없음 - Fallback 실행")
                return await self._fallback_recommendation(request)
            
            logger.info(f"📊 카테고리 필터링 후: {len(product_details)}개 상품")

            # 4. 순수 임베딩 점수만으로 추천 생성
            recommendations = await self._create_pure_embedding_recommendations(
                raw_scores, 
                product_ids, 
                product_details,
                request
            )
            
            logger.info(f"✅ 순수 AI 추천 완료: {len(recommendations)}개 상품")
            return recommendations
            
        except Exception as e:
            logger.error(f"AI 추천 실패: {e}")
            return await self._fallback_recommendation(request)
    
    async def _create_pure_embedding_recommendations(
        self, 
        raw_scores: List[float], 
        all_product_ids: List[int], 
        product_details: Dict[int, Dict[str, Any]],
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """순수 임베딩 점수만으로 추천 생성 - 키워드 매칭 없음"""
        
        recommendations = []
        
        # 점수 순서대로 처리 (이미 Faiss에서 유사도 순으로 정렬됨)
        for i, (raw_score, product_id) in enumerate(zip(raw_scores, all_product_ids)):
            if product_id not in product_details:
                continue
            
            details = product_details[product_id]
            product = details["product"]
            
            # 1. Faiss 점수를 정규화된 유사도로 변환 (0~1)
            similarity_score = self._convert_faiss_score_to_similarity(raw_score)
            
            # 2. 최소 유사도 임계값 체크
            min_threshold = request.min_similarity or settings.MIN_SIMILARITY_THRESHOLD
            if similarity_score < min_threshold:
                logger.debug(f"상품 {product_id} 유사도 임계값 미달: {similarity_score:.3f} < {min_threshold}")
                continue
            
            # 3. 가격 필터만 적용 (비즈니스 룰)
            if not self._passes_price_filter(product, request.max_price):
                logger.debug(f"상품 {product_id} 가격 필터 실패")
                continue
            
            # 4. 순수 임베딩 점수가 최종 점수 (AI가 모든 것을 고려함)
            final_score = similarity_score
            
            # 5. AI가 찾은 의미적 연관성 설명 생성
            recommendation_reason = self._generate_ai_recommendation_reason(
                similarity_score, 
                product,
                details,
                request.user_diagnosis
            )
            
            recommendations.append({
                "product_id": product_id,
                "similarity_score": similarity_score,
                "final_score": final_score,  # 순수 AI 점수
                "ai_confidence": self._calculate_ai_confidence(similarity_score, i),
                "ranking_position": len(recommendations) + 1,
                "recommendation_reason": recommendation_reason,
                "confidence_level": self._determine_confidence_level(final_score),
                "category_path": details["category_path"],
                "price_range": details["price_range"],
                "matched_keywords": [],  # 키워드 매칭 제거
                "keyword_boost": 0.0,    # 키워드 부스트 제거
                "ai_method": "pure_embedding"  # AI 방식 표시
            })
            
            logger.debug(f"✅ 상품 {product_id} AI 추천 (유사도: {similarity_score:.3f})")
        
        # AI가 계산한 유사도 순으로 이미 정렬되어 있음
        final_recommendations = recommendations[:request.top_k]
        
        logger.info(f"🧠 AI 추천 결과: 평균 유사도 {np.mean([r['similarity_score'] for r in final_recommendations]):.3f}")
        return final_recommendations
    
    def _calculate_ai_confidence(self, similarity_score: float, rank: int) -> float:
        """AI 신뢰도 계산 - 유사도와 순위를 종합"""
        # 유사도가 높고 순위가 높을수록 신뢰도 증가
        base_confidence = similarity_score
        rank_penalty = min(0.1, rank * 0.01)  # 순위가 낮을수록 약간 감소
        return max(0.0, min(1.0, base_confidence - rank_penalty))
    
    def _generate_ai_recommendation_reason(
        self, 
        similarity_score: float, 
        product: Product,
        product_details: Dict[str, Any],
        user_diagnosis: str
    ) -> str:
        """AI 기반 추천 이유 생성 - 의미적 유사도 중심"""
        
        reasons = []
        
        # AI 신뢰도 기반 설명
        if similarity_score > 0.8:
            reasons.append("AI가 분석한 결과 고객님의 니즈와 매우 높은 일치도를 보이는")
        elif similarity_score > 0.6:
            reasons.append("AI가 분석한 결과 고객님의 요구사항과 잘 맞는")
        elif similarity_score > 0.4:
            reasons.append("AI가 분석한 결과 고객님에게 적합한")
        else:
            reasons.append("AI가 추천하는")
        
        # 상품 카테고리
        category_path = product_details.get("category_path", "")
        if category_path:
            main_category = category_path.split(" > ")[0]
            reasons.append(f"{main_category}")
        
        # 가격대 정보
        price_range = product_details.get("price_range", "")
        if price_range and "가격미정" not in price_range:
            reasons.append(f"제품입니다. {price_range}")
        else:
            reasons.append("제품입니다")
        
        # AI 학습된 패턴 언급
        if similarity_score > 0.7:
            reasons.append("(AI 학습 데이터에서 유사한 니즈의 고객들이 선호한 패턴과 일치)")
        
        return " ".join(reasons)
    
    def _passes_price_filter(self, product: Product, max_price: Optional[float]) -> bool:
        """가격 필터만 적용 (비즈니스 룰)"""
        if max_price is None or max_price == 0:
            return True
        return float(product.base_price) <= max_price
    
    def _convert_faiss_score_to_similarity(self, raw_score: float) -> float:
        """Faiss 원시 점수를 정규화된 유사도로 변환"""
        index_type = self.vector_store.index_manager.index_type
        
        if index_type == "IndexFlatIP":
            # 내적 점수 (코사인 유사도) - 이미 -1~1 범위, 0~1로 정규화
            normalized = (raw_score + 1) / 2  # -1~1 → 0~1
            return float(np.clip(normalized, 0, 1))
        else:
            # L2 거리 - 거리를 유사도로 변환
            return float(1 / (1 + raw_score))
    
    def _determine_confidence_level(self, final_score: float) -> str:
        """AI 신뢰도 수준 결정"""
        if final_score > 0.8:
            return "high"
        elif final_score > 0.6:
            return "medium"
        else:
            return "low"
    
    # (카테고리 필터링, DB 조회 등)
    async def _get_product_details(
        self, 
        product_ids: List[int],
        include_categories: Optional[List[ProductCategory]] = None,
        exclude_categories: Optional[List[ProductCategory]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """실제 DB에서 상품 상세 정보 조회 """
        try:
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(DBProduct)
                    .options(
                        selectinload(DBProduct.product_options),
                        selectinload(DBProduct.product_categories).selectinload(DBProductCategory.category)
                    )
                    .where(DBProduct.id.in_(product_ids))
                    .where(DBProduct.status == "ACTIVE")
                )
                
                if include_categories or exclude_categories:
                    stmt = self._apply_category_filter_to_query(
                        stmt, include_categories, exclude_categories
                    )

                result = await db.execute(stmt)
                db_products = result.scalars().all()
                
                logger.info(f"📊 DB 쿼리 결과: {len(db_products)}개 상품")

                product_details = {}
                for db_product in db_products:
                    try:
                        product = await self.product_converter.db_to_pydantic(db, db_product)

                        if not self._passes_category_filter(product, include_categories, exclude_categories):
                            continue
                        
                        details = {
                            "product": product,
                            "category_path": await self._get_category_path(db, db_product.id),
                            "price_range": self._get_actual_price_range(db_product),
                            "key_ingredients": self._extract_actual_ingredients(db_product),
                            "skin_types": self._extract_skin_compatibility(product.description),
                            "benefits": self._extract_product_benefits(product.description)
                        }
                        
                        product_details[db_product.id] = details
                        
                    except Exception as e:
                        logger.error(f"상품 {db_product.id} 상세 정보 추출 실패: {e}")
                        continue
                
                return product_details
                
        except Exception as e:
            logger.error(f"상품 상세 정보 조회 실패: {e}")
            return {}
    
    def _apply_category_filter_to_query(
        self,
        stmt,
        include_categories: Optional[List[ProductCategory]] = None,
        exclude_categories: Optional[List[ProductCategory]] = None
    ):
        """카테고리 필터 적용 (비즈니스 룰)"""
        
        if include_categories and exclude_categories:
            include_names = [cat.value for cat in include_categories]
            stmt = stmt.join(DBProductCategory).join(DBCategory).where(
                DBCategory.name.in_(include_names)
            )
            return stmt
        
        elif include_categories:
            include_names = [cat.value for cat in include_categories]
            stmt = stmt.join(DBProductCategory).join(DBCategory).where(
                DBCategory.name.in_(include_names)
            )
            return stmt
        
        elif exclude_categories:
            exclude_names = [cat.value for cat in exclude_categories]
            exclude_subquery = (
                select(DBProductCategory.product_id)
                .join(DBCategory)
                .where(DBCategory.name.in_(exclude_names))
            )
            stmt = stmt.where(not_(DBProduct.id.in_(exclude_subquery)))
            return stmt

        return stmt

    def _passes_category_filter(
        self,
        product: Product,
        include_categories: Optional[List[ProductCategory]] = None,
        exclude_categories: Optional[List[ProductCategory]] = None
    ) -> bool:
        """메모리 레벨 카테고리 필터"""
        if include_categories:
            if product.category_main not in include_categories:
                return False
        if exclude_categories:
            if product.category_main in exclude_categories:
                return False
        return True
    
    async def _fallback_recommendation(
        self, 
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """AI 추천 실패 시 Fallback"""
        logger.info("🔄 AI 기반 Fallback 추천 실행")
        
        try:
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(DBProduct)
                    .options(selectinload(DBProduct.product_options))
                    .where(DBProduct.status == "ACTIVE")
                )

                if request.include_categories or request.exclude_categories:
                    stmt = self._apply_category_filter_to_query(
                        stmt, request.include_categories, request.exclude_categories
                    )
                
                stmt = stmt.limit(50)
                result = await db.execute(stmt)
                db_products = result.scalars().all()

                if not db_products:
                    return []
                
                fallback_results = []
                processed_count = 0

                for db_product in db_products:
                    try:
                        product = await self.product_converter.db_to_pydantic(db, db_product)

                        if not self._passes_category_filter(
                            product, request.include_categories, request.exclude_categories
                        ):
                            continue

                        if request.max_price and request.max_price > 0 and db_product.product_options:
                            active_options = [opt for opt in db_product.product_options if not opt.is_deleted]
                            if active_options:
                                min_price = min(opt.price for opt in active_options)
                                if min_price > request.max_price:
                                    continue
                        
                        fallback_results.append({
                            "product_id": db_product.id,
                            "similarity_score": 0.4 - (processed_count * 0.02),
                            "final_score": 0.4 - (processed_count * 0.02),
                            "ai_confidence": 0.3,
                            "ranking_position": processed_count + 1,
                            "recommendation_reason": f"AI 학습 패턴 기반 {product.category_main.value} 추천 제품",
                            "confidence_level": "low",
                            "category_path": f"{product.category_main.value} > {product.category_sub}",
                            "price_range": self._get_actual_price_range(db_product),
                            "matched_keywords": [],
                            "keyword_boost": 0.0,
                            "ai_method": "fallback"
                        })

                        processed_count += 1
                        if processed_count >= request.top_k:
                            break
                        
                    except Exception as e:
                        logger.error(f"Fallback 상품 {db_product.id} 처리 실패: {e}")
                        continue
                
                return fallback_results
                
        except Exception as e:
            logger.error(f"AI Fallback 추천 실패: {e}")
            return []
    
    # 유틸리티 메서드들 (변경 없음)
    async def _get_category_path(self, db: AsyncSession, product_id: int) -> str:
        try:
            stmt = (
                select(DBCategory.name, DBCategory.depth, DBCategory.path)
                .select_from(DBProductCategory)
                .join(DBCategory, DBProductCategory.category_id == DBCategory.id)
                .where(DBProductCategory.product_id == product_id)
                .order_by(DBCategory.depth.asc())
            )
            result = await db.execute(stmt)
            categories = result.fetchall()
            if not categories:
                return "기타"
            category_names = [cat.name for cat in categories]
            return " > ".join(category_names)
        except Exception as e:
            logger.error(f"카테고리 경로 조회 실패: {e}")
            return "기타"
    
    def _get_actual_price_range(self, db_product: DBProduct) -> str:
        try:
            if not db_product.product_options:
                return "가격미정"
            active_options = [opt for opt in db_product.product_options if not opt.is_deleted]
            if not active_options:
                return "가격미정"
            prices = [opt.price for opt in active_options]
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            price_range = self.product_tower_service._get_price_range(avg_price)
            return f"{price_range} ({min_price:,}원-{max_price:,}원)"
        except Exception as e:
            logger.error(f"가격대 계산 실패: {e}")
            return "가격미정"
    
    def _extract_actual_ingredients(self, db_product: DBProduct) -> List[str]:
        try:
            all_ingredients = []
            for option in db_product.product_options:
                if option.full_ingredients and not option.is_deleted:
                    all_ingredients.append(option.full_ingredients)
            if not all_ingredients:
                return []
            full_ingredients = max(all_ingredients, key=len)
            key_ingredients = self.product_tower_service._extract_key_ingredients(full_ingredients)
            return key_ingredients
        except Exception as e:
            logger.error(f"성분 추출 실패: {e}")
            return []
    
    def _extract_skin_compatibility(self, description: str) -> List[str]:
        if not description:
            return ["모든피부"]
        class TempProduct:
            def __init__(self, description):
                self.description = description
                self.category_main = None
        temp_product = TempProduct(description)
        skin_compatibility = self.product_tower_service._extract_skin_compatibility(temp_product)
        return skin_compatibility if skin_compatibility else ["모든피부"]
    
    def _extract_product_benefits(self, description: str) -> List[str]:
        if not description:
            return []
        benefits = self.product_tower_service._extract_benefits(description)
        return benefits
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """AI 추천 서비스 통계"""
        return {
            "service_name": "PureEmbeddingRecommendationService",
            "version": "3.0 (순수 AI 임베딩)",
            "approach": "pure_semantic_similarity",
            "vector_store_stats": self.vector_store.get_store_stats(),
            "embedding_model": self.embedding_service.get_model_info(),
            "ai_features": {
                "semantic_understanding": True,
                "contextual_matching": True,
                "pure_embedding_scoring": True,
                "no_manual_keywords": True,
                "deep_learning_based": True
            },
            "removed_features": {
                "keyword_extraction": False,
                "manual_boost_scoring": False,
                "rule_based_matching": False,
                "hybrid_scoring": False
            },
            "scoring_method": {
                "primary": "cosine_similarity",
                "weights": "AI_learned_patterns_only",
                "manual_rules": "minimal (price/category only)"
            }
        }