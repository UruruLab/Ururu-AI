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
from app.services.user_tower_service import UserTowerService
from app.services.product_converter import ProductConverter
from app.models.recommendation import ProfileBasedRecommendationRequest
from app.models.product import Product, ProductCategory
from app.models.user import BeautyProfile
from app.models.database import DBProduct, DBProductOption, DBCategory, DBProductCategory
from app.core.config import settings
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class RecommendationService:
    """프로필 기반 추천 서비스 - BeautyProfile 전용"""
    
    def __init__(
        self, 
        vector_store: FaissVectorStore,
        embedding_service: EmbeddingService,
        product_tower_service: ProductTowerService,
        user_tower_service: UserTowerService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.product_tower_service = product_tower_service
        self.user_tower_service = user_tower_service
        self.product_converter = ProductConverter()
        logger.info("🎯 프로필 기반 추천 서비스 초기화 완료")
    
    async def recommend_products(
        self, 
        request: ProfileBasedRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """BeautyProfile 기반 상품 추천 (메인 추천 방식)"""
        
        logger.info(f"👤 프로필 기반 추천 시작: {request.beauty_profile.skin_type.value} {request.beauty_profile.skin_tone.value}")
        
        try:
            # 1. 사용자 프로필을 임베딩으로 변환
            user_embedding = self.user_tower_service.generate_user_embedding(request.beauty_profile)
            logger.debug(f"🧠 사용자 임베딩 생성 완료: {len(user_embedding)}차원")
            
            # 2. Faiss 벡터 검색
            search_multiplier = 3 if (request.include_categories or request.exclude_categories) else 2
            search_k = min(request.top_k * search_multiplier, 100)

            raw_scores, product_ids = await self.vector_store.search_vectors(
                user_embedding.tolist(), 
                search_k 
            )
            
            if not product_ids:
                logger.warning("벡터 검색 결과 없음")
                return await self._fallback_recommendation(request)
            
            logger.info(f"🔍 벡터 검색 완료: {len(product_ids)}개 상품")
            
            # 3. 카테고리 및 가격 필터링 적용
            product_details = await self._get_product_details(
                product_ids,
                request.include_categories,
                request.exclude_categories
            )

            if not product_details:
                logger.warning("필터링 후 결과 없음 - Fallback 실행")
                return await self._fallback_recommendation(request)
            
            logger.info(f"📊 필터링 후: {len(product_details)}개 상품")

            # 4. 프로필 기반 추천 로직 적용
            recommendations = await self._create_recommendations(
                raw_scores, 
                product_ids, 
                product_details,
                request
            )
            
            logger.info(f"✅ 프로필 기반 추천 완료: {len(recommendations)}개 상품")
            return recommendations
            
        except Exception as e:
            logger.error(f"프로필 기반 추천 실패: {e}")
            return await self._fallback_recommendation(request)
    
    async def _create_recommendations(
        self, 
        raw_scores: List[float], 
        all_product_ids: List[int], 
        product_details: Dict[int, Dict[str, Any]],
        request: ProfileBasedRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """프로필 기반 추천 생성"""
        
        recommendations = []
        
        for i, (raw_score, product_id) in enumerate(zip(raw_scores, all_product_ids)):
            if product_id not in product_details:
                continue
            
            details = product_details[product_id]
            product = details["product"]
            
            # 1. Faiss 점수를 유사도로 변환
            similarity_score = self._convert_faiss_score_to_similarity(raw_score)
            
            # 2. 최소 유사도 임계값 체크
            min_threshold = request.min_similarity or settings.MIN_SIMILARITY_THRESHOLD
            if similarity_score < min_threshold:
                logger.debug(f"상품 {product_id} 유사도 임계값 미달: {similarity_score:.3f} < {min_threshold}")
                continue
            
            # 3. 가격 필터 적용 (프로필 기반)
            if not self._passes_price_filter(product, request.beauty_profile, request.use_price_filter):
                logger.debug(f"상품 {product_id} 가격 필터 실패")
                continue
            
            # 4. 프로필-상품 매칭 점수 계산
            profile_match_score = self._calculate_profile_match_score(
                request.beauty_profile, 
                product,
                details
            )
            
            # 5. 최종 점수 계산 (벡터 유사도 + 프로필 매칭)
            final_score = (similarity_score * 0.7 + profile_match_score * 0.3)
            
            # 6. 매칭된 특성 추출
            matched_features = self._extract_matched_features(
                request.beauty_profile,
                product,
                details
            )
            
            # 7. 추천 이유 생성
            recommendation_reason = self._generate_recommendation_reason(
                similarity_score, 
                profile_match_score,
                matched_features,
                request.beauty_profile,
                product,
                details
            )
            
            recommendations.append({
                "product_id": product_id,
                "similarity_score": similarity_score,
                "profile_match_score": profile_match_score,
                "final_score": final_score,
                "confidence_score": self._calculate_confidence_score(similarity_score, profile_match_score, i),
                "ranking_position": len(recommendations) + 1,
                "recommendation_reason": recommendation_reason,
                "matched_features": matched_features,
                "confidence_level": self._determine_confidence_level(final_score),
                "category_path": details["category_path"],
                "price_range": details["price_range"],
                "recommendation_method": "profile_based"
            })
            
            logger.debug(f"✅ 상품 {product_id} 프로필 추천 (유사도: {similarity_score:.3f}, 매칭: {profile_match_score:.3f})")
        
        # 최종 점수로 정렬
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 요청된 개수만큼 반환
        final_recommendations = recommendations[:request.top_k]
        
        logger.info(f"🎯 프로필 추천 결과: 평균 유사도 {np.mean([r['similarity_score'] for r in final_recommendations]):.3f}")
        return final_recommendations
    
    def _calculate_profile_match_score(
        self, 
        beauty_profile: BeautyProfile,
        product: Product,
        product_details: Dict[str, Any]
    ) -> float:
        """프로필과 상품 간의 매칭 점수 계산"""
        
        match_score = 0.0
        
        # 1. 피부 타입 매칭
        skin_compatibility = product_details.get("skin_types", [])
        if beauty_profile.skin_type.value in skin_compatibility or "모든피부" in skin_compatibility:
            match_score += 0.3
        
        # 2. 피부 고민 매칭
        if beauty_profile.concerns:
            product_benefits = product_details.get("benefits", [])
            concern_matches = 0
            for concern in beauty_profile.concerns:
                if any(concern.lower() in benefit.lower() for benefit in product_benefits):
                    concern_matches += 1
            
            if concern_matches > 0:
                match_score += 0.4 * (concern_matches / len(beauty_profile.concerns))
        
        # 3. 관심 카테고리 매칭
        if beauty_profile.interest_categories:
            category_path = product_details.get("category_path", "").lower()
            for interest_category in beauty_profile.interest_categories:
                if interest_category.lower() in category_path:
                    match_score += 0.2
                    break
        
        # 4. 알레르기 성분 체크 (감점)
        if beauty_profile.has_allergy and beauty_profile.allergies:
            key_ingredients = product_details.get("key_ingredients", [])
            ingredient_text = " ".join(key_ingredients).lower()
            
            for allergy in beauty_profile.allergies:
                if allergy.lower() in ingredient_text:
                    match_score -= 0.3
                    break
        
        # 5. 가격대 선호도 매칭
        product_price = float(product.base_price)
        if beauty_profile.min_price <= product_price <= beauty_profile.max_price:
            match_score += 0.1
        
        return max(0.0, min(1.0, match_score))
    
    def _extract_matched_features(
        self,
        beauty_profile: BeautyProfile,
        product: Product,
        product_details: Dict[str, Any]
    ) -> List[str]:
        """프로필과 매칭된 특성들 추출"""
        
        matched = []
        
        # 피부 타입 매칭
        skin_compatibility = product_details.get("skin_types", [])
        if beauty_profile.skin_type.value in skin_compatibility:
            matched.append(f"{beauty_profile.skin_type.value} 적합")
        
        # 피부 고민 매칭
        if beauty_profile.concerns:
            product_benefits = product_details.get("benefits", [])
            for concern in beauty_profile.concerns:
                for benefit in product_benefits:
                    if concern.lower() in benefit.lower():
                        matched.append(f"{concern} 케어")
                        break
        
        # 관심 카테고리 매칭
        if beauty_profile.interest_categories:
            category_path = product_details.get("category_path", "")
            for interest_category in beauty_profile.interest_categories:
                if interest_category.lower() in category_path.lower():
                    matched.append(f"{interest_category} 관심사")
                    break
        
        # 가격대 매칭
        product_price = float(product.base_price)
        if beauty_profile.min_price <= product_price <= beauty_profile.max_price:
            matched.append("가격대 적합")
        
        return matched[:5]
    
    def _generate_recommendation_reason(
        self,
        similarity_score: float,
        profile_match_score: float,
        matched_features: List[str],
        beauty_profile: BeautyProfile,
        product: Product,
        product_details: Dict[str, Any]
    ) -> str:
        """프로필 기반 추천 이유 생성"""
        
        reasons = []
        
        # AI 유사도 기반
        if similarity_score > 0.8:
            reasons.append("AI 분석 결과 매우 높은 적합도를 보이며")
        elif similarity_score > 0.6:
            reasons.append("AI 분석 결과 높은 적합도를 보이고")
        else:
            reasons.append("AI 분석 결과 적절한 적합도를 보이며")
        
        # 프로필 매칭 기반
        if profile_match_score > 0.7:
            reasons.append("프로필 분석 결과 매우 잘 맞는")
        elif profile_match_score > 0.5:
            reasons.append("프로필 분석 결과 잘 맞는")
        else:
            reasons.append("프로필에 적합한")
        
        # 매칭된 특성
        if matched_features:
            if len(matched_features) >= 3:
                feature_str = ", ".join(matched_features[:3])
                reasons.append(f"'{feature_str}' 등의 특성이 매칭되어")
            else:
                feature_str = ", ".join(matched_features)
                reasons.append(f"'{feature_str}' 특성이 매칭되어")
        
        # 카테고리 정보
        reasons.append(f"{product.category_main.value} 제품으로 추천합니다")
        
        return " ".join(reasons)
    
    def _passes_price_filter(
        self, 
        product: Product, 
        beauty_profile: BeautyProfile,
        use_price_filter: bool
    ) -> bool:
        """프로필 기반 가격 필터"""
        if not use_price_filter:
            return True
        
        product_price = float(product.base_price)
        return beauty_profile.min_price <= product_price <= beauty_profile.max_price
    
    def _calculate_confidence_score(self, similarity_score: float, profile_match_score: float, rank: int) -> float:
        """신뢰도 점수 계산"""
        base_confidence = (similarity_score * 0.6 + profile_match_score * 0.4)
        rank_penalty = min(0.1, rank * 0.01)
        return max(0.0, min(1.0, base_confidence - rank_penalty))
    
    def _convert_faiss_score_to_similarity(self, raw_score: float) -> float:
        """Faiss 원시 점수를 정규화된 유사도로 변환"""
        index_type = self.vector_store.index_manager.index_type
        
        if index_type == "IndexFlatIP":
            normalized = (raw_score + 1) / 2
            return float(np.clip(normalized, 0, 1))
        else:
            return float(1 / (1 + raw_score))
    
    def _determine_confidence_level(self, final_score: float) -> str:
        """신뢰도 수준 결정"""
        if final_score > 0.8:
            return "high"
        elif final_score > 0.6:
            return "medium"
        else:
            return "low"
    
    # 공통 유틸리티 메서드들
    async def _get_product_details(
        self, 
        product_ids: List[int],
        include_categories: Optional[List[ProductCategory]] = None,
        exclude_categories: Optional[List[ProductCategory]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """실제 DB에서 상품 상세 정보 조회"""
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
    
    def _apply_category_filter_to_query(self, stmt, include_categories, exclude_categories):
        """카테고리 필터 적용"""
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

    def _passes_category_filter(self, product, include_categories, exclude_categories):
        """메모리 레벨 카테고리 필터"""
        if include_categories:
            if product.category_main not in include_categories:
                return False
        if exclude_categories:
            if product.category_main in exclude_categories:
                return False
        return True
    
    async def _get_category_path(self, db: AsyncSession, product_id: int) -> str:
        """상품의 전체 카테고리 경로 조회"""
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
        """실제 상품 가격대 계산"""
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
        """실제 상품의 주요 성분 추출"""
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
        """실제 상품 설명에서 피부타입 호환성 추출"""
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
        """실제 상품 설명에서 효능 추출"""
        if not description:
            return []
        
        benefits = self.product_tower_service._extract_benefits(description)
        return benefits
    
    async def _fallback_recommendation(self, request: ProfileBasedRecommendationRequest) -> List[Dict[str, Any]]:
        """프로필 기반 Fallback 추천"""
        logger.info("🔄 프로필 기반 Fallback 추천 실행")
        
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

                        if request.use_price_filter:
                            product_price = float(product.base_price)
                            if not (request.beauty_profile.min_price <= product_price <= request.beauty_profile.max_price):
                                continue
                        
                        fallback_results.append({
                            "product_id": db_product.id,
                            "similarity_score": 0.4 - (processed_count * 0.02),
                            "profile_match_score": 0.3,
                            "final_score": 0.35 - (processed_count * 0.02),
                            "confidence_score": 0.3,
                            "ranking_position": processed_count + 1,
                            "recommendation_reason": f"프로필 기반 일반 {product.category_main.value} 추천 제품",
                            "matched_features": [],
                            "confidence_level": "low",
                            "category_path": f"{product.category_main.value} > {product.category_sub}",
                            "price_range": self._get_actual_price_range(db_product),
                            "recommendation_method": "profile_fallback"
                        })

                        processed_count += 1
                        if processed_count >= request.top_k:
                            break
                        
                    except Exception as e:
                        logger.error(f"Fallback 상품 {db_product.id} 처리 실패: {e}")
                        continue
                
                return fallback_results
                
        except Exception as e:
            logger.error(f"프로필 기반 Fallback 추천 실패: {e}")
            return []
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """프로필 기반 추천 서비스 통계"""
        return {
            "service_name": "ProfileBasedRecommendationService",
            "version": "4.0 (프로필 전용)",
            "approach": "beauty_profile_vector_similarity",
            "vector_store_stats": self.vector_store.get_store_stats(),
            "embedding_model": self.embedding_service.get_model_info(),
            "features": {
                "profile_based_recommendation": True,
                "user_profile_embedding": True,
                "product_vector_similarity": True,
                "profile_feature_matching": True,
                "skin_type_compatibility": True,
                "concern_benefit_matching": True,
                "allergy_ingredient_check": True,
                "price_range_filtering": True
            },
            "scoring_method": {
                "vector_similarity_weight": 0.7,
                "profile_match_weight": 0.3,
                "confidence_calculation": "similarity + match - rank_penalty"
            },
            "data_sources": {
                "user_profile": "BeautyProfile (structured)",
                "product_embeddings": "Faiss Vector Store",
                "product_details": "DB (products, ingredients, benefits)"
            }
        }