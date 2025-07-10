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
        self.product_converter = ProductConverter()
        logger.debug("🎯 추천 서비스 초기화 완료 (실제 DB 연동)")
    
    async def recommend_products(
        self, 
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """메인 상품 추천 로직 - 벡터 검색 + 실제 DB 연동"""
        
        logger.info(f"🔍 상품 추천 시작: '{request.user_diagnosis[:30]}...'")
        
        try:
            # 1. 사용자 진단을 임베딩으로 변환
            user_embedding = self.embedding_service.encode_text(request.user_diagnosis)
            
            # 2. Faiss 벡터 검색 (순수 검색)
            search_multiplier = 3 if (request.include_categories or request.exclude_categories) else 2
            search_k = min(request.top_k * search_multiplier, 100)

            raw_scores, product_ids = await self.vector_store.search_vectors(
                user_embedding, 
                search_k 
            )
            
            if not product_ids:
                logger.warning("벡터 검색 결과 없음")
                return await self._fallback_recommendation(request)
            
            logger.debug(f"🔎 벡터 검색 완료: {len(product_ids)}개 상품 ID")
            
            # 3. 실제 DB에서 상품 정보 조회
            product_details = await self._get_product_details(
                product_ids,
                request.include_categories,
                request.exclude_categories
            )

            if not product_details:
                logger.warning("카테고리 필터링 후 검색 결과 없음 - Fallback 실행")
                return await self._fallback_recommendation(request)
            
            logger.info(f"카테고리 필터링 후 : {len(product_details)}개 상품")

            # 4. 벡터 검색 점수를 필터링된 상품에만 매핑
            filtered_scores, filtered_product_ids = self._map_scores_to_filtered_products(
                raw_scores, product_ids, list(product_details.keys())
            )
            
            # 5. 비즈니스 로직 적용 (점수 변환, 필터링, 랭킹)
            recommendations = await self._apply_recommendation_logic(
                filtered_scores, 
                filtered_product_ids, 
                product_details,
                request
            )
            
            logger.info(f"✅ 추천 완료: {len(recommendations)}개 상품")
            return recommendations
            
        except Exception as e:
            logger.error(f"추천 실패: {e}")
            return await self._fallback_recommendation(request)
    
    async def _get_product_details(
            self, 
            product_ids: List[int],
            include_categories: Optional[List[ProductCategory]] = None,
            exclude_categories: Optional[List[ProductCategory]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """실제 DB에서 상품 상세 정보 조회"""
        try:
            async with AsyncSessionLocal() as db:
                # 상품 기본 정보 + 옵션 + 카테고리 정보 조회
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
                
                logger.debug(f" DB 쿼리 결과 : {len(db_products)}개 상품")

                if not db_product:
                    logger.warning(f"🔍 DB 쿼리 결과 없음 - 디버깅 정보:")
                    logger.warning(f"  - 검색 대상 상품 ID: {product_ids[:10]}...")
                    logger.warning(f"  - Include 카테고리: {[cat.value for cat in include_categories] if include_categories else None}")
                    logger.warning(f"  - Exclude 카테고리: {[cat.value for cat in exclude_categories] if exclude_categories else None}")

                    basic_stmt = (
                        select(DBProduct)
                        .where(DBProduct.id.in_(product_ids[:5]))
                        .where(DBProduct.status == "ACTIVE")
                    )
                    basic_result = await db.execute(basic_stmt)
                    basic_products = basic_result.scalar().all()
                    logger.warning(f"   - 필터 없이 조회한 상품 수: {len(basic_products)}개")
                
                if basic_products:
                    first_product = basic_products[0]
                    catetory_stmt = (
                        select(DBCategory.name)
                        .select_from(DBProductCategory)
                        .join(DBCategory)
                        .where(DBProductCategory.product_id == first_product.id)
                    )
                    catetory_result = await db.execute(catetory_stmt)
                    categories = [row[0] for row in catetory_result.fetchall()]
                    logger.warning(f"    - 첫 번째 상품({first_product.id})의 카테고리: {categories}")


                product_details = {}
                for db_product in db_products:
                    try:
                        # Pydantic 모델로 변환
                        product = await self.product_converter.db_to_pydantic(db, db_product)

                        # 한번 더 카테고리 필터 확인 (추후 삭제해도 무방)
                        if not self._passes_category_filter(product, include_categories, exclude_categories):
                            logger.debug(f"상품 {product.id} 메모리 레벨 카테고리 필터 실패")
                            continue
                        
                        # 추가 상세 정보 추출
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
                
                logger.debug(f"📊 상품 상세 정보 조회 완료: {len(product_details)}개")
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
        if include_categories and exclude_categories:
            logger.debug("include + exclude 카테고리 동시 적용")
            include_names = [cat.value for cat in include_categories]
            stmt = stmt.join(DBProductCategory).join(DBCategory).where(
                DBCategory.name.in_(include_names)
            )
            return stmt
        
        elif include_categories:
            include_names = [cat.value for cat in include_categories]
            logger.debug(f"Include 카테고리만 적용: {include_names}")
            stmt = stmt.join(DBProductCategory).join(DBCategory).where(
                DBCategory.name.in_(include_names)
            )
            return stmt
        
        if exclude_categories:
            exclude_names = [cat.value for cat in exclude_categories]
            logger.debug(f"Exclude 카테고리 적용: {exclude_names}")
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
        if include_categories:
            if product.category_main not in include_categories:
                return False
        if exclude_categories:
            if product.category_main in exclude_categories:
                return False
        return True
    
    def _map_scores_to_filtered_products(
            self,
            raw_scored: List[float],
            all_product_ids: List[int],
            filtered_product_ids: List[int]
    ) -> Tuple[List[float], List[int]]:
        filtered_scores = []
        final_product_ids = []

        for score, product_id in zip(raw_scored, all_product_ids):
            if product_id in filtered_product_ids:
                final_product_ids.append(product_id)

        return filtered_scores, final_product_ids
    
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
            
            # 카테고리 경로 구성 (메인 > 서브 > 상세)
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
            
            # 활성화된 옵션들의 가격 범위 계산
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
    
    async def _apply_recommendation_logic(
        self, 
        raw_scores: List[float], 
        product_ids: List[int], 
        product_details: Dict[int, Dict[str, Any]],
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """실제 상품 정보를 활용한 비즈니스 로직 적용"""
        
        recommendations = []
        
        for i, (raw_score, product_id) in enumerate(zip(raw_scores, product_ids)):
            if product_id not in product_details:
                continue
            
            details = product_details[product_id]
            product = details["product"]
            
            # 1. Faiss 점수를 유사도로 변환
            similarity_score = self._convert_faiss_score_to_similarity(raw_score)
            
            # 2. 최소 유사도 임계값 체크
            if similarity_score < (request.min_similarity or settings.MIN_SIMILARITY_THRESHOLD):
                continue
            
            # 3. 실제 키워드 매칭 보정 점수 계산
            keyword_boost = self._calculate_real_keyword_boost(
                request.user_diagnosis, 
                product,
                details["key_ingredients"],
                details["benefits"]
            )
            
            # 5. 실제 가격 필터 적용
            if not self._passes_real_price_filter(product, request.max_price):
                continue
            
            # 6. 최종 추천 점수 계산
            final_score = (similarity_score * 0.6 + keyword_boost * 0.4)
            
            # 7. 매칭된 키워드 추출
            matched_keywords = self._extract_matched_keywords(
                request.user_diagnosis,
                product.description,
                details["key_ingredients"],
                details["benefits"]
            )
            
            recommendations.append({
                "product_id": product_id,
                "similarity_score": similarity_score,
                "keyword_boost": keyword_boost,
                "final_score": final_score,
                "matched_keywords": matched_keywords,
                "ranking_position": len(recommendations) + 1,
                "recommendation_reason": self._generate_real_recommendation_reason(
                    similarity_score, 
                    keyword_boost, 
                    matched_keywords,
                    details
                ),
                "confidence_level": self._determine_confidence_level(final_score),
                "category_path": details["category_path"],
                "price_range": details["price_range"]
            })
        
        # 최종 점수로 정렬
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 요청된 개수만큼 반환
        return recommendations[:request.top_k]
    
    def _calculate_real_keyword_boost(
        self, 
        user_diagnosis: str, 
        product: Product,
        key_ingredients: List[str],
        benefits: List[str]
    ) -> float:
        """실제 상품 정보를 활용한 키워드 매칭 점수"""
        
        user_keywords = self._extract_beauty_keywords(user_diagnosis.lower())
        
        boost_score = 0.0
        match_count = 0
        
        # 사용자 키워드와 상품 정보 매칭
        product_text = (product.description + " " + " ".join(key_ingredients) + " " + " ".join(benefits)).lower()
        
        for user_keyword in user_keywords:
            if user_keyword in product_text:
                # 키워드별 가중치 적용
                if user_keyword in ["수분", "보습"]:
                    boost_score += 0.9
                elif user_keyword in ["민감", "진정"]:
                    boost_score += 0.8
                elif user_keyword in ["트러블", "여드름"]:
                    boost_score += 0.8
                elif user_keyword in ["미백", "브라이트닝"]:
                    boost_score += 0.7
                elif user_keyword in ["주름", "안티에이징"]:
                    boost_score += 0.7
                else:
                    boost_score += 0.5
                    
                match_count += 1
        
        # 매칭된 키워드 수에 따른 보정
        if match_count == 0:
            return 0.2  # 기본 점수
        elif match_count >= 3:
            return min(1.0, boost_score / match_count * 1.2)  
        else:
            return min(1.0, boost_score / match_count)
   
    
    def _passes_real_price_filter(self, product: Product, max_price: Optional[float]) -> bool:
        """실제 상품 가격 필터 적용"""
        if max_price is None:
            return True
        
        # 상품의 기본 가격이 최대 가격 이하인지 확인
        return float(product.base_price) <= max_price
    
    def _extract_matched_keywords(
        self,
        user_diagnosis: str,
        product_description: str,
        key_ingredients: List[str],
        benefits: List[str]
    ) -> List[str]:
        """실제 매칭된 키워드 추출"""
        
        user_keywords = self._extract_beauty_keywords(user_diagnosis.lower())
        product_text = (product_description + " " + " ".join(key_ingredients) + " " + " ".join(benefits)).lower()
        
        matched = []
        for keyword in user_keywords:
            if keyword in product_text:
                matched.append(keyword)
        
        return matched[:5] 
    
    def _generate_real_recommendation_reason(
        self, 
        similarity_score: float, 
        keyword_boost: float, 
        matched_keywords: List[str],
        product_details: Dict[str, Any]
    ) -> str:
        """실제 상품 정보를 활용한 추천 이유 생성"""
        
        reasons = []
        
        # AI 유사도 기반
        if similarity_score > 0.8:
            reasons.append("AI 분석 결과 매우 높은 적합도를 보이며")
        elif similarity_score > 0.6:
            reasons.append("AI 분석 결과 높은 적합도를 보이며")
        else:
            reasons.append("AI 분석 결과 적절한 적합도를 보이며")
        
        # 키워드 매칭 기반
        if matched_keywords:
            if len(matched_keywords) >= 3:
                keyword_str = ", ".join(matched_keywords[:3])
                reasons.append(f"'{keyword_str}' 등 주요 키워드가 매우 잘 매칭되고")
            else:
                keyword_str = ", ".join(matched_keywords)
                reasons.append(f"'{keyword_str}' 키워드가 매칭되며")
        
        # 카테고리 정보
        category_path = product_details.get("category_path", "")
        if category_path:
            main_category = category_path.split(" > ")[0]
            reasons.append(f"{main_category} 카테고리의")
        
        # 가격대 정보
        price_range = product_details.get("price_range", "")
        if price_range and "가격미정" not in price_range:
            reasons.append(f"{price_range} 제품입니다")
        else:
            reasons.append("제품입니다")
        
        return " ".join(reasons)
    
    def _convert_faiss_score_to_similarity(self, raw_score: float) -> float:
        """Faiss 원시 점수를 0-1 유사도로 변환"""
        index_type = self.vector_store.index_manager.index_type
        
        if index_type == "IndexFlatIP":
            # 내적 점수 (코사인 유사도) - 이미 0-1 범위
            return float(np.clip(raw_score, 0, 1))
        else:
            # L2 거리 - 거리를 유사도로 변환
            return float(1 / (1 + raw_score))
    
    def _determine_confidence_level(self, final_score: float) -> str:
        """신뢰도 수준 결정"""
        if final_score > 0.8:
            return "high"
        elif final_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _extract_beauty_keywords(self, text: str) -> List[str]:
        """뷰티 키워드 추출 (기존 로직 재사용)"""
        return self.product_tower_service._extract_beauty_keywords(text)
    
    async def _fallback_recommendation(
        self, 
        request: ProductRecommendationRequest
    ) -> List[Dict[str, Any]]:
        """실제 DB 기반 Fallback 추천 로직"""
        logger.info("🔄 실제 DB 기반 Fallback 추천 실행")
        
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
                
                # if request.max_price:
                #     stmt = stmt.join(DBProductOption).where(
                #         and_(
                #             DBProductOption.price <= request.max_price,
                #             DBProductOption.is_deleted == False
                #         )
                #     )
                # stmt = stmt.limit(request.top_k)
                
                result = await db.execute(stmt)
                db_products = result.scalars().all()

                if not db_products:
                    logger.warning("카테고리 필터 후에도 결과 없음")
                    return []
                
                fallback_results = []
                processd_count = 0

                for i, db_product in enumerate(db_products):
                    try:
                        product = await self.product_converter.db_to_pydantic(db, db_product)

                        if not self._passes_category_filter(
                            product, request.include_categories, request.exclude_categories
                        ):
                            logger.debug(f"Fallback: 상품 {product.id} 카테고리 필터 실패")
                            continue

                        if request.max_price and db_product.product_options:
                            active_options = [opt for opt in db_product.product_options if not opt.is_deleted]
                            if active_options:
                                min_price = min(opt.price for opt in active_options)
                                if min_price > request.max_price:
                                    continue
                        
                        fallback_results.append({
                            "product_id": db_product.id,
                            "similarity_score": 0.4 - (processd_count * 0.02),
                            "keyword_boost": 0.3,
                            "final_score": 0.35 - (processd_count * 0.02),
                            "matched_keywords": [],
                            "ranking_position": processd_count + 1,
                            "recommendation_reason": self._generate_fallback_reason(product, request),
                            "confidence_level": "low",
                            "category_path": f"{product.category_main.value} > {product.category_sub}",
                            "price_range": self._get_actual_price_range(db_product),
                            "source": "database_fallback"
                        })

                        processd_count += 1
                        if processd_count >= request.top_k:
                            break
                        
                    except Exception as e:
                        logger.error(f"Fallback 상품 {db_product.id} 처리 실패: {e}")
                        continue
                
                return fallback_results
                
        except Exception as e:
            logger.error(f"DB 기반 Fallback 추천 실패: {e}")
            return []
        
    def _generate_fallback_reason(
            self,
            product: Product,
            request: ProductRecommendationRequest
    ) -> str:
        reasons = []

        if request.include_categories:
            if product.category_main in request.include_categories:
                reasons.append(f"요청하신 {product.category_main.value} 카테고리의")
        
        reasons.append("인기 제품으로")
        
        if request.max_price:
            reasons.append(f"에산 {request.max_price:,}원 내에 ")

        reasons.append("추천됩니다.")
        return " ".join(reasons)
    
    async def add_product_to_index(self, product_data: Dict) -> bool:
        """새 상품을 벡터 인덱스에 추가 (실제 Product 객체 활용)"""
        try:
            if isinstance(product_data, dict):
                processed_text = self._create_product_text_from_dict(product_data)
            else:
                processed_text = self.product_tower_service.preprocess_product_text(product_data)
            
            embedding = self.embedding_service.encode_text(processed_text)
            await self.vector_store.add_embeddings([{
                "product_id": product_data.get("id") if isinstance(product_data, dict) else product_data.id,
                "embedding": embedding,
                "metadata": {
                    "processed_text": processed_text[:200],
                    "created_at": datetime.now().isoformat()
                }
            }])
            
            logger.info(f"✅ 상품 벡터 인덱스 추가 완료")
            return True
            
        except Exception as e:
            logger.error(f"상품 벡터 인덱스 추가 실패: {e}")
            return False
    
    def _create_product_text_from_dict(self, product_data: Dict) -> str:
        """딕셔너리 형태의 상품 데이터를 텍스트로 변환"""
        components = []
        
        if product_data.get("name"):
            components.append(f"상품명: {product_data['name']}")
        
        if product_data.get("category_main"):
            components.append(f"카테고리: {product_data['category_main']}")
        
        if product_data.get("description"):
            components.append(f"설명: {product_data['description']}")
        
        if product_data.get("ingredients"):
            components.append(f"성분: {product_data['ingredients']}")
        
        return " | ".join(components)
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """추천 서비스 통계"""
        return {
            "service_name": "RecommendationService",
            "version": "2.0 (실제 DB 연동)",
            "vector_store_stats": self.vector_store.get_store_stats(),
            "embedding_model": self.embedding_service.get_model_info(),
            "algorithms": {
                "vector_search": True,
                "category_filtering": True,  
                "db_level_filtering": True,  
                "memory_level_filtering": True, 
                "real_keyword_matching": True,
                "price_filtering": True,
                "fallback_with_filtering": True  
            },
            "scoring_weights": {
                "vector_similarity": 0.6,
                "keyword_boost": 0.4,
                "diversity_penalty": "dynamic"
            },
            "data_sources": {
                "product_info": "DB (products, product_options)",
                "category_info": "DB (categories, product_categories)",
                "embeddings": "Faiss Vector Store",
                "fallback": "Database Query"
            }
        }