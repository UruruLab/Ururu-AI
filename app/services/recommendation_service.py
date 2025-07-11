import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, not_
from sqlalchemy.orm import selectinload

from app.services.faiss_service import FaissVectorStore
from app.services.embedding_service import EmbeddingService
from app.services.product_tower_service import ProductTowerService
from app.services.user_tower_service import UserTowerService
from app.services.product_converter import ProductConverter
from app.models.recommendation import ProfileBasedRecommendationRequest
from app.models.product import Product, ProductCategory
from app.models.user import BeautyProfile
from app.models.database import DBProduct, DBCategory, DBProductCategory
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
            
            # 2. 더 많은 후보 상품 검색
            if request.include_categories or request.exclude_categories:
                search_multiplier = 8 
            else:
                search_multiplier = 5 
            
            search_k = min(request.top_k * search_multiplier, 200) 

            raw_scores, product_ids = await self.vector_store.search_vectors(
                user_embedding.tolist(), 
                search_k 
            )
            
            if not product_ids:
                logger.warning("벡터 검색 결과 없음")
                return await self._fallback_recommendation(request)
            
            logger.info(f"🔍 벡터 검색 완료: {len(product_ids)}개 상품")
            
            # 3. 개선된 카테고리 필터링 - SQL 레벨에서만 처리
            product_details = await self._get_product_details_with_category_filter(
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
            
            # 5. 충분한 결과가 없으면 fallback과 결합
            if len(recommendations) < request.top_k:
                logger.info(f"⚠️ 결과 부족 ({len(recommendations)}/{request.top_k}), fallback 추가")
                fallback_results = await self._fallback_recommendation(request)
                
                existing_ids = {r['product_id'] for r in recommendations}
                for fallback in fallback_results:
                    if fallback['product_id'] not in existing_ids and len(recommendations) < request.top_k:
                        recommendations.append(fallback)
            
            logger.info(f"✅ 프로필 기반 추천 완료: {len(recommendations)}개 상품")
            return recommendations[:request.top_k]  # 최종적으로 요청한 개수만 반환
            
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
        """프로필 기반 추천 생성 - 개선된 버전"""
        
        recommendations = []
        debug_stats = {
            "total_candidates": len(all_product_ids),
            "similarity_filtered": 0,
            "price_filtered": 0,
            "category_filtered": 0,
            "final_recommendations": 0
        }
        
        logger.info(f"🔍 추천 생성 시작: {len(all_product_ids)}개 후보, {len(product_details)}개 상세정보")
        
        min_threshold = min(request.min_similarity or settings.MIN_SIMILARITY_THRESHOLD, 0.1)
        logger.info(f"📊 유사도 임계값: {min_threshold} (기존보다 관대하게 설정)")
        
        for i, (raw_score, product_id) in enumerate(zip(raw_scores, all_product_ids)):
            if product_id not in product_details:
                logger.debug(f"상품 {product_id}: 상세정보 없음")
                debug_stats["category_filtered"] += 1
                continue
            
            details = product_details[product_id]
            product = details["product"]
            
            # 1. Faiss 점수를 유사도로 변환
            similarity_score = self._convert_faiss_score_to_similarity(raw_score)
            logger.debug(f"상품 {product_id} ({product.name[:20]}): 원시점수={raw_score:.4f}, 유사도={similarity_score:.4f}")
            
            # 2. 최소 유사도 임계값 체크 
            if similarity_score < min_threshold:
                logger.debug(f"상품 {product_id} 유사도 임계값 미달: {similarity_score:.3f} < {min_threshold}")
                debug_stats["similarity_filtered"] += 1
                continue
            
            # 3. 가격 필터
            if request.use_price_filter:
                product_price = float(product.base_price)
                expanded_min = request.beauty_profile.min_price * 0.8
                expanded_max = request.beauty_profile.max_price * 1.2
                
                if not (expanded_min <= product_price <= expanded_max):
                    logger.debug(f"상품 {product_id} 가격 필터 실패: {product_price}원 (범위: {expanded_min}-{expanded_max})")
                    debug_stats["price_filtered"] += 1
                    continue
            
            # 4. 프로필-상품 매칭 점수 계산
            profile_match_score = self._calculate_profile_match_score(
                request.beauty_profile, 
                product,
                details
            )
            
            # 5. 최종 점수 계산
            final_score = (similarity_score * 0.7 + profile_match_score * 0.3)
            
            # 6. 매칭된 특성 추출
            matched_features = self._extract_matched_features_improved(
                request.beauty_profile,
                product,
                details
            )
            
            # 7. 추천 이유 생성
            recommendation_reason = self._generate_recommendation_reason_improved(
                similarity_score, 
                profile_match_score,
                matched_features,
                request.beauty_profile,
                product
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
            
            debug_stats["final_recommendations"] += 1
            logger.debug(f"✅ 상품 {product_id} 추천 성공 (유사도: {similarity_score:.3f}, 매칭: {profile_match_score:.3f}, 최종: {final_score:.3f})")
        
        # 디버깅 통계 출력
        logger.info(f"📊 개선된 필터링 통계: 총 {debug_stats['total_candidates']}개 → "
                f"카테고리필터 {debug_stats['category_filtered']}개 제외 → "
                f"유사도필터 {debug_stats['similarity_filtered']}개 제외 → "
                f"가격필터 {debug_stats['price_filtered']}개 제외 → "
                f"최종 {debug_stats['final_recommendations']}개")
        
        # 최종 점수로 정렬
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        
        return recommendations
    
    def _generate_recommendation_reason_improved(
        self,
        similarity_score: float,
        profile_match_score: float,
        matched_features: List[str],
        beauty_profile: BeautyProfile,
        product: Product
    ) -> str:
        """개선된 추천 이유 생성"""
        
        reasons = []
        
        # 1. AI 분석 결과
        if similarity_score > 0.8:
            reasons.append("AI 분석 결과 매우 높은 적합도")
        elif similarity_score > 0.6:
            reasons.append("AI 분석 결과 높은 적합도")
        elif similarity_score > 0.4:
            reasons.append("AI 분석 결과 적절한 적합도")
        else:
            reasons.append("AI 분석 결과 기본 적합도")
        
        # 2. 프로필 매칭 결과
        if profile_match_score > 0.7:
            reasons.append("프로필 완벽 매칭")
        elif profile_match_score > 0.5:
            reasons.append("프로필 높은 매칭")
        elif profile_match_score > 0.3:
            reasons.append("프로필 기본 매칭")
        
        # 3. 구체적인 매칭 특성
        if matched_features:
            if len(matched_features) >= 3:
                reasons.append(f"'{matched_features[0]}', '{matched_features[1]}' 등 {len(matched_features)}개 특성 매칭")
            else:
                feature_str = "', '".join(matched_features)
                reasons.append(f"'{feature_str}' 특성 매칭")
        
        # 4. 카테고리 정보
        reasons.append(f"{product.category_main.value} 제품으로 추천")
        
        return ", ".join(reasons) + "합니다"
    

    def _extract_matched_features_improved(
        self,
        beauty_profile: BeautyProfile,
        product: Product,
        product_details: Dict[str, Any]
    ) -> List[str]:
        """개선된 매칭 특성 추출"""
        
        matched = []
        
        # 1. 피부타입 매칭
        skin_compatibility = product_details.get("skin_types", [])
        product_name = product.name.lower()
        product_description = (product.description or "").lower()
        
        # 직접 매칭
        if beauty_profile.skin_type.value in skin_compatibility:
            matched.append(f"{beauty_profile.skin_type.value} 적합")
        # 상품명/설명에서 피부타입 키워드 찾기
        elif beauty_profile.skin_type.value in product_name or beauty_profile.skin_type.value in product_description:
            matched.append(f"{beauty_profile.skin_type.value} 맞춤")
        # 모든피부용 제품
        elif "모든피부" in skin_compatibility or "전피부" in product_description:
            matched.append("모든 피부타입 적합")
        
        # 2. 피부 고민 매칭 
        if beauty_profile.concerns:
            product_benefits = product_details.get("benefits", [])
            all_text = f"{product_name} {product_description} {' '.join(product_benefits)}".lower()
            
            concern_keywords = {
                "여드름": ["여드름", "트러블", "뾰루지", "아크네", "acne", "진정", "항염"],
                "건조함": ["건조", "수분", "보습", "촉촉", "hydra"],
                "기름기": ["기름", "유분", "오일", "지성", "sebum", "테카", "피지과다"],
                "민감함": ["민감성", "자극", "순한", "gentle", "sensitive", "홍조", "아토피"],
                "주름": ["주름", "안티에이징", "리프팅", "탄력", "anti-aging"],
                "기미잡티": ["기미", "잡티", "미백", "브라이트닝", "화이트닝", "다크서클"],
                "모공": ["모공", "pore", "블랙헤드", "각질"],
                "탄력부족": ["탄력", "리프팅", "퍼밍", "콜라겐"]
            }
            
            for concern in beauty_profile.concerns:
                keywords = concern_keywords.get(concern, [concern])
                for keyword in keywords:
                    if keyword in all_text:
                        matched.append(f"{concern} 케어")
                        break
        
        # 3. 카테고리 관심사 매칭
        if beauty_profile.interest_categories:
            for interest_category in beauty_profile.interest_categories:
                if interest_category.lower() in product_details.get("category_path", "").lower():
                    matched.append(f"{interest_category} 관심사")
                    break
        
        # 4. 🔧 개선: 가격 매칭 (확장된 범위)
        product_price = float(product.base_price)
        expanded_min = beauty_profile.min_price * 0.8
        expanded_max = beauty_profile.max_price * 1.2
        
        if expanded_min <= product_price <= expanded_max:
            # 가격대별 메시지 차별화
            if product_price <= 20000:
                matched.append("가성비 좋은 가격")
            elif product_price <= 50000:
                matched.append("적정 가격대")
            else:
                matched.append("프리미엄 가격대")
        
        # 5. 🔧 개선: 알레르기 성분 체크 (안전성 강조)
        if beauty_profile.has_allergy and beauty_profile.allergies:
            key_ingredients = product_details.get("key_ingredients", [])
            ingredient_text = " ".join(key_ingredients).lower()
            
            has_allergy_ingredient = False
            for allergy in beauty_profile.allergies:
                if allergy.lower() in ingredient_text:
                    has_allergy_ingredient = True
                    break
            
            if not has_allergy_ingredient:
                matched.append("알레르기 성분 없음")
        
        return matched[:5]  # 최대 5개까지

    
    async def _get_product_details_with_category_filter(
        self, 
        product_ids: List[int],
        include_categories: Optional[List[ProductCategory]] = None,
        exclude_categories: Optional[List[ProductCategory]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """개선된 계층적 카테고리 필터링 - 메인 카테고리와 서브 카테고리 모두 고려"""
        try:
            async with AsyncSessionLocal() as db:
                # 기본 쿼리 (활성 상품 + 지정된 ID들)
                stmt = (
                    select(DBProduct)
                    .options(
                        selectinload(DBProduct.product_options),
                        selectinload(DBProduct.product_categories).selectinload(DBProductCategory.category)
                    )
                    .where(DBProduct.id.in_(product_ids))
                    .where(DBProduct.status == "ACTIVE")
                )
                
                # 🔧 계층적 카테고리 필터링
                if include_categories or exclude_categories:
                    logger.info(f"🏷️ 계층적 카테고리 필터 적용: include={[c.value for c in include_categories] if include_categories else None}, exclude={[c.value for c in exclude_categories] if exclude_categories else None}")
                    
                    # 카테고리 매핑 정보 생성
                    category_mapping = await self._get_category_hierarchy_mapping(db)
                    
                    if include_categories and exclude_categories:
                        # include와 exclude 모두 있는 경우
                        include_names = self._get_all_related_category_names(include_categories, category_mapping)
                        exclude_names = self._get_all_related_category_names(exclude_categories, category_mapping)
                        
                        logger.info(f"🔍 확장된 include 카테고리: {include_names}")
                        logger.info(f"🔍 확장된 exclude 카테고리: {exclude_names}")
                        
                        # 포함할 카테고리가 있는 상품만 선택
                        include_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(include_names))
                        )
                        
                        # 제외할 카테고리가 있는 상품은 제외
                        exclude_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(exclude_names))
                        )
                        
                        stmt = stmt.where(
                            and_(
                                DBProduct.id.in_(include_subquery),
                                not_(DBProduct.id.in_(exclude_subquery))
                            )
                        )
                        
                    elif include_categories:
                        include_names = self._get_all_related_category_names(include_categories, category_mapping)
                        logger.info(f"🔍 확장된 include 카테고리: {include_names}")
                        
                        include_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(include_names))
                        )
                        stmt = stmt.where(DBProduct.id.in_(include_subquery))
                        
                    elif exclude_categories:
                        exclude_names = self._get_all_related_category_names(exclude_categories, category_mapping)
                        logger.info(f"🔍 확장된 exclude 카테고리: {exclude_names}")
                        
                        exclude_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(exclude_names))
                        )
                        stmt = stmt.where(not_(DBProduct.id.in_(exclude_subquery)))

                result = await db.execute(stmt)
                db_products = result.scalars().all()
                
                logger.info(f"📊 계층적 필터링 결과: {len(db_products)}개 상품")

                product_details = {}
                for db_product in db_products:
                    try:
                        product = await self.product_converter.db_to_pydantic(db, db_product)
                        
                        details = {
                            "product": product,
                            "category_path": await self._get_category_path(db, db_product.id),
                            "price_range": self._get_actual_price_range(db_product),
                            "key_ingredients": self._extract_actual_ingredients(db_product),
                            "skin_types": self._extract_skin_compatibility(product.description),
                            "benefits": self._extract_product_benefits(product.description)
                        }
                        
                        product_details[db_product.id] = details
                        logger.debug(f"✅ 상품 {db_product.id} ({product.category_main.value}) 상세정보 추가")
                        
                    except Exception as e:
                        logger.error(f"상품 {db_product.id} 상세 정보 추출 실패: {e}")
                        continue
                
                logger.info(f"📊 최종 상품 상세정보: {len(product_details)}개")
                return product_details
                
        except Exception as e:
            logger.error(f"상품 상세 정보 조회 실패: {e}")
            return {}
        
    async def get_product_details(self, product_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        return await self._get_product_details_with_category_filter(product_ids)
    

    async def _get_category_hierarchy_mapping(self, db: AsyncSession) -> Dict[str, List[str]]:
        """카테고리 계층 매핑 정보 생성"""
        try:
            # 메인 카테고리와 연관된 모든 서브 카테고리 매핑 생성
            category_mapping = {
                "스킨케어": [
                    "스킨케어", "크림", "스킨/토너", "에센스/세럼/앰플", "아이크림", 
                    "로션", "로션/크림", "미스트/오일", "미스트/픽서", "올인원", 
                    "페이스오일", "스킨케어세트", "스킨케어 디바이스", "보습"
                ],
                "메이크업": [
                    "메이크업", "베이스메이크업", "아이메이크업", "립메이크업",
                    "쿠션", "파운데이션", "BB/CC", "컨실러", "프라이머/베이스",
                    "파우더/팩트", "블러셔", "쉐이딩", "하이라이터", "메이크업 픽서",
                    "아이라이너", "마스카라", "아이브로우", "아이섀도우", "아이 픽서", "아이래쉬 케어",
                    "립스틱", "립틴트", "립글로스", "립라이너", "립케어", "컬러립밤", "헤어메이크업"
                ],
                "클렌징": [
                    "클렌징", "클렌징폼/젤", "클렌징오일", "클렌징밤", "클렌징워터", 
                    "클렌징밀크/크림", "클렌징 비누", "립&아이리무버", "필링&스크럽",
                    "페이셜스크럽", "피지클리너", "파우더워시", "팩클렌저", "클렌징 디바이스"
                ],
                "마스크팩": [
                    "마스크팩", "시트팩", "워시오프팩", "모델링팩", "필오프팩", "슬리핑팩",
                    "패드", "페이셜팩", "코팩", "패치", "티슈/패드"
                ],
                "선케어": [
                    "선케어", "선크림", "선스틱", "선쿠션", "선파우더", "선스프레이",
                    "선패치", "선스프레이/선패치", "태닝", "애프터선", "태닝/애프터선"
                ],
                "향수": [
                    "향수", "액체향수", "고체향수", "바디퍼퓸", "헤어퍼퓸"
                ],
                "헤어케어": [
                    "헤어케어", "샴푸", "린스/컨디셔너", "샴푸/린스", "드라이샴푸", "스케일러",
                    "헤어 트리트먼트", "트리트먼트/팩", "노워시 트리트먼트", "두피앰플/토닉",
                    "헤어토닉/두피토닉", "헤어에센스", "헤어세럼", "헤어오일", "염색약/펌",
                    "새치염색", "컬러염색/탈색", "파마", "헤어메이크업", "헤어기기/브러시",
                    "헤어 브러시", "고데기", "드라이기", "스타일링", "컬크림/컬링에센스",
                    "왁스/젤/무스/토닉", "스프레이/픽서", "헤어퍼퓸"
                ],
                "바디케어": [
                    "바디케어", "바디워시", "바디스크럽", "입욕제", "샤워/입욕", "비누",
                    "로션/오일/미스트", "바디미스트", "바디오일", "핸드케어", "핸드크림", "핸드워시",
                    "풋케어", "풋크림", "풋샴푸", "발냄새제거제", "발각질제거제", "발관리용품",
                    "제모/왁싱", "면도기/면도날", "제모크림", "스트립/왁스", "제모기기", "남성 쉐이빙",
                    "데오드란트", "데오스틱", "데오롤온", "데오스프레이", "쿨링/데오시트", "베이비"
                ]
            }
            
            logger.info("🏗️ 카테고리 계층 매핑 생성 완료")
            return category_mapping
            
        except Exception as e:
            logger.error(f"카테고리 계층 매핑 생성 실패: {e}")
            return {
                "스킨케어": ["스킨케어"],
                "메이크업": ["메이크업"],
                "클렌징": ["클렌징"],
                "마스크팩": ["마스크팩"],
                "선케어": ["선케어"],
                "향수": ["향수"],
                "헤어케어": ["헤어케어"],
                "바디케어": ["바디케어"]
            }
    
    def _get_all_related_category_names(
        self, 
        categories: List[ProductCategory], 
        category_mapping: Dict[str, List[str]]
    ) -> List[str]:
        """메인 카테고리에 해당하는 모든 서브 카테고리 이름 반환"""
        all_names = []
        
        for category in categories:
            category_name = category.value
            if category_name in category_mapping:
                all_names.extend(category_mapping[category_name])
                logger.debug(f"📂 {category_name} 카테고리 확장: {len(category_mapping[category_name])}개 서브카테고리")
            else:
                # 매핑에 없으면 원본 이름만 사용
                all_names.append(category_name)
                logger.debug(f"📂 {category_name} 카테고리: 매핑 없음, 원본 사용")
        
        # 중복 제거
        unique_names = list(set(all_names))
        logger.info(f"🔍 최종 확장된 카테고리: {len(unique_names)}개 ({unique_names})")
        
        return unique_names
    
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
            
            active_options = list(db_product.product_options)
            
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
                if option.full_ingredients:
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
        """개선된 Fallback 추천 - 더 관대한 조건"""
        logger.info("🔄 개선된 Fallback 추천 실행")
        
        try:
            async with AsyncSessionLocal() as db:
                stmt = (
                    select(DBProduct)
                    .options(selectinload(DBProduct.product_options))
                    .where(DBProduct.status == "ACTIVE")
                )

                # 카테고리 필터링 (기존과 동일)
                if request.include_categories or request.exclude_categories:
                    if request.include_categories and request.exclude_categories:
                        include_names = [cat.value for cat in request.include_categories]
                        exclude_names = [cat.value for cat in request.exclude_categories]
                        
                        include_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(include_names))
                        )
                        
                        exclude_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(exclude_names))
                        )
                        
                        stmt = stmt.where(
                            and_(
                                DBProduct.id.in_(include_subquery),
                                not_(DBProduct.id.in_(exclude_subquery))
                            )
                        )
                        
                    elif request.include_categories:
                        include_names = [cat.value for cat in request.include_categories]
                        include_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(include_names))
                        )
                        stmt = stmt.where(DBProduct.id.in_(include_subquery))
                        
                    elif request.exclude_categories:
                        exclude_names = [cat.value for cat in request.exclude_categories]
                        exclude_subquery = (
                            select(DBProductCategory.product_id)
                            .join(DBCategory)
                            .where(DBCategory.name.in_(exclude_names))
                        )
                        stmt = stmt.where(not_(DBProduct.id.in_(exclude_subquery)))
                
                stmt = stmt.limit(100) 
                result = await db.execute(stmt)
                db_products = result.scalars().all()

                if not db_products:
                    logger.warning("🚨 Fallback에서도 결과 없음")
                    return []
                
                fallback_results = []
                processed_count = 0

                for db_product in db_products:
                    try:
                        product = await self.product_converter.db_to_pydantic(db, db_product)

                        if request.use_price_filter:
                            product_price = float(product.base_price)
                            expanded_min = request.beauty_profile.min_price * 0.5  
                            expanded_max = request.beauty_profile.max_price * 1.5  
                            
                            if not (expanded_min <= product_price <= expanded_max):
                                continue
                        
                        product_name = product.name.lower()
                        beauty_keywords = ["보습", "수분", "진정", "순한", "민감", "여드름", "트러블"]
                        found_keywords = [kw for kw in beauty_keywords if kw in product_name]
                        
                        if found_keywords:
                            reason = f"'{', '.join(found_keywords)}' 특성의 {product.category_main.value} 추천 제품"
                        else:
                            reason = f"인기 {product.category_main.value} 추천 제품"
                        
                        fallback_results.append({
                            "product_id": db_product.id,
                            "similarity_score": max(0.4 - (processed_count * 0.01), 0.1), 
                            "profile_match_score": 0.3,
                            "final_score": max(0.35 - (processed_count * 0.01), 0.15), 
                            "confidence_score": 0.3,
                            "ranking_position": processed_count + 1,
                            "recommendation_reason": reason,
                            "matched_features": found_keywords if found_keywords else ["일반 추천"],
                            "confidence_level": "low",
                            "category_path": f"{product.category_main.value} > {product.category_sub}",
                            "price_range": self._get_actual_price_range(db_product),
                            "recommendation_method": "fallback_improved"
                        })

                        processed_count += 1
                        if processed_count >= request.top_k * 2: 
                            break
                        
                    except Exception as e:
                        logger.error(f"Fallback 상품 {db_product.id} 처리 실패: {e}")
                        continue
                
                logger.info(f"🔄 개선된 Fallback 결과: {len(fallback_results)}개")
                return fallback_results
                
        except Exception as e:
            logger.error(f"개선된 Fallback 추천 실패: {e}")
            return []
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """프로필 기반 추천 서비스 통계"""
        return {
            "service_name": "ProfileBasedRecommendationService",
            "version": "5.0 (카테고리 필터링 개선)",
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
                "price_range_filtering": True,
                "improved_category_filtering": True
            },
            "scoring_method": {
                "vector_similarity_weight": 0.7,
                "profile_match_weight": 0.3,
                "confidence_calculation": "similarity + match - rank_penalty"
            },
            "data_sources": {
                "user_profile": "BeautyProfile (structured)",
                "product_embeddings": "Faiss Vector Store",
                "product_details": "DB (products, ingredients, benefits)",
                "category_filtering": "SQL subquery (완전 처리)"
            }
        }