import re
import logging
from typing import List
import numpy as np

from app.models.product import Product, ProductProfile, ProductCategory
from app.services.embedding_service import EmbeddingServiceInterface
from app.core.config import settings


logger = logging.getLogger(__name__)


class ProductTowerService:
    """Product Tower 서비스 - User Tower와 대칭적 구조"""
    
    def __init__(self, embedding_service: EmbeddingServiceInterface):
        self.embedding_service = embedding_service
        self.model_name = embedding_service.get_model_info().get('model_name', settings.EMBEDDING_MODEL_NAME)
        self.embedding_dim = settings.EMBEDDING_DIMENSION
        
    def preprocess_product_text(self, product: Product) -> str:
        """상품 정보를 임베딩용 텍스트로 전처리"""
        
        # 텍스트 구성 요소들
        components = []
        
        # 기본 정보
        if product.name:
            components.append(f"상품명: {self._clean_text(product.name)}")
        
        if product.brand:
            components.append(f"브랜드: {self._clean_text(product.brand)}")
            
        if product.category_main:
            components.append(f"카테고리: {product.category_main.value}")
            
        if product.category_sub:
            components.append(f"세부카테고리: {self._clean_text(product.category_sub)}")
        
        # 상품 설명
        if product.description:
            clean_desc = self._clean_text(product.description)
            clean_desc = self._normalize_beauty_terms(clean_desc)
            components.append(f"설명: {clean_desc}")
        
        # 성분 정보
        if product.ingredients:
            clean_ingredients = self._clean_text(product.ingredients)
            components.append(f"성분: {clean_ingredients}")
        
        # 가격대 정보 추가
        price_range = self._get_price_range(product.base_price)
        components.append(f"가격대: {price_range}")
        
        # 최종 텍스트 조합
        combined_text = " | ".join(components)
        
        # 길이 제한 (BERT 모델 제한 고려)
        max_length = settings.TEXT_MAX_LENGTH
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        return combined_text
    
    def _clean_text(self, text: str) -> str:
        """기본 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정리 (한국어, 영어, 숫자, 기본 구두점만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_beauty_terms(self, text: str) -> str:
        """뷰티 관련 용어 정규화"""
        # 설정에서 뷰티 용어 매핑 가져오기
        beauty_terms = settings.BEAUTY_TERMS_MAPPING
        
        for original, normalized in beauty_terms.items():
            text = text.replace(original, normalized)
        
        return text
    
    def _get_price_range(self, price: float) -> str:
        """가격대 분류"""
        if price < settings.PRICE_RANGE_LOW:
            return "저가"
        elif price < settings.PRICE_RANGE_MID_LOW:
            return "중저가"
        elif price < settings.PRICE_RANGE_MID:
            return "중가"
        elif price < settings.PRICE_RANGE_MID_HIGH:
            return "중고가"
        else:
            return "고가"
    
    def extract_product_profile(self, product: Product) -> ProductProfile:
        """상품에서 프로필 정보 추출"""
        
        # 피부 타입 호환성 추출
        skin_compatibility = self._extract_skin_compatibility(product)
        
        # 주요 성분 추출
        key_ingredients = self._extract_key_ingredients(product.ingredients or "")
        
        # 제품 효능 추출
        benefits = self._extract_benefits(product.description or "")
        
        # 타겟 고민 추출
        target_concerns = self._extract_target_concerns(product.description or "")
        
        return ProductProfile(
            product_type=product.category_main,
            skin_compatibility=skin_compatibility,
            key_ingredients=key_ingredients,
            benefits=benefits,
            price_range=self._get_price_range(product.base_price),
            target_concerns=target_concerns,
            brand_positioning=self._get_brand_positioning(product.brand)
        )
    
    def _extract_skin_compatibility(self, product: Product) -> List[str]:
        """상품 설명에서 적합한 피부 타입 추출"""
        description = (product.description or "").lower()
        compatibility = []
        
        skin_type_indicators = {
            "건성": ["건성", "수분", "보습", "촉촉"],
            "지성": ["지성", "유분", "오일", "기름", "테카"],
            "민감성": ["민감", "순한", "자극", "진정"],
            "복합성": ["복합성", "T존", "부분"],
            "트러블성": ["트러블", "여드름", "뾰루지", "진정"],
            "모든피부": ["모든피부", "모든타입", "전피부"]
        }
        
        for skin_type, indicators in skin_type_indicators.items():
            if any(indicator in description for indicator in indicators):
                compatibility.append(skin_type)
        
        return compatibility if compatibility else ["모든피부"]
    
    def _extract_key_ingredients(self, ingredients: str) -> List[str]:
        """주요 성분 추출"""
        if not ingredients:
            return []
        
        # 주요 뷰티 성분들
        key_beauty_ingredients = [
            "히알루론산", "세라마이드", "니아신아마이드", "비타민C", "레티놀",
            "펩타이드", "콜라겐", "엘라스틴", "알로에", "센텔라", "판테놀",
            "스쿠알란", "아르간오일", "호호바오일", "시어버터", "글리세린",
            "살리실산", "글리콜산", "젖산", "코엔자임Q10", "아데노신"
        ]
        
        found_ingredients = []
        ingredients_lower = ingredients.lower()
        
        for ingredient in key_beauty_ingredients:
            if ingredient.lower() in ingredients_lower:
                found_ingredients.append(ingredient)
        
        return found_ingredients[:settings.MAX_KEY_INGREDIENTS]
    
    def _extract_benefits(self, description: str) -> List[str]:
        """제품 효능 추출"""
        if not description:
            return []
        
        benefit_keywords = {
            "보습": ["보습", "수분", "촉촉"],
            "미백": ["미백", "브라이트닝", "화이트닝", "기미", "잡티"],
            "주름개선": ["주름", "안티에이징", "탄력", "리프팅"],
            "진정": ["진정", "수딩", "자극완화", "민감"],
            "모공관리": ["모공", "블랙헤드", "각질"],
            "트러블케어": ["트러블", "여드름", "뾰루지"],
            "선크림": ["자외선", "SPF", "PA", "선크림"],
            "각질제거": ["각질", "필링", "엑스폴리에이션"]
        }
        
        found_benefits = []
        description_lower = description.lower()
        
        for benefit, keywords in benefit_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                found_benefits.append(benefit)
        
        return found_benefits[:settings.MAX_BENEFITS]
    
    def _extract_target_concerns(self, description: str) -> List[str]:
        """타겟 피부 고민 추출"""
        if not description:
            return []
        
        concern_keywords = {
            "건조함": ["건조", "수분부족", "당김"],
            "기름기": ["기름", "유분", "테카", "번들"],
            "트러블": ["트러블", "여드름", "뾰루지", "염증"],
            "민감함": ["민감", "자극", "따끔", "빨갛"],
            "주름": ["주름", "잔주름", "노화"],
            "기미잡티": ["기미", "잡티", "색소침착", "칙칙"],
            "모공": ["모공", "블랙헤드", "각질"],
            "탄력부족": ["탄력", "처짐", "리프팅"]
        }
        
        found_concerns = []
        description_lower = description.lower()
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                found_concerns.append(concern)
        
        return found_concerns[:settings.MAX_TARGET_CONCERNS]
    
    def _get_brand_positioning(self, brand: str) -> str:
        """브랜드 포지셔닝 분류"""
        # 설정에서 브랜드 분류 가져오기
        if brand in settings.get_premium_brands():
            return "프리미엄"
        elif brand in settings.get_drugstore_brands():
            return "드럭스토어"
        elif brand in settings.get_korean_brands():
            return "K뷰티"
        else:
            return "일반"
    
    def calculate_similarity_score(self, user_vector: List[float], 
                                 product_vector: List[float]) -> float:
        """사용자와 상품 벡터 간 코사인 유사도 계산"""
        try:
            user_array = np.array(user_vector)
            product_array = np.array(product_vector)
            
            # 코사인 유사도 계산
            dot_product = np.dot(user_array, product_array)
            user_norm = np.linalg.norm(user_array)
            product_norm = np.linalg.norm(product_array)
            
            if user_norm == 0 or product_norm == 0:
                return 0.0
            
            similarity = dot_product / (user_norm * product_norm)
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"유사도 계산 오류: {e}")
            return 0.0
    
    def generate_recommendation_reason(self, product: Product, 
                                     matched_keywords: List[str],
                                     similarity_score: float) -> str:
        """추천 이유 생성"""
        
        reasons = []
        
        # 키워드 매칭 기반 이유
        if matched_keywords:
            keyword_str = ", ".join(matched_keywords[:3])
            reasons.append(f"'{keyword_str}' 키워드가 매칭됩니다")
        
        # 제품 카테고리 기반
        category_reason = f"{product.category_main.value} 제품으로"
        if product.category_sub:
            category_reason += f" {product.category_sub} 타입입니다"
        reasons.append(category_reason)
        
        # 유사도 점수 기반
        if similarity_score >= 0.8:
            reasons.append("매우 높은 유사도를 보입니다")
        elif similarity_score >= 0.6:
            reasons.append("높은 유사도를 보입니다")
        else:
            reasons.append("적절한 유사도를 보입니다")
        
        # 브랜드 신뢰도
        if product.brand:
            reasons.append(f"{product.brand} 브랜드 제품입니다")
        
        return ". ".join(reasons) + "."
