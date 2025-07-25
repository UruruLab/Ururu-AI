import re
import logging
from typing import List
import numpy as np

from app.models.product import Product
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
        
        # 최소한의 정보 체크: 상품명과 카테고리만 있고 설명/성분이 없으면 기본 설명 추가
        if not product.description and not product.ingredients and product.name:
            combined_text += f" | 기본정보: {product.category_main.value} 상품"
        
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
        """뷰티 관련 용어 정규화 (중복 방지)"""
        # 설정에서 뷰티 용어 매핑 가져오기
        beauty_terms = settings.BEAUTY_TERMS_MAPPING
        
        # 길이 순으로 정렬하여 긴 키워드부터 처리 (중복 방지)
        sorted_terms = sorted(beauty_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        for original, normalized in sorted_terms:
            # 단어 경계를 고려한 정확한 매칭
            pattern = rf'\b{re.escape(original)}\b'
            text = re.sub(pattern, normalized, text)
        
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
    