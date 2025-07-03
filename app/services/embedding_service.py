import re
import time
import logging
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from sentence_transformers import SentenceTransformer

from app.core.config import settings


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """기본 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정리 (한국어, 영어, 숫자, 기본 구두점만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        return text.strip()
    
    @staticmethod
    def remove_stopwords(text: str, stopwords: List[str] = None) -> str:
        """불용어 제거"""
        if stopwords is None:
            stopwords = settings.KOREAN_STOPWORDS
        
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        return ' '.join(filtered_words)
    
    @staticmethod
    def normalize_korean_text(text: str) -> str:
        """한국어 텍스트 정규화"""
        # 설정에서 정규화 매핑 가져오기
        replacements = settings.BEAUTY_TERMS_MAPPING
        
        for original, normalized in replacements.items():
            text = text.replace(original, normalized)
        
        return text
    
    @classmethod
    def preprocess_product_text(cls, product_name: str, brand: str,
                              description: str, ingredients: str = None,
                              category: str = None) -> str:
        """상품 정보를 임베딩용 텍스트로 전처리"""
        
        # 각 필드별 전처리
        clean_name = cls.clean_text(product_name)
        clean_brand = cls.clean_text(brand)
        clean_description = cls.clean_text(description)
        clean_ingredients = cls.clean_text(ingredients) if ingredients else ""
        clean_category = cls.clean_text(category) if category else ""
        
        # 한국어 정규화
        clean_description = cls.normalize_korean_text(clean_description)
        clean_ingredients = cls.normalize_korean_text(clean_ingredients)
        
        # 통합 텍스트 구성
        parts = []
        
        if clean_name:
            parts.append(f"상품명: {clean_name}")
        if clean_brand:
            parts.append(f"브랜드: {clean_brand}")
        if clean_category:
            parts.append(f"카테고리: {clean_category}")
        if clean_description:
            parts.append(f"설명: {clean_description}")
        if clean_ingredients:
            parts.append(f"성분: {clean_ingredients}")
        
        combined_text = " | ".join(parts)
        
        # 최종 정리
        final_text = cls.remove_stopwords(combined_text)
        
        # 최대 길이 제한 (BERT 모델 제한 고려)
        # 한국어는 토큰화 시 더 많은 토큰이 생성될 수 있어 보수적으로 제한
        if len(final_text) > settings.MAX_SEQUENCE_LENGTH * 2:
            final_text = final_text[:settings.MAX_SEQUENCE_LENGTH * 2]
        
        return final_text


class EmbeddingServiceInterface(ABC):
    """임베딩 서비스 인터페이스"""
    
    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩으로 변환"""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 변환"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        pass


class ProductEmbeddingGenerator:
    """상품 임베딩 생성 관리 클래스"""
    
    def __init__(self, embedding_service: EmbeddingServiceInterface):
        self.embedding_service = embedding_service
        self.preprocessor = TextPreprocessor()
        
    def generate_product_embedding(self, product_name: str, brand: str,
                                 description: str, ingredients: str = None,
                                 category: str = None) -> Tuple[List[float], str]:
        """단일 상품의 임베딩 생성"""
        
        # 텍스트 전처리
        processed_text = self.preprocessor.preprocess_product_text(
            product_name, brand, description, ingredients, category
        )
        
        logger.info(f"상품 임베딩 생성 중: {product_name[:20]}...")
        
        # 임베딩 생성
        try:
            embedding = self.embedding_service.encode_text(processed_text)
            logger.info(f"임베딩 생성 완료. 차원: {len(embedding)}")
            return embedding, processed_text
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def generate_batch_embeddings(self, products_data: List[Dict]) -> List[Tuple[List[float], str]]:
        """여러 상품의 임베딩을 배치로 생성"""
        
        processed_texts = []
        
        # 모든 상품 텍스트 전처리
        for product in products_data:
            processed_text = self.preprocessor.preprocess_product_text(
                product.get('name', ''),
                product.get('brand', ''),
                product.get('description', ''),
                product.get('ingredients'),
                product.get('category')
            )
            processed_texts.append(processed_text)
        
        logger.info(f"배치 임베딩 생성 중: {len(processed_texts)}개 상품")
        
        try:
            # 배치 임베딩 생성
            embeddings = self.embedding_service.encode_batch(processed_texts)
            
            # 결과 조합
            results = list(zip(embeddings, processed_texts))
            
            logger.info(f"배치 임베딩 생성 완료: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            raise


class EmbeddingCache:
    """임베딩 캐시 관리 클래스"""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.ttl_seconds = settings.CACHE_TTL_SECONDS
    
    def _generate_cache_key(self, text: str, model_version: str) -> str:
        """캐시 키 생성"""
        import hashlib
        content = f"{text}_{model_version}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_version: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회"""
        cache_key = self._generate_cache_key(text, model_version)
        
        if cache_key in self._cache:
            embedding, created_at = self._cache[cache_key]
            
            # TTL 체크
            elapsed = (datetime.now() - created_at).total_seconds()
            if elapsed < self.ttl_seconds:
                logger.debug(f"캐시 히트: {cache_key[:8]}...")
                return embedding
            else:
                # 만료된 캐시 제거
                del self._cache[cache_key]
                logger.debug(f"캐시 만료: {cache_key[:8]}...")
        
        return None
    
    def set(self, text: str, model_version: str, embedding: List[float]):
        """캐시에 임베딩 저장"""
        cache_key = self._generate_cache_key(text, model_version)
        self._cache[cache_key] = (embedding, datetime.now())
        logger.debug(f"캐시 저장: {cache_key[:8]}...")
    
    def clear_expired(self):
        """만료된 캐시 정리"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (_, created_at) in self._cache.items():
            elapsed = (current_time - created_at).total_seconds()
            if elapsed >= self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"만료된 캐시 {len(expired_keys)}개 정리 완료")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """캐시 통계 정보"""
        return {
            "total_items": len(self._cache),
            "ttl_seconds": self.ttl_seconds
        }


class EmbeddingService(EmbeddingServiceInterface):
    """SentenceTransformer 기반 임베딩 서비스 구현체"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.model = None
        self.cache = EmbeddingCache()
        self._validate_model_config()
        self._load_model()
    
    def _validate_model_config(self):
        """모델 설정 검증"""
        logger.info(f"🔧 현재 환경: {settings.ENVIRONMENT}")
        logger.info(f"📝 임베딩 모델: {self.model_name}")
        logger.info(f"📏 임베딩 차원: {settings.EMBEDDING_DIMENSION}")
        logger.info(f"📊 배치 크기: {settings.PRODUCT_EMBEDDING_BATCH_SIZE}")
        
        # 개발 환경에서 경량 모델 사용 권장
        if settings.is_development and "large" in self.model_name.lower():
            logger.warning("⚠️  개발 환경에서 대형 모델을 사용하고 있습니다. 성능 이슈가 있을 수 있습니다.")
        
        # 운영 환경에서 한국어 모델 사용 권장
        if settings.is_production and "kr-" not in self.model_name.lower():
            logger.warning("⚠️  운영 환경에서 한국어 특화 모델이 아닙니다. 성능 확인이 필요합니다.")
    
    def _load_model(self):
        """모델 로드"""
        try:
            logger.info(f"🤖 임베딩 모델 로드 시작: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name)
            
            load_time = time.time() - start_time
            logger.info(f"✅ 임베딩 모델 로드 완료 ({load_time:.2f}초)")
            
            # 모델 정보 검증
            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim != settings.EMBEDDING_DIMENSION:
                logger.error(f"❌ 모델 차원 불일치: 설정={settings.EMBEDDING_DIMENSION}, 실제={model_dim}")
                raise ValueError(f"모델 차원이 설정과 다릅니다: {model_dim} != {settings.EMBEDDING_DIMENSION}")
                
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def encode_text(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩으로 변환"""
        # 빈 텍스트나 공백만 있는 텍스트 처리
        if not text or not text.strip():
            logger.warning("빈 텍스트가 입력되어 제로 벡터를 반환합니다.")
            # 화장품 도메인에서는 텍스트 정보가 중요하므로 제로 벡터 사용
            # 이는 해당 상품이 추천에서 자연스럽게 제외되도록 함
            return [0.0] * settings.EMBEDDING_DIMENSION
        
        # 캐시 확인
        cached_embedding = self.cache.get(text, self.model_name)
        if cached_embedding:
            return cached_embedding
        
        try:
            # 임베딩 생성
            embedding = self.model.encode(text, convert_to_tensor=False)
            embedding_list = embedding.tolist()
            
            # 캐시 저장
            self.cache.set(text, self.model_name, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return [0.0] * settings.EMBEDDING_DIMENSION
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 변환"""
        if not texts:
            return []
        
        try:
            # 배치 임베딩 생성
            embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=settings.PRODUCT_EMBEDDING_BATCH_SIZE)
            
            # 각각 캐시에 저장
            results = []
            for text, embedding in zip(texts, embeddings):
                embedding_list = embedding.tolist()
                self.cache.set(text, self.model_name, embedding_list)
                results.append(embedding_list)
            
            return results
            
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            # 실패 시 개별적으로 처리
            return [self.encode_text(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, str]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "model_type": "SentenceTransformer",
            "embedding_dimension": str(settings.EMBEDDING_DIMENSION),
            "max_sequence_length": str(settings.MAX_SEQUENCE_LENGTH)
        }
