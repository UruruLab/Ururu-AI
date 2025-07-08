# app/services/faiss_service.py - 1단계 수정 완료 버전
import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

logger = logging.getLogger(__name__)

class FaissIndexManager:
    """Faiss 인덱스 관리 클래스"""

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = None
        self.product_ids = [] 
        self.metadata = {}
        self.index_type = settings.FAISS_INDEX_TYPE
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        thread_pool_size = getattr(settings, 'FAISS_THREAD_POOL_SIZE', 2)
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_size)

        logger.info(f"🔍 Faiss 인덱스 매니저 초기화 - 차원: {dimension}, 인덱스 타입: {self.index_type}")

    def _create_index(self, index_type: str = None) -> faiss.Index:
        """Faiss 인덱스 생성"""
        if index_type is None:
            index_type = self.index_type

        logger.debug(f"📊 Faiss 인덱스 생성: {index_type} (차원: {self.dimension})")

        if index_type == "IndexFlatIP":
            # 내적 기반 인덱스 - 코사인 유사도용
            index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexFlatL2":
            # L2 거리 기반 인덱스
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IndexIVFFlat":
            # IVF 인덱스 - 대용량 데이터용
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100) 
        elif index_type == "IndexHNSW":
            # HNSW 인덱스 - 빠른 근사 검색
            index = faiss.IndexHNSWFlat(self.dimension, 32) 
        else:
            # 기본값: Flat IP
            logger.warning(f"알 수 없는 인덱스 타입: {index_type}, IndexFlatIP 사용")
            index = faiss.IndexFlatIP(self.dimension)
        
        return index
    
    def initialize_index(self, force_recreate: bool = False):
        """Faiss 인덱스 초기화"""
        index_file = self.index_path / f"product_index_{self.index_type}.faiss"
        metadata_file = self.index_path / f"product_metadata_{self.index_type}.json"

        if not force_recreate and index_file.exists() and metadata_file.exists():
            try:
                self._load_index()
                logger.info(f"✅ 기존 Faiss 인덱스 로드 완료: {len(self.product_ids)}개 벡터")
                return
            except Exception as e:
                logger.warning(f"기존 인덱스 로드 실패: {e}, 인덱스 재생성 필요")

        self.index = self._create_index()
        self.product_ids = []  
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": 0
        }

        logger.info("🆕 새 Faiss 인덱스 생성 완료")

    def add_vectors(self, vectors: np.ndarray, product_ids: List[int], batch_metadata: Dict = None):  
        """벡터들을 인덱스에 추가"""
        if self.index is None:
            raise ValueError("인덱스가 초기화되지 않았습니다.")
        if len(vectors) != len(product_ids):
            raise ValueError("벡터와 제품 ID의 길이가 일치하지 않습니다.")

        if self.index_type == "IndexFlatIP":  
            vectors = self._normalize_vectors(vectors)

        start_idx = len(self.product_ids) 
        self.index.add(vectors.astype(np.float32))
        self.product_ids.extend(product_ids)  

        self.metadata["total_vectors"] = len(self.product_ids) 
        self.metadata["last_updated"] = datetime.now().isoformat()

        if batch_metadata: 
            self.metadata.update(batch_metadata)

        logger.info(f"➕ 벡터 추가 완료: {len(vectors)}개 (총 {len(self.product_ids)}개)")

        return start_idx, start_idx + len(vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """벡터 검색"""
        if self.index is None or len(self.product_ids) == 0:
            logger.warning("빈 인덱스에서 검색 요청")
            return [], []
        
        if self.index_type == "IndexFlatIP":
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1))
        else:
            query_vector = query_vector.reshape(1, -1)

        k = min(k, len(self.product_ids))  
        scores, indices = self.index.search(query_vector.astype(np.float32), k)
        scores = scores[0].tolist()
        product_ids = [self.product_ids[idx] for idx in indices[0] if idx < len(self.product_ids)]  

        logger.debug(f"🔍 벡터 검색 완료: top-{k}, 최고 점수: {scores[0]:.4f}")

        return scores, product_ids
    
   
    def search_raw(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """원시 벡터 검색 - Faiss 점수와 인덱스만 반환"""
        if self.index is None or len(self.product_ids) == 0:
            logger.warning("빈 인덱스에서 검색 요청")
            return np.array([]), np.array([])

        if self.index_type == "IndexFlatIP":
            query_vector = self._normalize_vectors(query_vector.reshape(1, -1))
        else:
            query_vector = query_vector.reshape(1, -1)

        k = min(k, len(self.product_ids))
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        logger.debug(f"🔍 원시 벡터 검색 완료: top-{k}")

        return scores[0], indices[0]
    
    async def search_async(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """비동기 벡터 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.search, query_vector, k)
    
    async def search_raw_async(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """비동기 원시 벡터 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.search_raw, query_vector, k)
    
    def get_product_ids_by_indices(self, indices: np.ndarray) -> List[int]:
        """인덱스를 상품 ID로 변환"""
        return [self.product_ids[idx] for idx in indices if idx < len(self.product_ids)]
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """벡터 정규화 (코사인 유사도용)"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        return vectors / norms
    
    def save_index(self):
        """인덱스를 파일로 저장"""
        if self.index is None:
            logger.warning("저장할 인덱스가 없습니다")
            return
        
        index_file = self.index_path / f"product_index_{self.index_type}.faiss"
        metadata_file = self.index_path / f"product_metadata_{self.index_type}.json"
        
        try:
            faiss.write_index(self.index, str(index_file))
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": self.metadata,
                    "product_ids": self.product_ids
                }, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 Faiss 인덱스 저장 완료: {index_file}")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {e}")
            raise
    
    def _load_index(self):
        """저장된 인덱스 로드"""
        index_file = self.index_path / f"product_index_{self.index_type}.faiss"
        metadata_file = self.index_path / f"product_metadata_{self.index_type}.json"
        
        self.index = faiss.read_index(str(index_file))
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data["metadata"]
            self.product_ids = data["product_ids"]
        
        logger.debug(f"📂 인덱스 로드 완료: {len(self.product_ids)}개 벡터")
    
    def remove_vectors(self, product_ids_to_remove: List[int]):
        """특정 상품들의 벡터 제거 (재구축 방식)"""
        if not product_ids_to_remove:
            return
        
        remaining_indices = []
        remaining_product_ids = []
        
        for i, pid in enumerate(self.product_ids):
            if pid not in product_ids_to_remove:
                remaining_indices.append(i)
                remaining_product_ids.append(pid)
        
        if not remaining_indices:
            self.index = self._create_index()
            self.product_ids = []
            logger.info("🗑️ 모든 벡터 제거 완료")
            return
        
        old_vectors = np.array([self.index.reconstruct(i) for i in remaining_indices])
        
        self.index = self._create_index()
        self.product_ids = []
        self.add_vectors(old_vectors, remaining_product_ids)
        
        logger.info(f"🗑️ 벡터 제거 완료: {len(product_ids_to_remove)}개 제거, {len(remaining_product_ids)}개 유지")

    def get_index_stats(self) -> Dict:
        """인덱스 통계 정보"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": len(self.product_ids),
            "index_size_mb": self.index.ntotal * self.dimension * 4 / (1024 * 1024),  # float32 기준
            "metadata": self.metadata
        }
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("🔒 Faiss 인덱스 매니저 종료")
        
    
class FaissVectorStore:
    """Faiss 벡터 저장소 - 수정 완료"""
    
    def __init__(self):
        self.index_manager = FaissIndexManager()
        self.index_manager.initialize_index()
        logger.info("🚀 Faiss 벡터 저장소 초기화 완료")
    
    async def add_embeddings(self, embeddings_data: List[Dict]) -> bool: 
        """임베딩 데이터를 저장소에 추가"""
        try:
            if not embeddings_data:
                logger.warning("추가할 임베딩 데이터가 없습니다")
                return False
            
            vectors = []
            product_ids = []
            
            for data in embeddings_data:
                vectors.append(data["embedding"])
                product_ids.append(data["product_id"])
            
            vectors_array = np.array(vectors)
            
            # 비동기적으로 추가
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.index_manager.executor,
                self.index_manager.add_vectors,
                vectors_array,
                product_ids
            )
            
            # 자동 저장
            await loop.run_in_executor(
                self.index_manager.executor,
                self.index_manager.save_index
            )
            
            logger.info(f"✅ 임베딩 저장 완료: {len(embeddings_data)}개")
            return True
            
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
            return False
    
    async def search_vectors(self, query_embedding: List[float], k: int = 10) -> Tuple[List[float], List[int]]:
        """순수 벡터 검색 - 원시 점수와 상품 ID만 반환"""
        try:
            query_vector = np.array(query_embedding)
            scores, indices = await self.index_manager.search_raw_async(query_vector, k)
            
            # 상품 ID 변환
            product_ids = self.index_manager.get_product_ids_by_indices(indices)
            
            logger.debug(f"🔍 벡터 검색 완료: {len(product_ids)}개 결과")
            return scores.tolist(), product_ids
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return [], []
    
    def get_store_stats(self) -> Dict:  
        """저장소 통계"""
        return {
            "store_name": "FaissVectorStore",
            "index_stats": self.index_manager.get_index_stats(),
            "settings": {
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "index_type": settings.FAISS_INDEX_TYPE,
                "min_similarity_threshold": getattr(settings, 'MIN_SIMILARITY_THRESHOLD', 0.3),
                "max_similarity_threshold": getattr(settings, 'MAX_SIMILARITY_THRESHOLD', 1.0)
            }
        }
    
    async def close(self):
        """저장소 종료"""
        if hasattr(self.index_manager, 'executor'):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.index_manager.close)
        logger.info("🔒 Faiss 벡터 저장소 종료")