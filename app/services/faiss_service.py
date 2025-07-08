import faiss
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Typle, Optional, Union
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

logger = logging.getLogger(__name__)

class FaissIndexManager:
    """Faiss 인덱스 관리 클래스"""

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        selt.dimension = dimension
        self.index = None
        self.proudct_ids = []
        self.metadata = {}
        self.index_type = settings.FAISS_INDEX_TYPE
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.executor = ThreadPoolExecutor(max_workers=settings.FAISS_THREAD_POOL_SIZE)

        logger.debug(f"Faiss 인덱스 매니저 초기화 - 차원: {dimension}, 인덱스 타입: {self.index_type}")

    def _create_index(self, index_type: str = None) -> faiss.Index:
        """Faiss 인덱스 생성"""
        if index_type is None:
            index_type = self.index_type

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
                logger.debug(f"기존 Faiss 인덱스 로드 완료: {len(self.product_ids)}개 벡터")
                return
            except Exception as e:
                logger.warning(f"기존 인덱스 로드 실패: {e}, 인덱스 재생성 필요")

        self.index = self._create_index()
        self.proudct_ids = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": 0
        }

        logger.debug("새 Faiss 인덱스 생성 완료")

    def add_vectors(self, vectors:np.ndarray, product_ids: List[int], batch_metadate: Dict = None):
        """벡터들을 인덱스에 추가"""
        if self.index is None:
            raise ValueError("인덱스가 초기화되지 않았습니다.")
        if len(vectors) != len(product_ids):
            raise ValueError("벡터와 제품 ID의 길이가 일치하지 않습니다.")
        if self.index_type == "IndexIVFFlat" :
            vectors = self._normalize_vectors(vectors)

        start_idx = len(self.proudct_ids)
        self.index.add(vectors.astype(np.float32))
        self.proudct_ids.extend(product_ids)

        self.metadata["total_vectors"] = len(self.proudct_ids)
        self.metadata["last_updated"] = datetime.now().isoformat()

        if batch_metadate:
            self.metadata.update(batch_metadate)

        logger.debug(f"{len(vectors)}개의 벡터를 인덱스에 추가했습니다. 현재 총 {len(self.proudct_ids)}개 벡터")

        return start_idx, start_idx + len(vectors)
    
    
