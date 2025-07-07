"""
FastAPI 전용 AI 테이블들
- AI 추천 시스템에서만 사용
- FastAPI에서 완전히 관리
- 자유롭게 구조 변경 가능
"""
from sqlalchemy import Column, String, Text, Integer, ForeignKey, JSON, DECIMAL, BigInteger
from sqlalchemy.orm import relationship
from .base import Base, BaseEntity

class DBProductEmbedding(Base, BaseEntity):
    """상품 임베딩 저장 테이블 - FastAPI 전용"""
    __tablename__ = "product_embeddings"
    
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False, unique=True)
    embedding_vector = Column(JSON, nullable=False, comment="임베딩 벡터 (JSON 배열)")
    text_content = Column(Text, nullable=False, comment="임베딩에 사용된 텍스트")
    model_version = Column(String(100), default="KoSBERT-v1", comment="모델 버전")
    embedding_dimension = Column(Integer, default=768, comment="임베딩 차원")
    
    product = relationship("DBProduct", back_populates="embeddings")

class DBRecommendationResult(Base, BaseEntity):
    """추천 결과 저장 테이블 - FastAPI 전용"""
    __tablename__ = "recommendation_results"
    
    member_id = Column(BigInteger, ForeignKey("members.id"), nullable=False)
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False)
    similarity_score = Column(DECIMAL(5, 4), nullable=False, comment="유사도 점수")
    recommendation_reason = Column(Text, nullable=True, comment="추천 이유")
    confidence_score = Column(DECIMAL(5, 4), nullable=False, comment="신뢰도 점수")
    algorithm_version = Column(String(50), nullable=False, comment="알고리즘 버전")

class DBVectorIndex(Base, BaseEntity):
    """벡터 인덱스 메타데이터 - FastAPI 전용"""
    __tablename__ = "vector_indices"
    
    index_name = Column(String(100), unique=True, nullable=False)
    index_type = Column(String(50), nullable=False, comment="Faiss 인덱스 타입")
    dimension = Column(Integer, nullable=False, comment="벡터 차원")
    total_vectors = Column(Integer, default=0, comment="저장된 벡터 수")
    model_version = Column(String(100), nullable=False, comment="사용된 모델 버전")
    file_path = Column(String(500), nullable=False, comment="인덱스 파일 경로")
    s3_backup_path = Column(String(500), nullable=True, comment="S3 백업 경로")

class DBUserEmbedding(Base, BaseEntity):
    """사용자 임베딩 캐시 테이블 - FastAPI 전용"""
    __tablename__ = "user_embeddings"
    
    member_id = Column(BigInteger, ForeignKey("members.id"), nullable=False, unique=True)
    embedding_vector = Column(JSON, nullable=False, comment="사용자 임베딩 벡터")
    profile_hash = Column(String(64), nullable=False, comment="프로필 해시값 (변경 감지용)")
    model_version = Column(String(100), nullable=False, comment="모델 버전")

