from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, BigInteger
from sqlalchemy.sql import func

# 공통 Base 클래스
Base = declarative_base()

class BaseEntity:
    """Spring Boot BaseEntity와 동일한 구조"""
    id = Column(BigInteger, primary_key=True, index=True)
    created_at = Column(DateTime, default=func.now(), comment="생성일시")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="수정일시")

