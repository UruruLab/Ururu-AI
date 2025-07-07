"""
Spring Boot와 공유하는 테이블들
- 읽기 전용으로 사용
- Spring Boot에서 생성/관리하는 테이블들
"""
from sqlalchemy import Column, String, Text, Integer, Boolean, ForeignKey, JSON, BigInteger, DateTime
from sqlalchemy.orm import relationship
from .base import Base, BaseEntity

class DBProduct(Base, BaseEntity):
    """products 테이블"""
    __tablename__ = "products"
    
    name = Column(String(255), nullable=False, comment="상품명")
    description = Column(Text, nullable=False, comment="상품 설명")
    status = Column(String(20), nullable=False, comment="상품 상태")
    
    # 관계 설정 (읽기 전용)
    product_options = relationship("DBProductOption", back_populates="product")
    product_categories = relationship("DBProductCategory", back_populates="product")
    product_notice = relationship("DBProductNotice", back_populates="product", uselist=False)
    product_tags = relationship("DBProductTag", back_populates="product")
    
    # AI 전용 관계
    embeddings = relationship("DBProductEmbedding", back_populates="product")

class DBProductOption(Base, BaseEntity):
    """product_options 테이블"""
    __tablename__ = "product_options"
    
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False)
    name = Column(String(255), nullable=False, comment="옵션명")
    price = Column(Integer, nullable=False, comment="가격")
    image_url = Column(String(500), nullable=True, comment="이미지 URL")
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    full_ingredients = Column(Text, nullable=False, comment="전성분")
    image_hash = Column(String(64), nullable=True, comment="이미지 해시값")
    
    product = relationship("DBProduct", back_populates="product_options")

class DBMember(Base, BaseEntity):
    """members 테이블"""
    __tablename__ = "members"
    
    nickname = Column(String(50), nullable=False, comment="닉네임")
    email = Column(String(255), unique=True, nullable=False, comment="이메일")
    social_provider = Column(String(20), nullable=False, comment="소셜 제공자")
    social_id = Column(String(255), nullable=False, comment="소셜 ID")
    gender = Column(String(10), nullable=False, comment="성별")
    birth = Column(DateTime, nullable=True, comment="생년월일")  
    phone = Column(String(20), nullable=True, comment="전화번호")  
    profile_image = Column(String(500), nullable=True, comment="프로필 이미지")
    role = Column(String(20), nullable=False, comment="역할")  
    point = Column(Integer, nullable=False, default=0, comment="포인트")
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    
    beauty_profile = relationship("DBBeautyProfile", back_populates="member", uselist=False)

class DBBeautyProfile(Base, BaseEntity):
    """beauty_profile 테이블 - Spring Boot 관리"""
    __tablename__ = "beauty_profile"
    
    member_id = Column(BigInteger, ForeignKey("members.id"), nullable=False, unique=True)
    skin_type = Column(String(20), nullable=False, comment="피부 타입")
    skin_tone = Column(String(20), nullable=False, comment="피부 톤")
    concerns = Column(JSON, nullable=False, comment="피부 고민 리스트")
    has_allergy = Column(Boolean, nullable=False, comment="알레르기 여부")
    allergies = Column(JSON, nullable=True, comment="알레르기 성분 리스트")
    interest_categories = Column(JSON, nullable=False, comment="관심 카테고리")
    min_price = Column(Integer, nullable=False, comment="최소 가격")
    max_price = Column(Integer, nullable=False, comment="최대 가격")
    additional_info = Column(Text, nullable=False, comment="추가 정보")
    
    member = relationship("DBMember", back_populates="beauty_profile")

class DBProductCategory(Base):
    """product_categories 테이블 매핑"""
    __tablename__ = "product_categories"
    
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False)
    category_id = Column(BigInteger, ForeignKey("categories.id"), nullable=False)
    
    product = relationship("DBProduct", back_populates="product_categories")

class DBProductNotice(Base):
    """product_notices 테이블 매핑 - Spring Boot Entity와 완전 일치"""
    __tablename__ = "product_notices"
    
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False, unique=True)
    capacity = Column(String(100), nullable=False, comment="용량")  
    spec = Column(String(255), nullable=False, comment="제품 주요 사양")  
    expiry = Column(String(100), nullable=False, comment="사용기한") 
    usage_guide = Column(Text, nullable=False, comment="사용 방법")  
    manufacturer = Column(String(100), nullable=False, comment="화장품 제조업자")  
    responsible_seller = Column(String(100), nullable=False, comment="화장품책임판매업자") 
    country_of_origin = Column(String(50), nullable=False, comment="제조국")  
    functional_cosmetics = Column(Boolean, nullable=False, comment="기능성 여부")
    caution = Column(Text, nullable=False, comment="사용 시 주의사항")
    warranty = Column(Text, nullable=False, comment="품질 보증 기준")
    customer_service_number = Column(String(20), nullable=False, comment="고객센터 번호") 
    
    product = relationship("DBProduct", back_populates="product_notice")

class DBProductTag(Base):
    """productTags 테이블 매핑"""
    __tablename__ = "productTags"  
    
    product_id = Column(BigInteger, ForeignKey("products.id"), nullable=False)
    tag_category_id = Column(BigInteger, ForeignKey("tag_category.id"), nullable=False)
    
    product = relationship("DBProduct", back_populates="product_tags")