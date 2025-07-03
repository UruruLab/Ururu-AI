#!/usr/bin/env python3
"""
Product Tower 기본 구조 테스트 스크립트
"""

from datetime import datetime
from decimal import Decimal
from app.models.product import (
    Product, ProductCreate, ProductCategory, ProductStatus,
    ProductRecommendationRequest, ProductProfile
)


def test_product_models():
    """Product 모델 테스트"""
    print("🧪 Product 모델 테스트...")
    
    # 상품 생성 테스트
    product_create = ProductCreate(
        name="히알루론산 수분 크림",
        brand="라로슈포제",
        description="건성 피부를 위한 깊은 보습 크림입니다. 히알루론산과 세라마이드가 함유되어 있습니다.",
        ingredients="히알루론산, 세라마이드, 글리세린, 니아신아마이드",
        category_main=ProductCategory.SKINCARE,
        category_sub="모이스처라이저",
        base_price=Decimal('35000.0')
    )
    print(f"✅ ProductCreate: {product_create.name}")
    
    # Product 객체 테스트
    product = Product(
        id=1,
        **product_create.model_dump(),
        status=ProductStatus.ACTIVE,
        created_at=datetime.now(),
        options=[]
    )
    print(f"✅ Product: {product.id} - {product.name}")
    
    return product


def test_recommendation_request():
    """추천 요청 모델 테스트"""
    print("\n🧪 추천 요청 모델 테스트...")
    
    request = ProductRecommendationRequest(
        user_diagnosis="건성 피부이고 수분이 부족해요. 민감성 피부라서 자극적이지 않은 제품을 원해요.",
        top_k=5,
        exclude_categories=[ProductCategory.MAKEUP],
        include_categories=[ProductCategory.SKINCARE],
        min_similarity=0.5,
        max_price=Decimal('50000.0')
    )
    
    print(f"✅ 추천 요청:")
    print(f"   - 진단: {request.user_diagnosis[:50]}...")
    print(f"   - Top K: {request.top_k}")
    print(f"   - 제외 카테고리: {request.exclude_categories}")
    print(f"   - 포함 카테고리: {request.include_categories}")
    print(f"   - 최소 유사도: {request.min_similarity}")
    print(f"   - 최대 가격: {request.max_price}")
    
    return True


def test_similarity_calculation():
    """유사도 계산 테스트"""
    print("\n🧪 유사도 계산 테스트...")
    
    # 간단한 NumPy 기반 코사인 유사도 계산 테스트
    import numpy as np
    
    def simple_cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    # 임시 벡터 생성 (실제로는 임베딩 모델에서 생성)
    user_vector = np.array([0.1] * 768)  # 사용자 임베딩
    product_vector = np.array([0.15] * 768)  # 상품 임베딩
    
    similarity = simple_cosine_similarity(user_vector, product_vector)
    print(f"✅ 코사인 유사도: {similarity:.4f}")
    
    # 다른 벡터로 테스트
    user_vector2 = np.array([0.8, 0.6] * 384)
    product_vector2 = np.array([0.7, 0.5] * 384)
    
    similarity2 = simple_cosine_similarity(user_vector2, product_vector2)
    print(f"✅ 코사인 유사도 2: {similarity2:.4f}")
    
    return True


def main():
    """전체 테스트 실행"""
    print("🚀 Product Tower 기본 구조 테스트 시작\n")
    
    try:
        # 모델 테스트
        product = test_product_models()
        
        # 추천 요청 테스트
        test_recommendation_request()
        
        # 유사도 계산 테스트
        test_similarity_calculation()
        
        print("\n🎉 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
