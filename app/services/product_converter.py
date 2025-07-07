# app/services/product_converter.py - 단순화된 버전
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Tuple
from decimal import Decimal
import logging

from app.models.product import Product, ProductCategory
from app.models.database import DBProduct, DBProductOption, DBProductCategory, DBCategory


logger = logging.getLogger(__name__)


class ProductConverter:
    """DB 모델과 Pydantic 모델 간 변환 (단순 버전)"""
    
    def __init__(self):
        self.category_mapping = {
            "스킨케어": ProductCategory.SKINCARE,
            "메이크업": ProductCategory.MAKEUP,
            "클렌징": ProductCategory.CLEANSING,
            "마스크팩": ProductCategory.MASK_PACK,
            "선케어": ProductCategory.SUNCARE,
            "향수": ProductCategory.PERFUME,
            "헤어케어": ProductCategory.HAIRCARE,
            "바디케어": ProductCategory.BODYCARE,
        }
    
    async def get_categories(self, db: AsyncSession, product_id: int) -> Tuple[ProductCategory, str]:
        """상품의 메인 카테고리와 최하단 카테고리 조회"""
        try:
            # 상품의 모든 카테고리를 depth 순으로 조회
            stmt = (
                select(DBCategory.name, DBCategory.depth)
                .select_from(DBProductCategory)
                .join(DBCategory, DBProductCategory.category_id == DBCategory.id)
                .where(DBProductCategory.product_id == product_id)
                .order_by(DBCategory.depth.desc())  # depth가 높은 순으로 정렬 (최하단부터)
            )
            
            result = await db.execute(stmt)
            categories = result.fetchall()
            
            if not categories:
                return ProductCategory.SKINCARE, "기타"
            
            # 메인 카테고리 찾기 (depth=0)
            main_category = ProductCategory.SKINCARE  # 기본값
            main_category_name = ""
            
            # 최하단 카테고리 (첫 번째가 가장 높은 depth)
            deepest_category = categories[0].name
            
            # depth=0인 메인 카테고리 찾기
            for category_name, depth in categories:
                if depth == 0:
                    main_category_name = category_name
                    main_category = self.category_mapping.get(category_name, ProductCategory.SKINCARE)
                    break
            
            logger.debug(f"상품 {product_id}: 메인='{main_category_name}', 최하단='{deepest_category}'")
            return main_category, deepest_category
                
        except Exception as e:
            logger.warning(f"카테고리 조회 실패 (product_id: {product_id}): {e}")
            return ProductCategory.SKINCARE, "기타"
    

    async def db_to_pydantic(self, db: AsyncSession, db_product: DBProduct, 
                           db_options: List[DBProductOption] = None) -> Product:
        """DB 모델을 Pydantic 모델로 변환 (단순 버전)"""
        
        # 활성화된 옵션만 필터링
        if db_options is None:
            db_options = [opt for opt in db_product.product_options if not opt.is_deleted]
        
        # 카테고리 조회 (메인 + 최하단)
        category_main, category_sub = await self.get_categories(db, db_product.id)
        
        # 첫 번째 옵션에서 정보 추출
        base_price = Decimal("0")
        ingredients = ""
        
        if db_options:
            first_option = db_options[0]
            base_price = Decimal(str(first_option.price))
            ingredients = first_option.full_ingredients or ""
        
        return Product(
            id=int(db_product.id),
            name=db_product.name,
            description=db_product.description or "",
            ingredients=ingredients,
            category_main=category_main,
            category_sub=category_sub,
            base_price=base_price,
            status=db_product.status,
            created_at=db_product.created_at,
            updated_at=db_product.updated_at,
            options=[]
        )


def get_product_converter() -> ProductConverter:
    """ProductConverter 의존성 주입"""
    return ProductConverter()