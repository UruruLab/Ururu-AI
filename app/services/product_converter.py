from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from typing import List, Tuple, Dict, Optional
from decimal import Decimal
import logging

from app.models.product import Product, ProductCategory
from app.models.database import DBProduct, DBProductOption, DBProductCategory, DBCategory


logger = logging.getLogger(__name__)


class CompleteCategoryMapper:
    """완전한 카테고리 매핑 시스템"""
    
    def __init__(self):
        self.main_category_mapping = {
            1: ProductCategory.SKINCARE,      # 스킨케어
            15: ProductCategory.MASK_PACK,    # 마스크팩
            25: ProductCategory.CLEANSING,    # 클렌징
            43: ProductCategory.SUNCARE,      # 선케어
            55: ProductCategory.MAKEUP,       # 메이크업
            81: ProductCategory.PERFUME,      # 향수
            86: ProductCategory.HAIRCARE,     # 헤어케어
            113: ProductCategory.BODYCARE,    # 바디케어
        }
        
        self.keyword_priority_mapping = [
            (["립메이크업", "립틴트", "립스틱", "립라이너", "립케어", "컬러립밤", "립글로스"], ProductCategory.MAKEUP),
            (["베이스메이크업", "쿠션", "파운데이션", "블러셔", "파우더", "팩트", "컨실러", "프라이머", "베이스", "쉐이딩", "하이라이터", "픽서", "BB", "CC"], ProductCategory.MAKEUP),
            (["아이메이크업", "아이라이너", "마스카라", "아이브로우", "아이섀도우", "아이래쉬", "아이픽서"], ProductCategory.MAKEUP),
            (["메이크업"], ProductCategory.MAKEUP),
            
            (["선케어", "선크림", "선스틱", "선쿠션", "선파우더", "선스프레이", "선패치", "태닝", "애프터선"], ProductCategory.SUNCARE),
            
            (["클렌징", "클렌징폼", "클렌징젤", "팩클렌저", "클렌징비누", "클렌징오일", "클렌징밤", "클렌징워터", "클렌징밀크", "클렌징크림", "필링", "스크럽", "페이셜스크럽", "피지클리너", "파우더워시", "립리무버", "아이리무버"], ProductCategory.CLEANSING),
            
            (["마스크팩", "시트팩", "워시오프팩", "모델링팩", "필오프팩", "슬리핑팩", "패드", "페이셜팩", "코팩", "패치"], ProductCategory.MASK_PACK),
            
            (["향수", "액체향수", "고체향수", "바디퍼퓸", "헤어퍼퓸"], ProductCategory.PERFUME),
            
            (["헤어케어", "헤어", "샴푸", "린스", "컨디셔너", "드라이샴푸", "스케일러", "트리트먼트", "헤어팩", "노워시트리트먼트", "두피앰플", "두피토닉", "헤어토닉", "헤어에센스", "헤어세럼", "헤어오일", "염색약", "탈색", "새치염색", "헤어메이크업", "파마", "고데기", "드라이기", "헤어브러시", "스타일링", "컬크림", "컬링에센스", "왁스", "젤", "무스", "헤어스프레이"], ProductCategory.HAIRCARE),
            
            (["바디케어", "바디", "샤워", "입욕", "바디워시", "바디스크럽", "입욕제", "비누", "바디로션", "바디크림", "바디미스트", "바디오일", "핸드케어", "핸드크림", "핸드워시", "풋케어", "풋크림", "풋샴푸", "발냄새제거제", "발각질제거제", "발관리용품", "제모", "왁싱", "면도기", "면도날", "제모크림", "스트립", "제모기기", "쉐이빙", "데오드란트", "데오스틱", "데오롤온", "데오스프레이", "쿨링", "데오시트", "베이비"], ProductCategory.BODYCARE),
            
            (["스킨케어", "스킨", "토너", "에센스", "세럼", "앰플", "크림", "아이크림", "로션", "올인원", "미스트", "픽서", "페이스오일", "스킨케어세트", "스킨케어디바이스"], ProductCategory.SKINCARE),
        ]
        
        self.specific_category_mapping = {
            56: ProductCategory.MAKEUP,  # 립메이크업
            57: ProductCategory.MAKEUP,  # 립틴트
            58: ProductCategory.MAKEUP,  # 립스틱
            59: ProductCategory.MAKEUP,  # 립라이너
            60: ProductCategory.MAKEUP,  # 립케어
            61: ProductCategory.MAKEUP,  # 컬러립밤
            62: ProductCategory.MAKEUP,  # 립글로스
            63: ProductCategory.MAKEUP,  # 베이스메이크업
            64: ProductCategory.MAKEUP,  # 쿠션
            65: ProductCategory.MAKEUP,  # 파운데이션
            66: ProductCategory.MAKEUP,  # 블러셔
            67: ProductCategory.MAKEUP,  # 파우더/팩트
            68: ProductCategory.MAKEUP,  # 컨실러
            69: ProductCategory.MAKEUP,  # 프라이머/베이스
            70: ProductCategory.MAKEUP,  # 쉐이딩
            71: ProductCategory.MAKEUP,  # 하이라이터
            72: ProductCategory.MAKEUP,  # 메이크업 픽서
            73: ProductCategory.MAKEUP,  # BB/CC
            74: ProductCategory.MAKEUP,  # 아이메이크업
            75: ProductCategory.MAKEUP,  # 아이라이너
            76: ProductCategory.MAKEUP,  # 마스카라
            77: ProductCategory.MAKEUP,  # 아이브로우
            78: ProductCategory.MAKEUP,  # 아이섀도우
            79: ProductCategory.MAKEUP,  # 아이래쉬 케어
            80: ProductCategory.MAKEUP,  # 아이 픽서
            
            # 베이비 선케어는 선케어로 분류
            146: ProductCategory.SUNCARE,  # 베이비 > 선케어
        }


class ProductConverter:
    """DB 모델과 Pydantic 모델 간 변환 - 완전한 카테고리 매핑"""
    
    def __init__(self):
        self.category_mapper = CompleteCategoryMapper()
        self._category_cache: Dict[int, Tuple[ProductCategory, str]] = {}
    
    async def get_category_mapping(self, db: AsyncSession, product_id: int) -> Tuple[ProductCategory, str]:
        """완전한 카테고리 매핑 시스템"""
        
        if product_id in self._category_cache:
            return self._category_cache[product_id]
        
        try:
            stmt = text("""
                SELECT c.id, c.name, c.depth, c.path, c.parent_id
                FROM product_categories pc
                JOIN categories c ON pc.category_id = c.id
                WHERE pc.product_id = :product_id
                ORDER BY c.depth ASC, c.order_index ASC
            """)
            
            result = await db.execute(stmt, {"product_id": product_id})
            categories = result.fetchall()
            
            if not categories:
                logger.warning(f"상품 {product_id}에 대한 카테고리를 찾을 수 없습니다")
                result = (ProductCategory.SKINCARE, "기타")
                self._category_cache[product_id] = result
                return result
            
            category_info = []
            deepest_category = ""
            main_category = ProductCategory.SKINCARE  
            
            for cat in categories:
                cat_id, cat_name, depth, path, parent_id = cat
                category_info.append({
                    'id': cat_id,
                    'name': cat_name,
                    'depth': depth,
                    'path': path,
                    'parent_id': parent_id
                })
            
            # 가장 깊은 카테고리명 추출
            max_depth = max(cat['depth'] for cat in category_info)
            deepest_categories = [cat for cat in category_info if cat['depth'] == max_depth]
            if deepest_categories:
                deepest_category = deepest_categories[0]['name']
            
            # 메인 카테고리 결정 (우선순위: 직접매핑 > 메인카테고리ID > 키워드매핑)
            main_category = self._determine_main_category(category_info)
            
            category_names = [cat['name'] for cat in category_info]
            logger.debug(f"상품 {product_id}: {' > '.join(category_names)} -> {main_category.value} > {deepest_category}")
            
            result = (main_category, deepest_category)
            self._category_cache[product_id] = result
            return result
                
        except Exception as e:
            logger.error(f"카테고리 조회 실패 (product_id: {product_id}): {e}")
            result = (ProductCategory.SKINCARE, "기타")
            self._category_cache[product_id] = result
            return result
    
    def _determine_main_category(self, category_info: List[dict]) -> ProductCategory:
        """완전한 메인 카테고리 결정 로직"""
        
        # 1단계: 특정 카테고리 ID 직접 매핑 확인
        for cat in category_info:
            if cat['id'] in self.category_mapper.specific_category_mapping:
                mapped_category = self.category_mapper.specific_category_mapping[cat['id']]
                logger.debug(f"직접 매핑: ID {cat['id']}({cat['name']}) -> {mapped_category.value}")
                return mapped_category
        
        # 2단계: 메인 카테고리 ID 확인 (depth=0)
        for cat in category_info:
            if cat['depth'] == 0 and cat['id'] in self.category_mapper.main_category_mapping:
                mapped_category = self.category_mapper.main_category_mapping[cat['id']]
                logger.debug(f"메인 카테고리 매핑: ID {cat['id']}({cat['name']}) -> {mapped_category.value}")
                return mapped_category
        
        # 3단계: 키워드 기반 우선순위 매핑
        all_category_names = [cat['name'] for cat in category_info]
        category_text = " ".join(all_category_names).lower()
        
        for keywords, category in self.category_mapper.keyword_priority_mapping:
            for keyword in keywords:
                if keyword.lower() in category_text:
                    logger.debug(f"키워드 매핑: '{keyword}' -> {category.value} (카테고리: {all_category_names})")
                    return category
        
        # 4단계: Path 기반 매핑 (최후 수단)
        for cat in category_info:
            path_parts = cat['path'].split('/')
            if len(path_parts) >= 2:
                root_id = int(path_parts[1]) if path_parts[1].isdigit() else None
                if root_id and root_id in self.category_mapper.main_category_mapping:
                    mapped_category = self.category_mapper.main_category_mapping[root_id]
                    logger.debug(f"Path 매핑: {cat['path']} -> {mapped_category.value}")
                    return mapped_category
        
        # 기본값
        logger.debug(f"기본 카테고리 사용: 스킨케어 (카테고리: {all_category_names})")
        return ProductCategory.SKINCARE

    async def db_to_pydantic(self, db: AsyncSession, db_product: DBProduct, 
                           db_options: List[DBProductOption] = None) -> Product:
        """DB 모델을 Pydantic 모델로 변환"""

        # 활성화된 옵션만 필터링
        if db_options is None:
            db_options = [opt for opt in db_product.product_options if opt.is_deleted]
        
        # 완전한 카테고리 매핑 사용
        category_main, category_sub = await self.get_category_mapping(db, db_product.id)
        
        # 첫 번째 옵션에서 정보 추출
        base_price = Decimal("0")
        ingredients = ""
        
        if db_options:
            first_option = db_options[0]
            base_price = Decimal(str(first_option.price))
            ingredients = first_option.full_ingredients or ""
        
            logger.debug(f"상품 {db_product.id} 가격 매핑: {first_option.price} -> {base_price}")
        else:
            logger.warning(f"상품 {db_product.id}: 활성화된 옵션이 없어 가격을 0으로 설정")


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

    def clear_cache(self):
        """카테고리 매핑 캐시 초기화"""
        self._category_cache.clear()
        logger.info("카테고리 매핑 캐시가 초기화되었습니다")

    def get_cache_stats(self) -> dict:
        """캐시 통계 정보"""
        return {
            "cached_products": len(self._category_cache),
            "cache_size": len(self._category_cache)
        }

