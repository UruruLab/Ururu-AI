from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from typing import List, Dict, Any
import logging

from app.core.database import get_async_db
from app.models.database import DBProduct, DBProductOption, DBMember, DBBeautyProfile

router = APIRouter(prefix="/database", tags=["database"])
logger = logging.getLogger(__name__)

@router.get("/products")
async def get_products(
    limit: int = 10,
    db: AsyncSession = Depends(get_async_db)
) -> List[Dict[str, Any]]:
    """실제 데이터베이스에서 상품 조회"""
    try:
        # Spring Boot 테이블에서 실제 상품 데이터 조회
        stmt = select(DBProduct).limit(limit)
        result = await db.execute(stmt)
        products = result.scalars().all()
        
        # 데이터 변환
        product_list = []
        for product in products:
            product_list.append({
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "status": product.status,
                "created_at": product.created_at.isoformat() if product.created_at else None
            })
        
        logger.info(f"상품 {len(product_list)}개 조회 완료")
        return product_list
        
    except Exception as e:
        logger.error(f"상품 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터베이스 조회 실패: {str(e)}")
    
@router.get("/stats")
async def get_database_stats(
    db: AsyncSession = Depends(get_async_db)
) -> Dict[str, Any]:
    """데이터베이스 통계 정보"""
    try:
        stats = {}
        
        # 각 테이블별 데이터 수 조회
        tables = [
            ("products", "SELECT COUNT(*) FROM products"),
            ("product_options", "SELECT COUNT(*) FROM product_options WHERE is_deleted = 0"),
            ("members", "SELECT COUNT(*) FROM members WHERE is_deleted = 0"),
            ("beauty_profile", "SELECT COUNT(*) FROM beauty_profile")
        ]
        
        for table_name, query in tables:
            try:
                result = await db.execute(text(query))
                count = result.scalar()
                stats[table_name] = count
            except Exception as e:
                stats[table_name] = f"조회 실패: {str(e)}"
        
        logger.info("데이터베이스 통계 조회 완료")
        return {
            "status": "success",
            "stats": stats,
            "database_name": "ururu"
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")