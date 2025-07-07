from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any
import logging

from app.core.database import get_async_db

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)

@router.get("/database/stats",
            summary="데이터베이스 통계 정보",
            description="관리자용 데이터베이스 통계 정보를 조회합니다. Spring Boot와 AI 관련 테이블들의 레코드 수를 반환합니다.")
async def get_database_stats(
    db: AsyncSession = Depends(get_async_db)
) -> Dict[str, Any]:
    """데이터베이스 통계 정보 (관리자용)"""
    try:
        # Spring Boot 테이블들
        spring_tables = [
            ("products", "SELECT COUNT(*) FROM products"),
            ("product_options", "SELECT COUNT(*) FROM product_options WHERE is_deleted = 0"),
            ("members", "SELECT COUNT(*) FROM members WHERE is_deleted = 0"),
            ("beauty_profile", "SELECT COUNT(*) FROM beauty_profile")
        ]
        
        # AI 전용 테이블들
        ai_tables = [
            ("product_embeddings", "SELECT COUNT(*) FROM product_embeddings"),
            ("recommendation_results", "SELECT COUNT(*) FROM recommendation_results"),
            ("vector_indices", "SELECT COUNT(*) FROM vector_indices")
        ]
        
        # Spring Boot 테이블 통계
        spring_stats = {}
        for table_name, query in spring_tables:
            try:
                result = await db.execute(text(query))
                count = result.scalar()
                spring_stats[table_name] = count
            except Exception as e:
                spring_stats[table_name] = f"조회 실패: {str(e)}"
        
        # AI 테이블 통계
        ai_stats = {}
        for table_name, query in ai_tables:
            try:
                result = await db.execute(text(query))
                count = result.scalar()
                ai_stats[table_name] = count
            except Exception as e:
                ai_stats[table_name] = f"조회 실패: {str(e)}"
        
        return {
            "status": "success",
            "database_name": "ururu",
            "spring_boot_tables": spring_stats,
            "ai_tables": ai_stats,
            "total_tables": len(spring_stats) + len(ai_stats)
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}") from e

@router.get("/database/connection-test",
            summary="데이터베이스 연결 테스트",
            description="관리자용 데이터베이스 연결 상태를 테스트합니다. 성공 시 현재 시간과 연결 상태를 반환합니다.")
async def test_database_connection(
    db: AsyncSession = Depends(get_async_db)
) -> Dict[str, Any]:
    """데이터베이스 연결 테스트 (관리자용)"""
    try:
        result = await db.execute(text("SELECT 1 as test, NOW() as `current_time`"))
        row = result.fetchone()
        
        return {
            "status": "success",
            "message": "데이터베이스 연결 정상",
            "test_result": row.test,
            "server_time": row.current_time.isoformat(),
            "connection_pool": "active"
        }
        
    except Exception as e:
        logger.error(f"연결 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"연결 테스트 실패: {str(e)}") from e