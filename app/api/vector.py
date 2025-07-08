# app/api/vector.py - 최소화 버전 (정말 필요한 것만)
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any
import logging
from datetime import datetime

from app.services.faiss_service import FaissVectorStore
from app.services.recommendation_service import RecommendationService
from app.core.dependencies import get_vector_store, get_recommendation_service
from app.models.database import DBProduct, DBProductEmbedding
from app.core.database import get_async_db


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vector", tags=["vector-admin"])


@router.get("/status",
            summary="벡터 저장소 상태",
            description="Faiss 벡터 저장소의 기본 상태 정보 (관리자용)")
async def get_vector_status(
    vector_store: FaissVectorStore = Depends(get_vector_store),
    db: AsyncSession = Depends(get_async_db)
):
    """벡터 저장소 기본 상태 - 관리자용"""
    try:
        # 벡터 저장소 기본 정보
        store_stats = vector_store.get_store_stats()
        
        # DB 기본 통계
        active_products_stmt = select(func.count(DBProduct.id)).where(DBProduct.status == "ACTIVE")
        active_result = await db.execute(active_products_stmt)
        active_product_count = active_result.scalar() or 0
        
        db_embeddings_stmt = select(func.count(DBProductEmbedding.id))
        db_result = await db.execute(db_embeddings_stmt)
        db_embedding_count = db_result.scalar() or 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy" if store_stats["index_stats"]["total_vectors"] > 0 else "empty",
            "summary": {
                "total_vectors": store_stats["index_stats"]["total_vectors"],
                "active_products": active_product_count,
                "embedding_coverage": f"{(store_stats['index_stats']['total_vectors'] / max(active_product_count, 1) * 100):.1f}%",
                "index_type": store_stats["index_stats"]["index_type"],
                "last_updated": store_stats["index_stats"]["metadata"].get("last_updated", "unknown")
            },
            "health": {
                "vectors_loaded": store_stats["index_stats"]["total_vectors"] > 0,
                "index_ready": store_stats["index_stats"]["status"] == "ready",
                "sufficient_data": store_stats["index_stats"]["total_vectors"] >= 100  # 최소 임계값
            }
        }
        
    except Exception as e:
        logger.error(f"벡터 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


@router.post("/embeddings/batch",
             summary="배치 임베딩 생성", 
             description="활성화된 모든 상품의 임베딩을 배치로 생성합니다 (관리자 전용)")
async def generate_batch_embeddings(
    background_tasks: BackgroundTasks,
    batch_size: int = 100,
    force_recreate: bool = False,
    db: AsyncSession = Depends(get_async_db)
):
    """배치 임베딩 생성 - 시스템 초기화용"""
    try:
        # 활성화된 상품 수 조회
        stmt = select(func.count(DBProduct.id)).where(DBProduct.status == "ACTIVE")
        result = await db.execute(stmt)
        total_products = result.scalar() or 0
        
        if total_products == 0:
            raise HTTPException(status_code=404, detail="처리할 활성화된 상품이 없습니다")
        
        # 백그라운드 태스크로 처리
        background_tasks.add_task(
            _process_batch_embeddings,
            force_recreate,
            batch_size
        )
        
        return {
            "status": "started",
            "message": f"배치 임베딩 생성이 시작되었습니다",
            "details": {
                "total_products": total_products,
                "batch_size": batch_size,
                "estimated_batches": (total_products + batch_size - 1) // batch_size,
                "force_recreate": force_recreate
            },
            "note": "처리 상태는 /vector/status에서 확인할 수 있습니다"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 임베딩 생성 실패: {str(e)}")


async def _process_batch_embeddings(force_recreate: bool, batch_size: int):
    """배치 임베딩 처리 - 간소화 버전"""
    from app.core.database import AsyncSessionLocal
    from app.core.dependencies import get_recommendation_service
    from sqlalchemy.orm import selectinload
    
    logger.info(f"🔄 배치 임베딩 처리 시작: {batch_size}개씩")
    
    try:
        recommendation_service = await get_recommendation_service()
        processed_count = 0
        
        offset = 0
        while True:
            async with AsyncSessionLocal() as db:
                # 상품 조회
                stmt = (
                    select(DBProduct)
                    .options(selectinload(DBProduct.product_options))
                    .where(DBProduct.status == "ACTIVE")
                    .offset(offset)
                    .limit(batch_size)
                )
                
                result = await db.execute(stmt)
                db_products = result.scalars().all()
                
                if not db_products:
                    break
                
                # 임베딩 생성 및 저장
                embeddings_data = []
                for db_product in db_products:
                    try:
                        product = await recommendation_service.product_converter.db_to_pydantic(db, db_product)
                        processed_text = recommendation_service.product_tower_service.preprocess_product_text(product)
                        embedding = recommendation_service.embedding_service.encode_text(processed_text)
                        
                        embeddings_data.append({
                            "product_id": int(db_product.id),
                            "embedding": embedding,
                            "metadata": {"created_at": datetime.now().isoformat()}
                        })
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"상품 {db_product.id} 임베딩 실패: {e}")
                        continue
                
                # 벡터 저장소에 추가
                if embeddings_data:
                    await recommendation_service.vector_store.add_embeddings(embeddings_data)
                    logger.info(f"✅ 배치 완료: {len(embeddings_data)}개 (총 {processed_count}개)")
                
                offset += batch_size
                
                # 메모리 정리
                import asyncio
                await asyncio.sleep(0.1)
        
        logger.info(f"🎉 배치 임베딩 완료: {processed_count}개 상품 처리")
        
    except Exception as e:
        logger.error(f"배치 임베딩 처리 실패: {e}")


async def cleanup_vector_service():
    """벡터 서비스 리소스 정리"""
    try:
        vector_store = await get_vector_store()
        await vector_store.close()
        logger.info("🔧 벡터 서비스 리소스 정리 완료")
    except Exception as e:
        logger.error(f"벡터 서비스 정리 실패: {e}")
