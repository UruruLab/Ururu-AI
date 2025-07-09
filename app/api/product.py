from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import logging
import time
from typing import List
from datetime import datetime

from app.services.recommendation_service import RecommendationService
from app.core.dependencies import get_product_converter  
from app.services.product_converter import ProductConverter 
from app.core.dependencies import get_recommendation_service
from app.models.database import DBProduct
from app.core.database import get_async_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/products", tags=["products"])


@router.post("/embedding/generate",
             summary="개별 상품 임베딩 생성",
             description="""
             **시스템 초기화용 API** - 새로운 상품이 DB에 추가되었을 때 해당 상품의 임베딩을 생성하여 추천 시스템에 등록합니다.
             
             ## 사용 시나리오:
             1. **신상품 등록**: Spring Boot에서 새 상품 추가 후 이 API 호출
             2. **상품 정보 변경**: 상품명/설명/성분 등이 수정된 경우 임베딩 재생성
             3. **개별 테스트**: 특정 상품의 임베딩이 제대로 생성되는지 확인
             
             ## 추천 시스템과의 관계:
             - 이 API로 생성된 임베딩이 벡터 저장소에 저장됨
             - 사용자가 추천 요청 시 이 벡터들과 유사도 비교하여 상품 추천
             - 임베딩이 없는 상품은 추천 결과에 나오지 않음
             """)
async def generate_product_embedding(
    product_id: int,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    converter: ProductConverter = Depends(get_product_converter),
    db: AsyncSession = Depends(get_async_db)
):
    """상품 임베딩 생성"""
    try:
        logger.info(f"상품 {product_id} 임베딩 생성 시작")
        
        # DB에서 상품 조회 (옵션과 함께)
        stmt = (
            select(DBProduct)
            .options(selectinload(DBProduct.product_options))
            .where(DBProduct.id == product_id)
        )
        result = await db.execute(stmt)
        db_product = result.scalar_one_or_none()
        
        if not db_product:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")
        
        product = await converter.db_to_pydantic(db, db_product)
        processed_text = recommendation_service.product_tower_service.preprocess_product_text(product)
        
        # 임베딩 벡터 생성
        embedding_vector = recommendation_service.embedding_service.encode_text(processed_text)
        
        success = await recommendation_service.vector_store.add_embeddings([{
            "product_id": product_id,
            "embedding": embedding_vector,
            "metadata": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed_text": processed_text[:200],
                "model_version": recommendation_service.embedding_service.get_model_info().get("model_name")

            }
        }])

        if not success:
            raise HTTPException(status_code=500, detail="벡터 저장소 추가 실패")

        return {
            "success": True,
            "message": f"상품 {product_id}({product.name})의 임베딩이 생성되었습니다",
            "product_info": {
                "id": product.id,
                "name": product.name,
                "category": f"{product.category_main.value} > {product.category_sub}",
                "base_price": float(product.base_price)
            },
            "embedding_info": {
                "dimension": len(embedding_vector),
                "text_length": len(processed_text),
                "model_version": recommendation_service.embedding_service.get_model_info().get("model_name"),
                "vector_store_added": success,
                "sample_values": embedding_vector[:5]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"상품 임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}") from e
    
    
@router.get("/recommendation-readiness",
            summary="추천 시스템 준비 상태 확인",
            description="""
            **추천 시스템 상태 점검 API** - 사용자 추천 요청 전에 시스템이 준비되었는지 확인합니다.
            
            ## 사용 시나리오:
            1. **시스템 점검**: 추천 API 호출 전 시스템 상태 확인
            2. **관리자 모니터링**: 얼마나 많은 상품이 추천 가능한 상태인지 확인
            3. **디버깅**: 추천 결과가 적게 나올 때 원인 파악
            
            ## 반환 정보:
            - 임베딩이 생성된 상품 수
            - 전체 활성 상품 수 대비 커버리지
            - 추천 시스템 구성 요소별 상태
            - 예상 추천 품질 수준
            """)
async def get_product_service_status(
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """상품 서비스 상태 조회"""
    try:
        from sqlalchemy import func
        from app.models.database import DBProductOption
        
        # 기본 상품 통계
        active_products_stmt = select(func.count(DBProduct.id)).where(DBProduct.status == "ACTIVE")
        active_result = await db.execute(active_products_stmt)
        active_products = active_result.scalar() or 0
        
        total_products_stmt = select(func.count(DBProduct.id))
        total_result = await db.execute(total_products_stmt)
        total_products = total_result.scalar() or 0
        
        # 상품 옵션 통계
        active_options_stmt = select(func.count(DBProductOption.id)).where(DBProductOption.is_deleted == False)
        options_result = await db.execute(active_options_stmt)
        active_options = options_result.scalar() or 0
        
        # 추천 서비스 통계
        recommendation_stats = recommendation_service.get_recommendation_stats()

        # 벡터 저장소 통계
        vector_stats = recommendation_service.vector_store.get_store_stats()
        indexed_products = vector_stats["index_stats"]["total_vectors"]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service_status": "healthy",
            "product_statistics": {
                "total_products": total_products,
                "active_products": active_products,
                "inactive_products": total_products - active_products,
                "active_options": active_options,
                "average_options_per_product": round(active_options / max(active_products, 1), 2)
            },
            "recommendation_integration": {
                "service_connected": True,
                "vector_store_ready": recommendation_stats["vector_store_stats"]["index_stats"]["status"] == "ready",
                "database_connection": "active",
                "index_type": vector_stats["index_stats"]["index_type"],
                "embedding_model": recommendation_stats["embedding_model"]["model_name"],
                "total_vectors": recommendation_stats["vector_store_stats"]["index_stats"]["total_vectors"]
            },
            "capabilities": {
                "individual_embedding_generation": True,
                "product_text_preprocessing": True,
                "category_mapping": True,
                "price_analysis": True
            }
        }
        
    except Exception as e:
        logger.error(f"상품 서비스 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 상태 조회 실패: {str(e)}") from e
    
@router.get("/embedding/verify/{product_id}",
            summary="개별 상품 추천 가능 여부 확인",
            description="""
            **상품별 추천 등록 상태 확인 API** - 특정 상품이 추천 시스템에 등록되어 있는지 확인합니다.
            
            ## 사용 시나리오:
            1. **상품 등록 확인**: 새로 추가한 상품이 추천 시스템에 반영되었는지 확인
            2. **문제 진단**: 특정 상품이 추천 결과에 나오지 않을 때 원인 파악  
            3. **품질 검증**: 상품의 임베딩이 적절하게 생성되었는지 유사도 테스트
            
            ## 반환 정보:
            - 해당 상품의 추천 시스템 등록 여부
            - 유사한 상품과의 관계 (유사도 점수)
            - 추천 결과에서의 예상 순위
            """)
async def verify_product_embedding(
    product_id: int,
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    db: AsyncSession = Depends(get_async_db)
):
    """개별 상품의 추천 시스템 등록 상태 및 품질 확인"""
    try:
        # 상품 존재 여부 확인
        stmt = select(DBProduct).where(DBProduct.id == product_id)
        result = await db.execute(stmt)
        db_product = result.scalar_one_or_none()
        
        if not db_product:
            raise HTTPException(status_code=404, detail="상품을 찾을 수 없습니다.")
        
        # 벡터 저장소에서 확인 (상품 이름으로 검색해서 자기 자신이 나오는지 테스트)
        test_embedding = recommendation_service.embedding_service.encode_text(db_product.name)
        scores, product_ids = await recommendation_service.vector_store.search_vectors(test_embedding, k=10)
        
        is_indexed = product_id in product_ids
        similarity_score = None
        search_rank = None
        
        if is_indexed:
            # 해당 상품의 유사도 점수와 순위 찾기
            for i, (score, pid) in enumerate(zip(scores, product_ids)):
                if pid == product_id:
                    similarity_score = score
                    search_rank = i + 1
                    break
        
        # 전체 벡터 저장소 상태
        vector_stats = recommendation_service.vector_store.get_store_stats()
        
        return {
            "product_id": product_id,
            "product_name": db_product.name,
            "recommendation_status": {
                "is_indexed": is_indexed,
                "can_be_recommended": is_indexed,
                "recommendation_ready": is_indexed
            },
            "quality_metrics": {
                "self_similarity_score": similarity_score,
                "search_rank_for_own_name": search_rank,
                "quality_assessment": "excellent" if similarity_score and similarity_score > 0.9 else
                                   "good" if similarity_score and similarity_score > 0.7 else
                                   "fair" if similarity_score and similarity_score > 0.5 else "poor"
            } if is_indexed else None,
            "similar_products": [
                {"product_id": pid, "similarity": score} 
                for pid, score in zip(product_ids[:5], scores[:5]) 
                if pid != product_id
            ] if is_indexed else [],
            "system_context": {
                "total_indexed_products": vector_stats["index_stats"]["total_vectors"],
                "index_type": vector_stats["index_stats"]["index_type"],
                "embedding_dimension": vector_stats["settings"]["embedding_dimension"]
            },
            "next_steps": [
                "상품이 추천 시스템에 등록되어 있습니다" if is_indexed else 
                "POST /products/embedding/generate 를 사용하여 이 상품을 추천 시스템에 등록하세요"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"임베딩 검증 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 검증 실패: {str(e)}") from e