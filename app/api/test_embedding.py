from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import time
import psutil
import gc
import os

router = APIRouter()

# 테스트할 모델 목록
MODELS_TO_TEST = {
    "kosbrt_klue": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko_sroberta": "jhgan/ko-sroberta-multitask", 
    "ko_sbert": "jhgan/ko-sbert-multitask"
}

# 전역 변수로 로딩된 모델들 저장
loaded_models = {}

class ModelComparisonResult(BaseModel):
    model_name: str
    model_key: str
    load_success: bool
    load_time: float
    error_message: str = ""
    embedding_dimension: int = 0
    device: str = ""
    memory_usage_mb: float = 0.0

def get_memory_usage():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def _load_model(model_key: str, model_name: str) -> ModelComparisonResult:
    """모델 로딩을 위한 내부 함수"""
    result = ModelComparisonResult(
        model_name=model_name, 
        model_key=model_key, 
        load_success=False, 
        load_time=0.0)

    try:
        print(f"모델 로딩 시작: {model_name}")
        start_time = time.time()
        start_memory = get_memory_usage()

        # 모델 로드
        model = SentenceTransformer(model_name)

        end_time = time.time()
        end_memory = get_memory_usage()

        # 결과 저장
        result.load_success = True
        result.load_time = round(end_time - start_time, 2)
        result.embedding_dimension = model.get_sentence_embedding_dimension()
        result.device = str(model.device)
        result.memory_usage_mb = round(end_memory - start_memory, 2)

        loaded_models[model_key] = model 
        print(f"{model_name} 모델 로딩 완료")
    except Exception as e:
        result.error_message = str(e)
        print(f"모델 로딩 실패: {model_name}, 에러: {str(e)}")
    
    return result


@router.get(
        "/models/list",
        tags=["모델 테스트"],
        summary="모델 목록 조회",
        description="테스트할 모델 목록을 반환합니다.")
async def get_model_list():
    """테스트할 모델 목록 반환"""
    return {
        "status": "success",
        "models": MODELS_TO_TEST,
        "total_models": len(MODELS_TO_TEST)
    }

@router.get(
    "/models/load-single/{model_key}",
    tags=["모델 테스트"],
    summary="단일 모델 로드",
    description="주어진 모델 키에 해당하는 모델을 로드하고, 로드 성공 여부와 메모리 사용량을 반환합니다."
)
async def load_single_model(model_key: str):
    """단일 모델 로딩 테스트"""
    if model_key not in MODELS_TO_TEST:
        raise HTTPException(status_code=404, detail=f"모델을 찾을 수 없습니다.: {model_key}")
    
    model_name = MODELS_TO_TEST[model_key]
    result = _load_model(model_key, model_name)

    return {
        "status": "success" if result.load_success else "error",
        "result": result.dict()
    }

@router.get(
    "/models/load-all",
    tags=["모델 테스트"],
    summary="모든 모델 로드",
    description="모든 모델을 로드하고, 각 모델의 로드 성공 여부와 메모리 사용량을 반환합니다."
)
async def load_all_models():
    """모든 모델 로딩 테스트"""
    results = []

    for model_key, model_name in MODELS_TO_TEST.items():
        result = _load_model(model_key, model_name)
        results.append(result.dict())
    
    successful_loads = sum(1 for r in results if r['load_success'])

    return {
        "status": "success",
        "summary": {
            "total_models": len(MODELS_TO_TEST),
            "successful_loads": successful_loads,
            "failed_loads": len(MODELS_TO_TEST) - successful_loads
        },
        "results": results
    }

@router.get(
    "/models/loaded",
    tags=["모델 테스트"],
    summary="로딩된 모델 목록",
    description="현재 로딩된 모델들의 목록과 상태를 반환합니다."
)
async def get_loaded_models():
    """현재 로딩된 모델 목록 반환"""
    return {
        "status": "success",
        "loaded_models": list(loaded_models.keys()),
        "total_loaded": len(loaded_models)
    }

@router.get(
    "/models/test-embedding",
    tags=["모델 테스트"],
    summary="모델 임베딩 테스트",
    description="로딩된 모델들에 대해 임베딩 테스트를 수행하고, 결과를 반환합니다."
)
async def test_model_embeddings():
    """로딩된 모든 모델의 임베딩 생성 속도 테스트"""
    if not loaded_models:
        raise HTTPException(status_code=404, detail="로딩된 모델이 없습니다. 먼저 모델을 로드하세요.")
    
    test_sentences = [
        "건성 피부용 보습 크림",
        "민감성 피부 진정 에센스", 
        "지성 피부 모공 관리 토너",
        "복합성 피부 밸런싱 세럼",
        "아토피 피부 저자극 클렌징폼"
    ]

    results = []

    for model_key, model in loaded_models.items():
        try:
            start_time = time.time()
            embeddings = model.encode(test_sentences)
            end_time = time.time()

            result = {
                "model_key": model_key,
                "model_name": MODELS_TO_TEST[model_key],
                "embedding_time": round(end_time - start_time, 4),
                "texts_count": len(test_sentences),
                "avg_time_per_text": round((end_time - start_time) / len(test_sentences), 4),
                "embedding_shape": embeddings.shape,
                "memory_usage_mb": round(get_memory_usage(), 2),
                "success": True
            }
        except Exception as e:
            result = {
                "model_key": model_key,
                "model_name": MODELS_TO_TEST[model_key],
                "error": str(e),
                "success": False
            }
        results.append(result) 
        print(f"{model_key} : {result.get('embedding_time', 'N/A')}초, 성공: {result.get('success', False)}")

    return {
        "status": "success",
        "test_sentences": test_sentences,
        "results": results
    }

@router.delete(
    "/models/clear",
    tags=["모델 테스트"],
    summary="모델 캐시 초기화",
    description="로딩된 모든 모델을 메모리에서 제거하고 캐시를 초기화합니다."
)
async def clear_loaded_models():
    """로딩된 모든 모델 메모리에서 제거"""

    model_count = len(loaded_models)
    loaded_models.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "status": "success",
        "message": f"{model_count}개의 로딩된 모델이 메모리에서 제거되었습니다.",
        "current_memory_mb": round(get_memory_usage(), 2)
    }
    