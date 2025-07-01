from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import time
import psutil
import os
from typing import Dict, List, Any

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
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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
