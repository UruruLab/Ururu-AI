from fastapi import FastAPI
from app.api import recommendations, admin, product, vector
from app.clients.spring_client import close_spring_client
import logging
from app.core.config import settings
from app.core.database import init_database

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ururu AI Recommendation API",
    description="AI-powered beauty product recommendation system",
    version="1.0.0"
)

# 추천 API 라우터 등록
app.include_router(recommendations.router)
app.include_router(admin.router)
app.include_router(product.router)
app.include_router(vector.router)

@app.get("/")
async def root():
    return {
        "message": "Ururu AI Recommendation API is running!",
        "docs": "/docs",
        "health": "/api/recommendations/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ururu-ai-recommendation",
        "version": "1.0.0"
    }

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 데이터베이스 초기화"""
    try:
        await init_database()
        logger.debug("데이터베이스 초기화 성공!")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")
        if settings.ENVIRONMENT == "production":
            raise 
        else:
            logger.warning("⚠️  데이터베이스 없이 계속 실행합니다...")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 리소스 정리"""
    await close_spring_client()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
