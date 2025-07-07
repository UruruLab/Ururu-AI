from fastapi import FastAPI
from app.api import recommendations, database_test
from app.clients.spring_client import close_spring_client

app = FastAPI(
    title="Ururu AI Recommendation API",
    description="AI-powered beauty product recommendation system",
    version="1.0.0"
)

# 추천 API 라우터 등록
app.include_router(recommendations.router)
app.include_router(database_test.router)

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
        print("데이터베이스 초기화 성공!")
    except Exception as e:
        print(f"데이터베이스 초기화 실패: {e}")
        # 개발 중에는 DB 없이도 실행되도록 (운영에서는 raise로 변경)
        print("⚠️  데이터베이스 없이 계속 실행합니다...")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 리소스 정리"""
    await close_spring_client()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
