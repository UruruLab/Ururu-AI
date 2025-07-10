from fastapi import FastAPI
from app.api import recommendations, admin, product, vector
from app.clients.spring_client import close_spring_client
import logging
import sys
from app.core.config import settings
from app.core.database import init_database

def setup_logging():
    """로깅 설정 - DEBUG 레벨이 제대로 출력되도록 설정"""
    
    # 1. 환경변수에서 로그 레벨 가져오기
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # 2. 로그 포매터 설정
    formatter = logging.Formatter(
        fmt=getattr(settings, 'LOG_FORMAT', 
                   "%(asctime)s [%(processName)s:%(process)d] [%(levelname)s] %(name)s: %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 3. 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # 4. 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # 기존 핸들러 제거
    root_logger.addHandler(console_handler)
    
    # 5. Uvicorn 로거들 설정
    uvicorn_loggers = [
        "uvicorn",
        "uvicorn.error", 
        "uvicorn.access"
    ]
    
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.propagate = False  # 중복 로그 방지
    
    # 6. FastAPI 앱 로거들 설정
    app_loggers = [
        "app",
        "app.api",
        "app.services", 
        "app.core"
    ]
    
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        # 루트 로거로 전파하도록 설정 (중복 방지를 위해 핸들러는 추가하지 않음)
        logger.propagate = True
    
    print(f"✅ 로깅 설정 완료: {log_level_str} 레벨")
    
    # 7. 테스트 로그 출력
    test_logger = logging.getLogger(__name__)
    test_logger.debug("🐛 DEBUG 레벨 테스트 로그")
    test_logger.info("ℹ️ INFO 레벨 테스트 로그") 
    test_logger.warning("⚠️ WARNING 레벨 테스트 로그")


setup_logging()
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
