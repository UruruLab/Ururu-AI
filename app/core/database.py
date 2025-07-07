from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import asyncio
from typing import AsyncGenerator
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# 동기 데이터베이스 설정 (기존 코드와의 호환성)
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

# 비동기 데이터베이스 설정 (FastAPI 최적화)
ASYNC_SQLALCHEMY_DATABASE_URL = f"mysql+aiomysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

# 동기 엔진
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=20,
    max_overflow=0,
    echo=settings.DEBUG
)

# 비동기 엔진 생성
async_engine = create_async_engine(
    ASYNC_SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,  # 5분마다 커넥션 재활용
    pool_size=20,
    max_overflow=0,  # 최대 0개의 커넥션을 추가
    echo=settings.DEBUG
)

SesisonLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
) 
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

metadata = MetaData()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """비동기 데이터베이스 세션 생성"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"비동기 데이터베이스 세션 오류: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

def get_sync_db():
    """동기 데이터베이스 세션 생성"""
    db = SesisonLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"동기 데이터베이스 세션 오류: {e}")
        db.rollback()
        raise
    finally:
        db.close()

async def test_database_connection():
    """데이터베이스 연결 테스트"""
    try:
        async with async_engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            logger.info("✅ 데이터베이스 연결 성공")
            return True
    except Exception as e:
        logger.error(f"비동기 데이터베이스 연결 실패: {e}")
        raise

async def check_required_tables():
    """필수 테이블 존재 여부 확인"""
    required_tables = ['prouducts', 'product_options', 'beauty_profile', 'members']

    try:
        async with async_engine.begin() as conn:
            from sqlalchemy import text

            for table in required_tables:
                result = await conn.execute(
                    text(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{settings.DB_NAME}' AND table_name = '{table}'")
                    )
                count = result.scalar()
                
                if count > 0:
                    logger.info(f"테이블 '{table}' 존재")
                else:
                    logger.warning(f"⚠️ 테이블 '{table}' 없음")
    except Exception as e:
        logger.error(f"테이블 존재 여부 확인 중 오류 발생: {e}")
        
async def init_database():
    if await test_database_connection():
        await check_required_tables()
        logger.info("데이터베이스 초기화 완료")
    else:
        logger.error("데이터베이스 초기화 실패: 연결 오류")
        raise Exception("데이터베이스 연결 실패")
    
async def close_database():
    await async_engine.dispose()
    logger.info("데이터베이스 연결 종료")