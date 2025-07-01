from fastapi import FastAPI
from app.api import test_embedding


app = FastAPI(
    title="Beauty Recommendation API",
    description="AI-powered beauty product recommendation system",
    version="1.0.0"
)

app.include_router(test_embedding.router, prefix="/api/test", tags=["model-comparison"])

@app.get("/")
async def root():
    return {"message": "Beauty Recommendation API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "beauty-recommendation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)