from fastapi import FastAPI

app = FastAPI(
    title="Beauty Recommendation API",
    description="AI-powered beauty product recommendation system",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Beauty Recommendation API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "beauty-recommendation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)