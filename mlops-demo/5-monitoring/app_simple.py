from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API with Monitoring",
    description="API for sentiment analysis with monitoring capabilities",
    version="1.0.0"
)

# Request model
class SentimentRequest(BaseModel):
    text: str

# Response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    processing_time: float

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    logger.info("Health check called")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Mock sentiment prediction endpoint
    """
    logger.info(f"Prediction request received: {request.text}")
    
    # Input validation
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    # Simulate processing
    start_time = time.time()
    time.sleep(0.5)  # Simulate model processing time
    
    # Mock prediction
    sentiment = "positive" if "good" in request.text.lower() or "great" in request.text.lower() else "negative"
    confidence = 0.98 if sentiment == "positive" else 0.95
    
    processing_time = time.time() - start_time
    
    logger.info(f"Prediction result: {sentiment} with confidence {confidence}")
    
    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        processing_time=processing_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 