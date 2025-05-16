from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import sys
import time

# Import the model
from model.model import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using DistilBERT",
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

# Global variable for the model
sentiment_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the sentiment analyzer during startup"""
    global sentiment_analyzer
    logger.info("Loading Sentiment Analyzer model...")
    start_time = time.time()
    try:
        sentiment_analyzer = SentimentAnalyzer()
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize the model")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Predict sentiment for the input text
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        start_time = time.time()
        
        # Input validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Get prediction
        sentiment, confidence = sentiment_analyzer.predict_sentiment(request.text)
        
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 