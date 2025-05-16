from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import logging
import time
import json
import os

# Import our components
from model.model import SentimentAnalyzer
from logging_middleware import LoggingMiddleware
from cloud_monitoring import CloudMonitoring

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

# Global variables
sentiment_analyzer = None
cloud_monitoring = None

@app.on_event("startup")
async def startup_event():
    """Initialize the sentiment analyzer and monitoring during startup"""
    global sentiment_analyzer, cloud_monitoring
    
    logger.info("Loading Sentiment Analyzer model...")
    start_time = time.time()
    
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds!")
        
        # Initialize cloud monitoring
        try:
            with open('tidal-fusion-399118-9bf0691a3825.json') as f:
                config = json.load(f)
                project_id = config['project_id']
                cloud_monitoring = CloudMonitoring(project_id)
                logger.info("Cloud Monitoring initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Monitoring: {str(e)}")
            cloud_monitoring = None
            
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Send metrics to Cloud Monitoring
        if cloud_monitoring:
            try:
                cloud_monitoring.write_latency_metric(processing_time * 1000, sentiment)
                cloud_monitoring.write_request_metric(sentiment)
                cloud_monitoring.write_confidence_metric(confidence, sentiment)
            except Exception as e:
                logger.error(f"Error sending metrics: {str(e)}")
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add logging middleware
app.add_middleware(LoggingMiddleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 