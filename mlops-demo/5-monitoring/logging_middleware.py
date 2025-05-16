from fastapi import Request
import time
import logging
import json
from typing import Callable
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingMiddleware:
    def __init__(self):
        self.logger = logger

    async def __call__(self, request: Request, call_next: Callable):
        # Record request start time
        start_time = time.time()
        
        # Get request body
        body = await self.get_request_body(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        text_length = len(body.get("text", "")) if body else 0
        
        # Log request details
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": round(process_time, 2),
            "text_length": text_length,
        }
        
        # Add response data for prediction endpoint
        if request.url.path == "/predict" and response.status_code == 200:
            response_body = await self.get_response_body(response)
            log_data.update({
                "prediction": response_body.get("sentiment"),
                "confidence": response_body.get("confidence")
            })
        
        self.logger.info(f"Request processed: {json.dumps(log_data)}")
        
        return response
    
    async def get_request_body(self, request: Request) -> dict:
        """Get request body if available"""
        try:
            body = await request.json()
            return body
        except:
            return {}
    
    async def get_response_body(self, response) -> dict:
        """Get response body if available"""
        try:
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk
            
            # Restore response body for downstream middleware
            response.body_iterator = aiter([body_bytes])
            
            return json.loads(body_bytes)
        except:
            return {} 