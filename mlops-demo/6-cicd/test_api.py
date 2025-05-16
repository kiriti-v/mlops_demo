import unittest
import requests
import json
import time
import os
import statistics
from typing import Dict, Any, List, Tuple

class TestSentimentAPI(unittest.TestCase):
    """Tests for the Sentiment Analysis API."""
    
    def setUp(self):
        """Set up the test environment."""
        # Get API URL from environment variable or use default
        self.api_url = os.environ.get("API_URL", "https://sentiment-analysis-api-monitored-240846069363.us-central1.run.app")
        
        # Ensure URL has no trailing slash
        self.api_url = self.api_url.rstrip("/")
        
        print(f"Running tests against API at: {self.api_url}")
        
        # Test data
        self.test_texts = [
            "This product is excellent and I love it!",
            "The service was terrible and I'm disappointed.",
            "The item arrived on time as expected.",
            "I had a great experience with this company.",
            "The quality is poor and not worth the price."
        ]
    
    def test_health_endpoint(self):
        """Test that the health endpoint returns 200 OK."""
        url = f"{self.api_url}/health"
        response = requests.get(url)
        
        self.assertEqual(response.status_code, 200, "Health endpoint did not return 200")
        data = response.json()
        self.assertEqual(data["status"], "ok", "Health status is not 'ok'")
        self.assertTrue("timestamp" in data, "Health response missing timestamp")
    
    def test_predict_endpoint(self):
        """Test that the predict endpoint correctly classifies text."""
        url = f"{self.api_url}/predict"
        
        for text in self.test_texts:
            payload = {"text": text}
            response = requests.post(url, json=payload)
            
            self.assertEqual(response.status_code, 200, f"Predict endpoint failed for text: {text}")
            data = response.json()
            
            # Check for required fields
            self.assertIn("sentiment", data, "Response missing sentiment field")
            self.assertIn("confidence", data, "Response missing confidence field")
            self.assertIn("text", data, "Response missing text field")
            self.assertIn("timestamp", data, "Response missing timestamp field")
            
            # Check data types
            self.assertIn(data["sentiment"], ["positive", "negative"], "Invalid sentiment value")
            self.assertIsInstance(data["confidence"], float, "Confidence is not a float")
            self.assertGreaterEqual(data["confidence"], 0.0, "Confidence below 0")
            self.assertLessEqual(data["confidence"], 1.0, "Confidence above 1")
    
    def test_error_handling(self):
        """Test API error handling."""
        url = f"{self.api_url}/predict"
        
        # Test missing text field
        response = requests.post(url, json={})
        self.assertEqual(response.status_code, 422, "API should return 422 for missing text")
        
        # Test empty text
        response = requests.post(url, json={"text": ""})
        self.assertEqual(response.status_code, 200, "API should handle empty text")
        
        # Test invalid endpoint
        response = requests.get(f"{self.api_url}/nonexistent_endpoint")
        self.assertEqual(response.status_code, 404, "API should return 404 for invalid endpoint")
    
    def test_performance(self):
        """Test API performance metrics."""
        url = f"{self.api_url}/predict"
        payload = {"text": "This is a simple test of API performance."}
        
        # Measure response times over multiple requests
        response_times = []
        
        for _ in range(5):
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200, "API request failed during performance test")
            response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"\nPerformance Metrics:")
        print(f"  Average Response Time: {avg_response_time:.2f} ms")
        print(f"  Maximum Response Time: {max_response_time:.2f} ms")
        print(f"  Minimum Response Time: {min_response_time:.2f} ms")
        
        # Assert reasonable performance (adjust thresholds as needed)
        self.assertLess(avg_response_time, 2000, "Average response time exceeds threshold")

if __name__ == "__main__":
    unittest.main() 