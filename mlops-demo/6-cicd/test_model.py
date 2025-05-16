import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import SentimentAnalyzer

class TestSentimentModel(unittest.TestCase):
    """Tests for the Sentiment Analysis model."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model = SentimentAnalyzer()
        
        # Define test cases
        self.positive_texts = [
            "This product is excellent and I really love it!",
            "The service was amazing and exceeded my expectations.",
            "I had a great experience with this company.",
            "The quality is outstanding and worth every penny.",
            "This is the best purchase I've made this year."
        ]
        
        self.negative_texts = [
            "This product is terrible and I regret buying it.",
            "The service was awful and below my expectations.",
            "I had a horrible experience with this company.",
            "The quality is poor and not worth the price.",
            "This is the worst purchase I've made this year."
        ]
        
        self.neutral_texts = [
            "The product arrived yesterday.",
            "I received the item as described.",
            "The package contained what I ordered.",
            "I used the service as instructed.",
            "The transaction was completed."
        ]
        
        self.edge_cases = [
            "",  # Empty string
            "!@#$%^&*()",  # Special characters only
            "a" * 1000,  # Very long text
            "good bad good bad good",  # Mixed sentiment
            "EXCELLENT!!!!"  # All caps with punctuation
        ]
    
    def test_positive_sentiment(self):
        """Test that positive texts are correctly classified."""
        for text in self.positive_texts:
            sentiment, confidence = self.model.predict_sentiment(text)
            self.assertEqual(sentiment, "positive", f"Failed on: {text}")
            self.assertGreater(confidence, 0.5, f"Confidence too low for: {text}")
    
    def test_negative_sentiment(self):
        """Test that negative texts are correctly classified."""
        for text in self.negative_texts:
            sentiment, confidence = self.model.predict_sentiment(text)
            self.assertEqual(sentiment, "negative", f"Failed on: {text}")
            self.assertGreater(confidence, 0.5, f"Confidence too low for: {text}")
    
    def test_confidence_range(self):
        """Test that confidence scores are always between 0 and 1."""
        all_texts = self.positive_texts + self.negative_texts + self.neutral_texts + self.edge_cases
        for text in all_texts:
            _, confidence = self.model.predict_sentiment(text)
            self.assertGreaterEqual(confidence, 0.0, f"Confidence below 0 for: {text}")
            self.assertLessEqual(confidence, 1.0, f"Confidence above 1 for: {text}")
    
    def test_edge_cases(self):
        """Test model behavior on edge cases."""
        # Empty string should not crash
        sentiment, confidence = self.model.predict_sentiment("")
        self.assertIn(sentiment, ["positive", "negative"], "Invalid sentiment for empty string")
        
        # Special characters only
        sentiment, confidence = self.model.predict_sentiment("!@#$%^&*()")
        self.assertIn(sentiment, ["positive", "negative"], "Invalid sentiment for special characters")
        
        # Very long text should not crash
        long_text = "a" * 1000
        sentiment, confidence = self.model.predict_sentiment(long_text)
        self.assertIn(sentiment, ["positive", "negative"], "Invalid sentiment for long text")
    
    def test_consistency(self):
        """Test that the model produces consistent results for the same input."""
        for text in self.positive_texts + self.negative_texts:
            first_sentiment, first_confidence = self.model.predict_sentiment(text)
            second_sentiment, second_confidence = self.model.predict_sentiment(text)
            
            self.assertEqual(first_sentiment, second_sentiment, 
                           f"Inconsistent sentiment for: {text}")
            self.assertAlmostEqual(first_confidence, second_confidence, places=5, 
                                 msg=f"Inconsistent confidence for: {text}")

if __name__ == "__main__":
    unittest.main() 