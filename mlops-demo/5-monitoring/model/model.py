from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        # Load pre-trained model and tokenizer
        print("Loading DistilBERT model and tokenizer...")
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        print("Model and tokenizer loaded successfully!")

        # Set model to evaluation mode
        self.model.eval()

    def predict_sentiment(self, text):
        """
        Predict sentiment for the input text.
        Returns: tuple (sentiment, confidence_score)
        """
        print(f"\nAnalyzing text: '{text}'")

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        print("Text tokenized and encoded")

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        # Convert prediction to sentiment label
        sentiment = "positive" if prediction.item() == 1 else "negative"
        confidence_score = confidence.item()

        print(f"Predicted sentiment: {sentiment}")
        print(f"Confidence score: {confidence_score:.4f}")

        return sentiment, confidence_score

def test_model():
    """
    Test the sentiment analyzer with example sentences
    """
    print("Initializing Sentiment Analyzer for testing...")
    analyzer = SentimentAnalyzer()

    test_sentences = [
        "I absolutely loved this movie, it was fantastic!",
        "The service was terrible and the food was cold.",
        "The product is okay, but could be better.",
        "This is the best purchase I've ever made!"
    ]

    print("\nRunning sentiment analysis tests...")
    for sentence in test_sentences:
        sentiment, confidence = analyzer.predict_sentiment(sentence)
        print("-" * 50)
        print(f"Test sentence: {sentence}")
        print(f"Result: {sentiment} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    test_model() 