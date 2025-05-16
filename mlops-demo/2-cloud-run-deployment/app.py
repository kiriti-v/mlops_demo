# Sentiment Analysis API Service
# Version: 1.0.0
# Last updated: 2024-03-21
# This service provides sentiment analysis using DistilBERT

import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.cloud.logging
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Setup Google Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()

# Load model and tokenizer
MODEL_PATH = os.getenv('MODEL_PATH', 'model/saved_model')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for sentiment prediction."""
    try:
        # Get input text
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "error": "No text provided",
                "timestamp": datetime.utcnow().isoformat()
            }), 422
        
        text = data['text']
        logging.info(f"Received prediction request for text: {text}")

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction and confidence
        predicted_class = torch.argmax(predictions).item()
        confidence = predictions[0][predicted_class].item()
        
        # Map prediction to sentiment
        sentiment = "positive" if predicted_class == 1 else "negative"
        
        response = {
            "sentiment": sentiment,
            "confidence": confidence,
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logging.info(f"Prediction response: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 