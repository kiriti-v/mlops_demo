
import torch
from ts.torch_handler.text_handler import TextHandler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DistilBERTSentimentHandler(TextHandler):
    """Handler for DistilBERT sentiment analysis model"""
    
    def initialize(self, context):
        """Initialize the handler"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.initialized = True
        
    def preprocess(self, data):
        """Tokenize the input text"""
        text_inputs = []
        for row in data:
            text = row.get("data") or row.get("body")
            if text is None:
                text = ""
            text_inputs.append(text)
        
        # Tokenize
        inputs = self.tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt")
        return inputs
    
    def inference(self, inputs):
        """Run prediction"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        return predictions
    
    def postprocess(self, inference_output):
        """Return the class with the highest probability"""
        results = []
        for output in inference_output:
            prob, label = torch.max(output, dim=0)
            # Format results for Vertex AI
            sentiment = "positive" if label.item() == 1 else "negative"
            confidence = prob.item()
            results.append({"sentiment": sentiment, "confidence": float(confidence)})
        return results
