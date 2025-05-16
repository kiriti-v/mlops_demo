
import os
import json
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

class SentimentHandler:
    """
    TorchServe handler for sentiment analysis with DistilBERT.
    """
    
    def __init__(self):
        self.initialized = False
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_map = {0: "negative", 1: "positive"}
    
    def initialize(self, context):
        """
        Initialize model and tokenizer.
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load label map if available
        label_map_path = os.path.join(model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
    
    def preprocess(self, data):
        """
        Tokenize input text.
        """
        text = data[0].get("body").get("text", "")
        if not text:
            raise ValueError("No text provided in the request")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return inputs.to(self.device)
    
    def inference(self, inputs):
        """
        Run inference on the preprocessed inputs.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "class_id": predicted_class,
            "class_name": self.label_map.get(predicted_class, str(predicted_class)),
            "confidence": confidence
        }
    
    def postprocess(self, inference_output):
        """
        Return the inference result as JSON.
        """
        return [inference_output]

_service = SentimentHandler()

def handle(data, context):
    """
    TorchServe handler function.
    """
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None
    
    inputs = _service.preprocess(data)
    result = _service.inference(inputs)
    return _service.postprocess(result)
