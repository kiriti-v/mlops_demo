import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model():
    """Download and save the DistilBERT model for sentiment analysis."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    print(f"Downloading model: {model_name}")
    
    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save to appropriate directory
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model to: {save_dir}")
    
    # Save the model and tokenizer
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    print("Model download and save complete!")

if __name__ == "__main__":
    download_model() 