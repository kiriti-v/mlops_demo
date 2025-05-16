import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Any, Tuple, Union
import sys
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import html
from IPython.display import HTML, display
import seaborn as sns
import pandas as pd

# Add parent directory to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model.model import SentimentAnalyzer
except ImportError:
    print("Warning: Could not import SentimentAnalyzer. Using built-in DistilBERT model for explanations.")


class SentimentExplainer:
    """Explains sentiment predictions by highlighting influential words."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment explainer.
        
        Args:
            model_name: Hugging Face model name to use for explanations
        """
        print(f"Initializing Sentiment Explainer with model: {model_name}")
        self.model_name = model_name
        
        # Load model and tokenizer directly from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define positive and negative words for simpler analysis
        self.positive_words = set([
            'good', 'great', 'excellent', 'best', 'amazing', 'love', 'wonderful', 'fantastic',
            'outstanding', 'exceptional', 'superb', 'brilliant', 'perfect', 'awesome',
            'favorite', 'lovely', 'impressive', 'happy', 'positive', 'recommend'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 'disappointing',
            'horrible', 'negative', 'mediocre', 'subpar', 'failure', 'waste',
            'unpleasant', 'inferior', 'inadequate', 'rubbish', 'useless', 'sad'
        ])
    
    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment and generate explanation for the prediction.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction and explanation information
        """
        print(f"\nAnalyzing text: '{text}'")
        
        # Get basic token importances using gradient-based approach
        token_importances = self._get_token_importances(text)
        
        # Get prediction
        sentiment, confidence = self._predict(text)
        
        # Simple rule-based word classification
        simple_explanation = self._generate_simple_explanation(text)
        
        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "token_importances": token_importances,
            "simple_explanation": simple_explanation
        }
        
        return result
    
    def _predict(self, text: str) -> Tuple[str, float]:
        """
        Get sentiment prediction for input text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Predict
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
    
    def _get_token_importances(self, text: str) -> List[Dict[str, Any]]:
        """
        Calculate token importances using a gradient-based approach.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with token and importance information
        """
        # Tokenize with return_tensors
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        
        # Get token embeddings
        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax(dim=1)
        
        # Compute gradient
        self.model.zero_grad()
        outputs.logits[0, predicted_class].backward()
        
        # Get gradient at embedding
        embedding_gradients = embeddings.grad
        
        # Calculate importance as L2 norm of gradient for each token
        importances = torch.norm(embedding_gradients, dim=2).squeeze().detach().numpy()
        
        # Create list of tokens with importances
        tokens = []
        for i in range(len(input_ids[0])):
            # Skip special tokens
            if input_ids[0, i] in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                                 self.tokenizer.pad_token_id]:
                continue
            
            token = self.tokenizer.decode([input_ids[0, i]])
            token = token.strip()
            if token:
                tokens.append({
                    "token": token,
                    "importance": importances[i],
                    "position": i
                })
        
        # Sort tokens by importance
        tokens.sort(key=lambda x: x["importance"], reverse=True)
        
        return tokens
    
    def _generate_simple_explanation(self, text: str) -> Dict[str, Any]:
        """
        Generate a simple explanation based on positive/negative word matching.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with simple explanation information
        """
        # Tokenize text into words
        words = re.findall(r'\w+', text.lower())
        
        # Find positive and negative words
        found_positive = [word for word in words if word in self.positive_words]
        found_negative = [word for word in words if word in self.negative_words]
        
        # Count occurrences
        positive_count = len(found_positive)
        negative_count = len(found_negative)
        
        # Determine sentiment based on word counts
        if positive_count > negative_count:
            simple_sentiment = "positive"
            dominant_words = found_positive
            explanation = f"Found {positive_count} positive words vs. {negative_count} negative words"
        elif negative_count > positive_count:
            simple_sentiment = "negative"
            dominant_words = found_negative
            explanation = f"Found {negative_count} negative words vs. {positive_count} positive words"
        else:
            # Equal counts or no sentiment words found
            simple_sentiment = "neutral"
            dominant_words = []
            explanation = f"Found {positive_count} positive words and {negative_count} negative words"
        
        return {
            "sentiment": simple_sentiment,
            "explanation": explanation,
            "positive_words": found_positive,
            "negative_words": found_negative,
            "dominant_words": dominant_words
        }
    
    def visualize_word_importances(self, result: Dict[str, Any], 
                                  max_words: int = 10,
                                  output_file: str = None) -> None:
        """
        Visualize the most important words for the prediction.
        
        Args:
            result: Result dictionary from predict_with_explanation()
            max_words: Maximum number of words to show
            output_file: File path to save the visualization (optional)
        """
        # Get top words
        top_tokens = result["token_importances"][:max_words]
        
        # Extract data for plotting
        tokens = [item["token"] for item in top_tokens]
        importances = [item["importance"] for item in top_tokens]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 6))
        colors = ["blue" if result["sentiment"] == "positive" else "red" for _ in tokens]
        
        # Plot bars
        bars = plt.barh(range(len(tokens)), importances, color=colors)
        
        # Set labels and title
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"Top {max_words} Most Influential Words for {result['sentiment'].capitalize()} Prediction")
        plt.xlabel("Importance Score")
        plt.ylabel("Words")
        plt.gca().invert_yaxis()  # Display most important at the top
        
        plt.tight_layout()
        
        # Save if output file provided
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        
        plt.show()
        plt.close()
    
    def generate_html_explanation(self, result: Dict[str, Any]) -> str:
        """
        Generate HTML with highlighted text based on word importance.
        
        Args:
            result: Result dictionary from predict_with_explanation()
            
        Returns:
            HTML string with highlighted text
        """
        text = result["text"]
        importances = result["token_importances"]
        sentiment = result["sentiment"]
        
        # Create a mapping of tokens to importances
        token_to_importance = {}
        max_importance = 0
        
        for item in importances:
            token = item["token"]
            importance = item["importance"]
            token_to_importance[token] = importance
            max_importance = max(max_importance, importance)
        
        # Normalize importances to 0-1 range if we have any importances
        if max_importance > 0:
            for token in token_to_importance:
                token_to_importance[token] /= max_importance
        
        # Helper function to get color for importance
        def get_color(importance, sentiment):
            if sentiment == "positive":
                # Blue gradient for positive
                return f"rgba(0, 0, 255, {importance:.2f})"
            else:
                # Red gradient for negative
                return f"rgba(255, 0, 0, {importance:.2f})"
        
        # Process text to add highlighting
        words = re.findall(r'\w+|\W+', text)
        colored_text = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in token_to_importance:
                importance = token_to_importance[word_lower]
                color = get_color(importance, sentiment)
                colored_text.append(f'<span style="background-color: {color};">{html.escape(word)}</span>')
            else:
                colored_text.append(html.escape(word))
        
        # Assemble HTML
        result_html = "".join(colored_text)
        legend = (
            f'<div style="margin-top: 10px;">'
            f'<strong>Predicted sentiment: {sentiment}</strong> '
            f'(Confidence: {result["confidence"]:.2f})<br/>'
            f'<span style="color: {"blue" if sentiment == "positive" else "red"};">'
            f'Highlighting indicates words that influenced the prediction.'
            f'</span>'
            f'</div>'
        )
        
        full_html = f'<div style="font-family: Arial, sans-serif; padding: 10px;">{result_html}{legend}</div>'
        
        # Write HTML to file for viewing
        with open("explanation.html", "w") as f:
            f.write(full_html)
        print("HTML explanation saved to explanation.html")
        
        return full_html


def test_explainer():
    """Test the sentiment explainer with example sentences."""
    print("\n=== Testing Sentiment Explainer ===\n")
    
    # Initialize explainer
    explainer = SentimentExplainer()
    
    # Example positive and negative texts
    positive_text = "The movie was fantastic! I really enjoyed the plot and the acting was superb."
    negative_text = "The customer service was terrible and the product quality was disappointing."
    
    # Generate explanations
    print("\n--- Analyzing positive text ---")
    positive_result = explainer.predict_with_explanation(positive_text)
    
    print("\n--- Analyzing negative text ---")
    negative_result = explainer.predict_with_explanation(negative_text)
    
    # Visualize results
    explainer.visualize_word_importances(positive_result, output_file="positive_explanation.png")
    explainer.visualize_word_importances(negative_result, output_file="negative_explanation.png")
    
    # Generate HTML explanations
    positive_html = explainer.generate_html_explanation(positive_result)
    negative_html = explainer.generate_html_explanation(negative_result)
    
    print("\nExplanations generated successfully!")
    
    return positive_result, negative_result


def test_explainer_simple():
    """Test with simple mock predictions if transformers not available."""
    try:
        import torch
        import transformers
        # If we can import these, use the full test
        return test_explainer()
    except ImportError:
        print("Transformers or torch not available, using simplified test...")
        
        # Function for visualization without the explainer
        def visualize_simple(text, output_file):
            # Tokenize text into words
            words = re.findall(r'\w+', text.lower())
            
            positive_words = ['good', 'great', 'excellent', 'best', 'amazing', 'love', 
                             'wonderful', 'fantastic', 'superb', 'enjoyed']
            
            negative_words = ['bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 
                             'disappointing', 'horrible', 'subpar', 'waste']
            
            # Assign importance scores
            importances = []
            tokens = []
            
            for word in set(words):
                if word in positive_words:
                    importance = 0.8
                    tokens.append(word)
                    importances.append(importance)
                elif word in negative_words:
                    importance = 0.7
                    tokens.append(word)
                    importances.append(importance)
            
            # Sort by importance
            pairs = sorted(zip(tokens, importances), key=lambda x: x[1], reverse=True)
            tokens = [p[0] for p in pairs]
            importances = [p[1] for p in pairs]
            
            if not tokens:
                print(f"No sentiment words found in text: '{text}'")
                return
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 6))
            is_positive = sum(1 for w in words if w in positive_words) > sum(1 for w in words if w in negative_words)
            colors = ["blue" if is_positive else "red" for _ in tokens]
            
            # Plot bars
            plt.barh(range(len(tokens)), importances, color=colors)
            
            # Set labels and title
            plt.yticks(range(len(tokens)), tokens)
            plt.title(f"Influential Words for {'Positive' if is_positive else 'Negative'} Sentiment")
            plt.xlabel("Importance Score")
            plt.ylabel("Words")
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(output_file)
            print(f"Simple visualization saved to {output_file}")
            plt.close()
        
        # Example texts
        positive_text = "The movie was fantastic! I really enjoyed the plot and the acting was superb."
        negative_text = "The customer service was terrible and the product quality was disappointing."
        
        # Generate simple visualizations
        visualize_simple(positive_text, "positive_simple.png")
        visualize_simple(negative_text, "negative_simple.png")
        
        print("\nSimple explanations generated!")
        return None, None


if __name__ == "__main__":
    test_explainer_simple() 