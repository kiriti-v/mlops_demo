import re
import matplotlib.pyplot as plt

def analyze_sentiment(text):
    """Simple rule-based sentiment analysis."""
    words = re.findall(r'\w+', text.lower())
    
    positive_words = ['good', 'great', 'excellent', 'best', 'amazing', 'love', 
                     'wonderful', 'fantastic', 'superb', 'enjoyed']
    
    negative_words = ['bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 
                     'disappointing', 'horrible', 'subpar', 'waste']
    
    positive_count = sum(1 for w in words if w in positive_words)
    negative_count = sum(1 for w in words if w in negative_words)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = 0.5 + (positive_count - negative_count) * 0.1
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = 0.5 + (negative_count - positive_count) * 0.1
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return sentiment, confidence, positive_count, negative_count

if __name__ == "__main__":
    print("\n=== Simple Sentiment Analysis Test ===\n")
    
    # Test texts
    positive_text = "The movie was fantastic! I really enjoyed the plot and the acting was superb."
    negative_text = "The customer service was terrible and the product quality was disappointing."
    
    # Analyze positive text
    print(f"Text: '{positive_text}'")
    sentiment, confidence, pos_count, neg_count = analyze_sentiment(positive_text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Positive words: {pos_count}")
    print(f"Negative words: {neg_count}")
    
    # Analyze negative text
    print(f"\nText: '{negative_text}'")
    sentiment, confidence, pos_count, neg_count = analyze_sentiment(negative_text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Positive words: {pos_count}")
    print(f"Negative words: {neg_count}")
    
    print("\nAnalysis complete!") 