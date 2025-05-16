import re
import matplotlib.pyplot as plt

def visualize_simple(text, output_file):
    """Simple visualization of sentiment word importance."""
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

def calculate_simple_metrics(text):
    """Calculate simple metrics for the text."""
    words = re.findall(r'\w+', text.lower())
    
    positive_words = ['good', 'great', 'excellent', 'best', 'amazing', 'love', 
                     'wonderful', 'fantastic', 'superb', 'enjoyed']
    
    negative_words = ['bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 
                     'disappointing', 'horrible', 'subpar', 'waste']
    
    # Count sentiment words
    positive_count = sum(1 for w in words if w in positive_words)
    negative_count = sum(1 for w in words if w in negative_words)
    
    # Determine sentiment
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = 0.5 + (positive_count - negative_count) * 0.1
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = 0.5 + (negative_count - positive_count) * 0.1
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    metrics = {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "total_words": len(words),
        "sentiment_words_ratio": (positive_count + negative_count) / len(words) if words else 0
    }
    
    return metrics

if __name__ == "__main__":
    print("\n=== Simple Sentiment Explainer Test ===\n")
    
    # Example texts
    positive_text = "The movie was fantastic! I really enjoyed the plot and the acting was superb."
    negative_text = "The customer service was terrible and the product quality was disappointing."
    
    # Calculate metrics
    print("--- Analyzing positive text ---")
    positive_metrics = calculate_simple_metrics(positive_text)
    print(f"Sentiment: {positive_metrics['sentiment']}")
    print(f"Confidence: {positive_metrics['confidence']:.2f}")
    print(f"Positive words: {positive_metrics['positive_count']}")
    print(f"Negative words: {positive_metrics['negative_count']}")
    
    print("\n--- Analyzing negative text ---")
    negative_metrics = calculate_simple_metrics(negative_text)
    print(f"Sentiment: {negative_metrics['sentiment']}")
    print(f"Confidence: {negative_metrics['confidence']:.2f}")
    print(f"Positive words: {negative_metrics['positive_count']}")
    print(f"Negative words: {negative_metrics['negative_count']}")
    
    # Generate visualizations
    visualize_simple(positive_text, "positive_simple.png")
    visualize_simple(negative_text, "negative_simple.png")
    
    print("\nExplanations generated successfully!") 