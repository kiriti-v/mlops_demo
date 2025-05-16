import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import sys
import os
from collections import defaultdict

# Add parent directory to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from model.model import SentimentAnalyzer
except ImportError:
    print("Warning: Could not import SentimentAnalyzer. Using mock model for testing.")
    # Mock SentimentAnalyzer for testing
    class SentimentAnalyzer:
        def __init__(self):
            print("Initializing Mock Sentiment Analyzer...")
            
        def predict_sentiment(self, text):
            # Simple rule-based mock prediction
            positive_words = ['good', 'great', 'excellent', 'best', 'amazing', 'love', 'wonderful']
            negative_words = ['bad', 'terrible', 'worst', 'awful', 'hate', 'poor', 'disappointing']
            
            text_lower = text.lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)
            
            if pos_count > neg_count:
                return "positive", 0.5 + (pos_count - neg_count) * 0.1
            elif neg_count > pos_count:
                return "negative", 0.5 + (neg_count - pos_count) * 0.1
            else:
                # Slightly favor positive predictions (bias for testing)
                return "positive", 0.51


class BiasAnalyzer:
    """Analyzes bias in sentiment predictions across different categories."""
    
    def __init__(self, model=None):
        """
        Initialize the bias analyzer.
        
        Args:
            model: The sentiment analysis model to evaluate
        """
        self.model = model if model else SentimentAnalyzer()
        self.categories = {}
        self.results = {}
        self.metrics = {}
    
    def add_category(self, name: str, texts: List[str]) -> None:
        """
        Add a category of texts for bias analysis.
        
        Args:
            name: Name of the category
            texts: List of texts in this category
        """
        self.categories[name] = texts
        print(f"Added category '{name}' with {len(texts)} texts")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze bias across all registered categories.
        
        Returns:
            Dictionary with bias metrics
        """
        if not self.categories:
            raise ValueError("No categories registered. Use add_category() first.")
        
        # Run predictions for each category
        for category, texts in self.categories.items():
            print(f"\nAnalyzing category: {category}")
            results = []
            
            for text in texts:
                sentiment, confidence = self.model.predict_sentiment(text)
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence
                })
            
            self.results[category] = results
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Print summary
        self._print_summary()
        
        return self.metrics
    
    def _calculate_metrics(self) -> None:
        """Calculate bias metrics from prediction results."""
        overall_metrics = {
            "total_samples": 0,
            "positive_rate": 0,
            "negative_rate": 0,
            "avg_confidence": 0
        }
        
        category_metrics = {}
        
        # Calculate metrics per category
        for category, results in self.results.items():
            positive_count = sum(1 for r in results if r["sentiment"] == "positive")
            negative_count = sum(1 for r in results if r["sentiment"] == "negative")
            total = len(results)
            
            positive_rate = positive_count / total if total > 0 else 0
            negative_rate = negative_count / total if total > 0 else 0
            avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0
            
            # Store category metrics
            category_metrics[category] = {
                "total_samples": total,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
                "avg_confidence": avg_confidence
            }
            
            # Add to overall metrics
            overall_metrics["total_samples"] += total
            overall_metrics["positive_rate"] += positive_count
            overall_metrics["negative_rate"] += negative_count
            overall_metrics["avg_confidence"] += sum(r["confidence"] for r in results)
        
        # Finalize overall metrics
        if overall_metrics["total_samples"] > 0:
            overall_metrics["positive_rate"] /= overall_metrics["total_samples"]
            overall_metrics["negative_rate"] /= overall_metrics["total_samples"]
            overall_metrics["avg_confidence"] /= overall_metrics["total_samples"]
        
        # Calculate bias metrics
        bias_metrics = {}
        fairness_metrics = {}
        
        # Calculate pairwise disparities
        categories = list(category_metrics.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                disparity_key = f"{cat1}_vs_{cat2}"
                positive_disparity = abs(category_metrics[cat1]["positive_rate"] - 
                                       category_metrics[cat2]["positive_rate"])
                
                confidence_disparity = abs(category_metrics[cat1]["avg_confidence"] - 
                                         category_metrics[cat2]["avg_confidence"])
                
                bias_metrics[disparity_key] = {
                    "positive_rate_disparity": positive_disparity,
                    "confidence_disparity": confidence_disparity
                }
        
        # Calculate overall fairness metrics
        positive_rates = [m["positive_rate"] for m in category_metrics.values()]
        max_disparity = max(positive_rates) - min(positive_rates) if positive_rates else 0
        
        fairness_metrics["max_positive_rate_disparity"] = max_disparity
        fairness_metrics["acceptable_threshold"] = 0.1  # Suggested fairness threshold
        fairness_metrics["is_model_fair"] = max_disparity <= fairness_metrics["acceptable_threshold"]
        
        # Store all metrics
        self.metrics = {
            "overall": overall_metrics,
            "categories": category_metrics,
            "bias": bias_metrics,
            "fairness": fairness_metrics
        }
    
    def _print_summary(self) -> None:
        """Print a summary of bias analysis results."""
        fairness = self.metrics["fairness"]
        categories = self.metrics["categories"]
        
        print("\n===== BIAS ANALYSIS SUMMARY =====")
        print(f"Total samples analyzed: {self.metrics['overall']['total_samples']}")
        print(f"Overall positive sentiment rate: {self.metrics['overall']['positive_rate']:.2f}")
        
        print("\nPositive sentiment rates by category:")
        for category, metrics in categories.items():
            print(f"  {category}: {metrics['positive_rate']:.2f}")
        
        print(f"\nMaximum disparity between categories: {fairness['max_positive_rate_disparity']:.4f}")
        print(f"Suggested maximum acceptable disparity: {fairness['acceptable_threshold']:.2f}")
        
        if fairness["is_model_fair"]:
            print("\n✅ Model appears to be FAIR across categories.")
        else:
            print("\n⚠️ Potential BIAS detected - disparities exceed recommended threshold.")
            
        # Identify highest disparity categories
        highest_disparity = 0
        highest_pair = ""
        
        for pair, metrics in self.metrics["bias"].items():
            if metrics["positive_rate_disparity"] > highest_disparity:
                highest_disparity = metrics["positive_rate_disparity"]
                highest_pair = pair
        
        if highest_pair:
            print(f"\nHighest disparity found between {highest_pair}: {highest_disparity:.4f}")
            print("Consider reviewing model performance on these categories.")
    
    def visualize(self) -> None:
        """Create visualizations of bias metrics."""
        if not self.metrics:
            print("No metrics available. Run analyze() first.")
            return
        
        # Extract data for plotting
        categories = list(self.metrics["categories"].keys())
        positive_rates = [self.metrics["categories"][cat]["positive_rate"] for cat in categories]
        negative_rates = [self.metrics["categories"][cat]["negative_rate"] for cat in categories]
        
        # Create bar chart of sentiment distribution by category
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, positive_rates, width, label='Positive')
        plt.bar(x + width/2, negative_rates, width, label='Negative')
        
        plt.xlabel('Categories')
        plt.ylabel('Proportion')
        plt.title('Sentiment Distribution by Category')
        plt.xticks(x, categories)
        plt.ylim(0, 1.0)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('bias_visualization.png')
        print("Bias visualization saved as 'bias_visualization.png'")
        plt.close()


def test_bias_analyzer():
    """Test the bias analyzer with example data."""
    print("\n=== Testing Bias Analyzer ===\n")
    
    # Create test data for different demographic categories
    data = {
        "Professional Subjects": [
            "The quarterly financial report shows promising growth trends.",
            "Our team achieved all project milestones ahead of schedule.",
            "The new management strategy has improved efficiency.",
            "The conference presented valuable networking opportunities.",
            "The merger will create significant shareholder value.",
            "Market analysis indicates favorable conditions for expansion.",
            "The board approved the proposed budget allocations.",
            "Our company ranks highly in employee satisfaction surveys.",
            "The implementation of the new system was successful.",
            "Productivity metrics show positive trends this quarter."
        ],
        "Personal Subjects": [
            "I enjoyed spending time with my family this weekend.",
            "My vacation to the beach was relaxing and rejuvenating.",
            "I'm learning to play guitar in my free time.",
            "The birthday party was fun and everyone had a great time.",
            "I'm excited about moving to my new apartment next month.",
            "The restaurant we tried last night had delicious food.",
            "My garden is blooming with beautiful flowers this spring.",
            "I finished reading an interesting book yesterday.",
            "Our new puppy is adorable and full of energy.",
            "The concert last night exceeded my expectations."
        ],
        "Technical Subjects": [
            "The server upgrade improved response times by 40%.",
            "We identified and fixed a critical bug in the database.",
            "The new algorithm processes data more efficiently.",
            "Our API documentation needs comprehensive updates.",
            "The user interface redesign simplifies navigation significantly.",
            "We're experiencing compatibility issues with legacy systems.",
            "Database queries are taking longer than expected to execute.",
            "The code refactoring eliminated redundant functions.",
            "Mobile performance metrics show inconsistent results.",
            "Unit tests caught several edge case failures."
        ]
    }
    
    # Initialize analyzer with mock model (will use built-in mock if import fails)
    analyzer = BiasAnalyzer()
    
    # Add categories for analysis
    for category, texts in data.items():
        analyzer.add_category(category, texts)
    
    # Run analysis
    analyzer.analyze()
    
    # Create visualization
    analyzer.visualize()


if __name__ == "__main__":
    test_bias_analyzer() 