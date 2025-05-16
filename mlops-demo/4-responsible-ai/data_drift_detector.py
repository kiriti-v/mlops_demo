import numpy as np
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import scipy.stats as stats

class DataDriftDetector:
    """Detects drift in text data by comparing distributions of features."""
    
    def __init__(self, 
                 baseline_texts: List[str] = None, 
                 max_features: int = 100,
                 drift_threshold: float = 0.05,
                 p_value_threshold: float = 0.01):
        """
        Initialize the drift detector.
        
        Args:
            baseline_texts: List of texts to establish the baseline distribution
            max_features: Maximum number of features to track
            drift_threshold: Threshold for JS divergence to consider drift
            p_value_threshold: p-value threshold for statistical tests
        """
        self.max_features = max_features
        self.drift_threshold = drift_threshold
        self.p_value_threshold = p_value_threshold
        self.baseline_distribution = None
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        
        if baseline_texts:
            self.fit(baseline_texts)
    
    def fit(self, texts: List[str]) -> None:
        """
        Establish the baseline distribution from a list of texts.
        
        Args:
            texts: List of texts to establish the baseline
        """
        print(f"Establishing baseline from {len(texts)} texts...")
        
        # Extract features
        self.vectorizer.fit(texts)
        baseline_counts = self.vectorizer.transform(texts).toarray().sum(axis=0)
        
        # Calculate baseline distribution
        self.baseline_distribution = baseline_counts / baseline_counts.sum()
        
        # Calculate additional baseline statistics
        self.avg_text_length = np.mean([len(text) for text in texts])
        self.std_text_length = np.std([len(text) for text in texts])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Baseline established. Top 5 features: {', '.join(self.feature_names[:5])}")
    
    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence score (0-1 scale, lower is more similar)
        """
        # Ensure valid probability distributions
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate the average distribution
        m = 0.5 * (p + q)
        
        # Calculate KL divergences
        kl_p_m = np.sum(p * np.log2(p / m, where=(p != 0)))
        kl_q_m = np.sum(q * np.log2(q / m, where=(q != 0)))
        
        # Return JS divergence
        return 0.5 * (kl_p_m + kl_q_m)
    
    def detect_drift(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect drift in a new batch of texts compared to the baseline.
        
        Args:
            texts: List of texts to check for drift
            
        Returns:
            Dictionary with drift metrics and results
        """
        if self.baseline_distribution is None:
            raise ValueError("Baseline distribution not established. Call fit() first.")
        
        print(f"Analyzing {len(texts)} texts for distribution drift...")
        
        # Extract features from new texts
        new_counts = self.vectorizer.transform(texts).toarray().sum(axis=0)
        new_distribution = new_counts / new_counts.sum() if new_counts.sum() > 0 else np.zeros_like(new_counts)
        
        # Calculate JS divergence
        js_divergence = self.jensen_shannon_divergence(self.baseline_distribution, new_distribution)
        
        # Calculate length statistics
        new_lengths = np.array([len(text) for text in texts])
        avg_length = np.mean(new_lengths)
        std_length = np.std(new_lengths)
        
        # Perform chi-square test for word frequency distribution
        chi2_stat, p_value = stats.chisquare(new_counts, self.baseline_distribution * new_counts.sum())
        
        # Determine if drift is detected
        is_drift_detected = (js_divergence > self.drift_threshold) or (p_value < self.p_value_threshold)
        
        # Prepare feature-specific information
        feature_drifts = []
        for i, feature in enumerate(self.feature_names):
            baseline_prob = self.baseline_distribution[i]
            new_prob = new_distribution[i] if i < len(new_distribution) else 0
            rel_change = (new_prob - baseline_prob) / (baseline_prob if baseline_prob > 0 else 1)
            
            if abs(rel_change) > 0.5:  # Report features with significant change
                feature_drifts.append({
                    "feature": feature,
                    "baseline_prob": baseline_prob,
                    "new_prob": new_prob,
                    "relative_change": rel_change
                })
        
        # Sort feature drifts by absolute relative change
        feature_drifts.sort(key=lambda x: abs(x["relative_change"]), reverse=True)
        
        result = {
            "is_drift_detected": is_drift_detected,
            "js_divergence": js_divergence,
            "p_value": p_value,
            "avg_length": avg_length,
            "baseline_avg_length": self.avg_text_length,
            "length_difference": avg_length - self.avg_text_length,
            "feature_drifts": feature_drifts[:10]  # Top 10 drifting features
        }
        
        if is_drift_detected:
            print(f"⚠️ DRIFT DETECTED! JS Divergence: {js_divergence:.4f}, p-value: {p_value:.4f}")
        else:
            print(f"No significant drift detected. JS Divergence: {js_divergence:.4f}, p-value: {p_value:.4f}")
            
        return result
    
    def visualize_drift(self, result: Dict[str, Any]) -> None:
        """
        Visualize the drift detection results.
        
        Args:
            result: The result dictionary from detect_drift()
        """
        if not result.get("feature_drifts"):
            print("No feature drift data to visualize.")
            return
        
        # Extract data for visualization
        features = [item["feature"] for item in result["feature_drifts"]]
        baseline_probs = [item["baseline_prob"] for item in result["feature_drifts"]]
        new_probs = [item["new_prob"] for item in result["feature_drifts"]]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(features))
        width = 0.35
        
        plt.bar(x - width/2, baseline_probs, width, label='Baseline')
        plt.bar(x + width/2, new_probs, width, label='Current')
        
        plt.xlabel('Features')
        plt.ylabel('Probability')
        plt.title('Feature Distribution Drift')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('drift_visualization.png')
        print("Drift visualization saved as 'drift_visualization.png'")
        plt.close()


def test_drift_detector():
    """Test the data drift detector with example data."""
    print("\n=== Testing Data Drift Detector ===\n")
    
    # Create baseline dataset - reviews about electronics
    baseline_texts = [
        "This smartphone has excellent battery life and a great camera.",
        "The laptop is fast and reliable for everyday tasks.",
        "I love the sound quality of these headphones.",
        "The screen resolution on this monitor is crystal clear.",
        "This device is easy to set up and use.",
        "The keyboard is comfortable for long typing sessions.",
        "Battery life could be better but overall a solid device.",
        "The camera takes excellent photos in low light conditions.",
        "This gadget is worth every penny.",
        "The software is intuitive and user-friendly.",
        "Great build quality and sleek design.",
        "The processor handles multitasking with ease.",
        "These wireless earbuds have amazing sound quality.",
        "The touchscreen is responsive and accurate.",
        "I'm impressed with the storage capacity of this device."
    ]
    
    # Create similar test set - still about electronics
    similar_texts = [
        "The battery life on this phone is impressive.",
        "This laptop performs well for productivity tasks.",
        "The sound quality of the speakers is excellent.",
        "I'm happy with the screen resolution.",
        "Setting up this device was straightforward.",
        "Typing on this keyboard feels natural.",
        "Battery performance is solid overall.",
        "Photo quality in various lighting conditions is good.",
        "This product provides good value for money.",
        "The interface is easy to navigate."
    ]
    
    # Create drifted test set - about food instead of electronics
    drifted_texts = [
        "This restaurant serves delicious pasta dishes.",
        "The pizza has a perfect thin crust and fresh toppings.",
        "I love the spicy flavor of their signature sauce.",
        "The dessert menu offers a variety of options.",
        "This cafe has a cozy atmosphere and friendly staff.",
        "The food was served hot and the portions were generous.",
        "I enjoyed the balance of flavors in this dish.",
        "Their breakfast menu includes both sweet and savory options.",
        "The steak was cooked to perfection.",
        "This bakery has the best bread in town."
    ]
    
    # Initialize and fit the detector
    detector = DataDriftDetector(baseline_texts=baseline_texts)
    
    # Test with similar data
    print("\n--- Testing with similar data ---")
    similar_result = detector.detect_drift(similar_texts)
    
    # Test with drifted data
    print("\n--- Testing with drifted data ---")
    drifted_result = detector.detect_drift(drifted_texts)
    
    # Visualize the drifted result
    detector.visualize_drift(drifted_result)
    
    return similar_result, drifted_result


if __name__ == "__main__":
    similar_result, drifted_result = test_drift_detector() 