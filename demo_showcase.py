"""
MLOps Demo Showcase - Simulated Execution

This script simulates the execution of our MLOps pipeline to demonstrate
the capabilities without requiring all dependencies to be installed.
"""
import os
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for generated artifacts
ARTIFACTS_DIR = "demo_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}  ".center(80, "="))
    print("=" * 80 + "\n")

def simulate_data_preparation():
    """Simulate data preparation step."""
    print_section_header("DATA PREPARATION")
    
    # Load sample data or create it if not available
    sample_path = "sample_data.csv"
    
    if os.path.exists(sample_path):
        print(f"Loading sample data from {sample_path}")
        data = pd.read_csv(sample_path)
    else:
        print("Creating sample data")
        data = pd.DataFrame({
            'text': [
                "This movie was fantastic! I loved every minute of it.",
                "The restaurant service was terrible and the food was cold.",
                "I had a great experience with customer support, they were very helpful.",
                "The product quality is poor and it broke after one use.",
                "This book is amazing, I couldn't put it down!",
                "The hotel room was dirty and the staff was rude.",
                "The concert was incredible, best show I've seen all year!",
                "My flight was delayed and I missed my connection.",
                "The software is intuitive and easy to use.",
                "This app crashes constantly and loses my data."
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        data.to_csv(sample_path, index=False)
    
    # Simulate train/test split
    train_idx = np.random.choice(len(data), int(len(data) * 0.8), replace=False)
    test_idx = np.array([i for i in range(len(data)) if i not in train_idx])
    
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    
    # Save split data
    train_path = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_path = os.path.join(ARTIFACTS_DIR, "test.csv")
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Data preparation complete:")
    print(f"- Training data: {train_path} ({len(train_data)} samples)")
    print(f"- Testing data: {test_path} ({len(test_data)} samples)")
    
    # Visualize data distribution
    plt.figure(figsize=(10, 5))
    labels = ['Negative', 'Positive']
    counts = [sum(data['label'] == 0), sum(data['label'] == 1)]
    plt.bar(labels, counts, color=['red', 'green'])
    plt.title('Sentiment Distribution')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    
    # Save visualization
    distribution_path = os.path.join(ARTIFACTS_DIR, "data_distribution.png")
    plt.savefig(distribution_path)
    plt.close()
    
    print(f"Generated visualization: {distribution_path}")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "distribution_path": distribution_path,
        "num_train_samples": len(train_data),
        "num_test_samples": len(test_data)
    }

def simulate_model_training(train_path):
    """Simulate model training step with Vertex AI."""
    print_section_header("MODEL TRAINING (VERTEX AI)")
    
    print(f"Loading training data from {train_path}")
    train_data = pd.read_csv(train_path)
    
    # Simulate model training
    print("Training model with Vertex AI...")
    print("- Using pre-trained DistilBERT model")
    print("- Fine-tuning on sentiment data")
    print("- Running on Vertex AI Training service")
    
    # Mock training metrics
    train_accuracy = 0.93
    val_accuracy = 0.89
    
    # Simulate metrics visualization
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, 6))
    train_metrics = [0.65, 0.82, 0.87, 0.91, 0.93]
    val_metrics = [0.63, 0.79, 0.85, 0.87, 0.89]
    
    plt.plot(epochs, train_metrics, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_metrics, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save visualization
    training_viz_path = os.path.join(ARTIFACTS_DIR, "training_metrics.png")
    plt.savefig(training_viz_path)
    plt.close()
    
    # Save mock model path
    model_path = os.path.join(ARTIFACTS_DIR, "sentiment_model")
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Model training complete:")
    print(f"- Model saved to: {model_path}")
    print(f"- Training accuracy: {train_accuracy:.4f}")
    print(f"- Validation accuracy: {val_accuracy:.4f}")
    print(f"- Generated visualization: {training_viz_path}")
    
    return {
        "model_path": model_path,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "training_viz_path": training_viz_path
    }

def simulate_model_evaluation(model_path, test_path):
    """Simulate model evaluation on test data."""
    print_section_header("MODEL EVALUATION")
    
    print(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path)
    
    print(f"Evaluating model from {model_path}")
    
    # Mock evaluation metrics
    metrics = {
        "accuracy": 0.88,
        "precision": 0.92,
        "recall": 0.85,
        "f1_score": 0.88
    }
    
    # Save metrics to file
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Visualize evaluation metrics
    plt.figure(figsize=(10, 5))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.bar(metric_names, metric_values, color='green')
    plt.ylim(0, 1.0)
    plt.title('Model Evaluation Metrics')
    plt.tight_layout()
    
    # Save visualization
    eval_viz_path = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png")
    plt.savefig(eval_viz_path)
    plt.close()
    
    print(f"Model evaluation complete:")
    print(f"- Metrics saved to: {metrics_path}")
    for name, value in metrics.items():
        print(f"- {name.capitalize()}: {value:.4f}")
    print(f"- Generated visualization: {eval_viz_path}")
    
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "eval_viz_path": eval_viz_path
    }

def simulate_responsible_ai(model_path, test_path):
    """Simulate Responsible AI checks."""
    print_section_header("RESPONSIBLE AI")
    
    # Simulate model explanation
    print("Generating model explanations...")
    
    # Example texts
    pos_text = "This product exceeded my expectations and works perfectly!"
    neg_text = "The quality is terrible and customer service was unhelpful."
    
    # Mock token importance scores for positive example
    pos_tokens = ["exceeded", "expectations", "perfectly", "product", "works"]
    pos_scores = [0.42, 0.38, 0.35, 0.12, 0.28]
    
    # Visualize positive example explanation
    plt.figure(figsize=(10, 5))
    plt.barh(pos_tokens, pos_scores, color='green')
    plt.xlabel('Importance Score')
    plt.title('Words Contributing to Positive Sentiment')
    plt.tight_layout()
    
    # Save visualization
    pos_explain_path = os.path.join(ARTIFACTS_DIR, "positive_explanation.png")
    plt.savefig(pos_explain_path)
    plt.close()
    
    # Mock token importance scores for negative example
    neg_tokens = ["terrible", "unhelpful", "quality", "customer", "service"]
    neg_scores = [0.45, 0.40, 0.32, 0.15, 0.18]
    
    # Visualize negative example explanation
    plt.figure(figsize=(10, 5))
    plt.barh(neg_tokens, neg_scores, color='red')
    plt.xlabel('Importance Score')
    plt.title('Words Contributing to Negative Sentiment')
    plt.tight_layout()
    
    # Save visualization
    neg_explain_path = os.path.join(ARTIFACTS_DIR, "negative_explanation.png")
    plt.savefig(neg_explain_path)
    plt.close()
    
    # Simulate bias detection
    print("\nDetecting potential model bias...")
    bias_results = {
        "gender_bias": 0.05,
        "age_bias": 0.07,
        "race_bias": 0.03
    }
    
    # Visualize bias scores
    plt.figure(figsize=(10, 5))
    bias_categories = list(bias_results.keys())
    bias_values = list(bias_results.values())
    
    plt.bar(bias_categories, bias_values, color='blue')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
    plt.ylim(0, 0.2)
    plt.title('Bias Detection Results')
    plt.ylabel('Bias Score')
    plt.legend()
    plt.tight_layout()
    
    # Save visualization
    bias_viz_path = os.path.join(ARTIFACTS_DIR, "bias_detection.png")
    plt.savefig(bias_viz_path)
    plt.close()
    
    # Simulate drift detection
    print("\nSetting up data drift monitoring...")
    drift_thresholds = {
        "prediction_drift": 0.1,
        "feature_drift": 0.15,
        "concept_drift": 0.2
    }
    
    print("All Responsible AI checks passed!")
    print(f"Generated visualizations:")
    print(f"- Positive explanation: {pos_explain_path}")
    print(f"- Negative explanation: {neg_explain_path}")
    print(f"- Bias detection: {bias_viz_path}")
    
    return {
        "explanations": {
            "positive": pos_explain_path,
            "negative": neg_explain_path
        },
        "bias_results": bias_results,
        "drift_thresholds": drift_thresholds,
        "bias_viz_path": bias_viz_path
    }

def simulate_model_deployment(model_path):
    """Simulate model deployment to Vertex AI."""
    print_section_header("MODEL DEPLOYMENT TO VERTEX AI")
    
    print(f"Uploading model from {model_path} to Vertex AI Model Registry...")
    print("Creating Vertex AI Endpoint...")
    print("Deploying model to endpoint...")
    
    # Mock deployment details
    deployment = {
        "model_id": "sentiment_model_1",
        "endpoint_id": "4235711890123456789",
        "endpoint_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/tidal-fusion-399118/locations/us-central1/endpoints/4235711890123456789",
        "region": "us-central1",
        "version": "v1"
    }
    
    # Visualize deployment architecture
    plt.figure(figsize=(12, 6))
    plt.annotate('Client', xy=(0.1, 0.5), xytext=(0.1, 0.5), horizontalalignment='center',
                size=15, bbox=dict(boxstyle='round', fc='lightblue', ec='blue'))
    
    plt.annotate('Cloud Run\nService', xy=(0.4, 0.5), xytext=(0.4, 0.5), horizontalalignment='center',
                size=15, bbox=dict(boxstyle='round', fc='lightgreen', ec='green'))
    
    plt.annotate('Vertex AI\nEndpoint', xy=(0.7, 0.5), xytext=(0.7, 0.5), horizontalalignment='center',
                size=15, bbox=dict(boxstyle='round', fc='#FFCCCC', ec='red'))
    
    # Add arrows
    plt.annotate('', xy=(0.25, 0.5), xytext=(0.1, 0.5), 
                arrowprops=dict(facecolor='black', width=1, headwidth=10))
    
    plt.annotate('', xy=(0.55, 0.5), xytext=(0.4, 0.5), 
                arrowprops=dict(facecolor='black', width=1, headwidth=10))
    
    # Add monitoring and logging
    plt.annotate('Cloud\nMonitoring', xy=(0.4, 0.2), xytext=(0.4, 0.2), horizontalalignment='center',
                size=12, bbox=dict(boxstyle='round', fc='#FFFFCC', ec='gold'))
    
    plt.annotate('Cloud\nLogging', xy=(0.7, 0.2), xytext=(0.7, 0.2), horizontalalignment='center',
                size=12, bbox=dict(boxstyle='round', fc='#FFFFCC', ec='gold'))
    
    plt.axis('off')
    plt.title('Deployment Architecture')
    plt.tight_layout()
    
    # Save visualization
    deploy_viz_path = os.path.join(ARTIFACTS_DIR, "deployment_architecture.png")
    plt.savefig(deploy_viz_path)
    plt.close()
    
    print(f"Model deployment complete:")
    print(f"- Model ID: {deployment['model_id']}")
    print(f"- Endpoint ID: {deployment['endpoint_id']}")
    print(f"- Endpoint URL: {deployment['endpoint_url']}")
    print(f"- Generated visualization: {deploy_viz_path}")
    
    return {
        "deployment": deployment,
        "deploy_viz_path": deploy_viz_path
    }

def generate_grafana_dashboard():
    """Simulate Grafana dashboard generation."""
    print_section_header("MONITORING SETUP (GRAFANA)")
    
    print("Generating Grafana dashboard for monitoring...")
    
    # Create mock monitoring dashboard with 2x2 panels
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Request volume panel
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    requests = [130, 145, 165, 154, 189, 201, 176]
    axs[0, 0].plot(days, requests, 'b-o')
    axs[0, 0].set_title('Request Volume')
    axs[0, 0].set_ylabel('Requests per hour')
    axs[0, 0].grid(True)
    
    # Latency panel
    latencies = [145, 132, 158, 132, 120, 138, 142]
    axs[0, 1].plot(days, latencies, 'g-o')
    axs[0, 1].set_title('Response Latency')
    axs[0, 1].set_ylabel('Milliseconds')
    axs[0, 1].grid(True)
    
    # Error rate panel
    errors = [0.8, 1.2, 0.5, 0.3, 0.9, 1.0, 0.7]
    axs[1, 0].plot(days, errors, 'r-o')
    axs[1, 0].set_title('Error Rate')
    axs[1, 0].set_ylabel('Percentage (%)')
    axs[1, 0].grid(True)
    
    # Prediction distribution panel
    axs[1, 1].pie([65, 35], labels=['Positive', 'Negative'], 
             autopct='%1.1f%%', colors=['green', 'red'])
    axs[1, 1].set_title('Prediction Distribution')
    
    plt.tight_layout()
    
    # Save visualization
    dashboard_path = os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png")
    plt.savefig(dashboard_path)
    plt.close()
    
    # Generate sample Grafana JSON (simplified)
    dashboard_json = {
        "title": "Sentiment Analysis Model Monitoring",
        "description": "Dashboard for ML model monitoring",
        "panels": [
            {"title": "Request Volume", "type": "timeseries"},
            {"title": "Response Latency", "type": "timeseries"},
            {"title": "Error Rate", "type": "timeseries"},
            {"title": "Prediction Distribution", "type": "piechart"},
            {"title": "Data Drift", "type": "timeseries"},
            {"title": "Bias Metrics", "type": "timeseries"}
        ]
    }
    
    # Save dashboard JSON
    json_path = os.path.join(ARTIFACTS_DIR, "grafana_dashboard.json")
    with open(json_path, 'w') as f:
        json.dump(dashboard_json, f, indent=2)
    
    print(f"Monitoring setup complete:")
    print(f"- Dashboard visualization: {dashboard_path}")
    print(f"- Dashboard JSON saved to: {json_path}")
    print("")
    print("To use this dashboard in a real Grafana instance:")
    print("1. Log in to your Grafana instance")
    print("2. Go to Dashboards > Import")
    print("3. Upload the JSON file")
    print("4. Connect to your Prometheus data source")
    
    return {
        "dashboard_path": dashboard_path,
        "json_path": json_path
    }

def simulate_terraform_infra():
    """Simulate Terraform infrastructure setup."""
    print_section_header("INFRASTRUCTURE AS CODE (TERRAFORM)")
    
    print("The project includes Terraform configuration for GCP resources:")
    print("")
    print("- terraform/main.tf: Main infrastructure configuration")
    print("- terraform/variables.tf: Variable definitions")
    print("")
    print("Resources that would be created:")
    print("1. Google Cloud Storage bucket for ML artifacts")
    print("2. Vertex AI API enablement")
    print("3. Service accounts with appropriate permissions")
    print("4. Artifact Registry for container images")
    print("")
    print("To provision this infrastructure:")
    print("$ cd terraform")
    print("$ terraform init")
    print("$ terraform plan")
    print("$ terraform apply")
    
    return {
        "terraform_files": ["terraform/main.tf", "terraform/variables.tf"],
        "resources": ["gcs_bucket", "vertex_ai_api", "service_accounts", "artifact_registry"]
    }

def run_full_demo():
    """Run the full demonstration."""
    print("\n" + "*" * 90)
    print(" SENTIMENT ANALYSIS MLOPS with GCP VERTEX AI & KUBEFLOW ".center(90, "*"))
    print("*" * 90 + "\n")
    
    # Step 1: Data Preparation
    data_results = simulate_data_preparation()
    
    # Step 2: Model Training
    train_results = simulate_model_training(data_results['train_path'])
    
    # Step 3: Model Evaluation
    eval_results = simulate_model_evaluation(
        train_results['model_path'],
        data_results['test_path']
    )
    
    # Step 4: Responsible AI
    responsible_ai_results = simulate_responsible_ai(
        train_results['model_path'],
        data_results['test_path']
    )
    
    # Step 5: Model Deployment
    deployment_results = simulate_model_deployment(train_results['model_path'])
    
    # Step 6: Monitoring
    monitoring_results = generate_grafana_dashboard()
    
    # Step 7: Infrastructure as Code
    terraform_results = simulate_terraform_infra()
    
    print("\n" + "*" * 90)
    print(" DEMO SUMMARY ".center(90, "*"))
    print("*" * 90 + "\n")
    
    print("This demo showcases a complete MLOps solution with:")
    print("1. GCP Vertex AI integration for ML pipeline execution")
    print("2. Kubeflow Pipelines for workflow orchestration")
    print("3. Responsible AI practices (explainability, bias detection, drift monitoring)")
    print("4. Comprehensive monitoring with Grafana dashboards")
    print("5. Infrastructure as Code with Terraform")
    print("\nAll artifacts are available in the '{}' directory".format(ARTIFACTS_DIR))
    
    # Save demo results for reference
    all_results = {
        "data": data_results,
        "training": train_results,
        "evaluation": eval_results,
        "responsible_ai": responsible_ai_results,
        "deployment": deployment_results,
        "monitoring": monitoring_results,
        "terraform": terraform_results
    }
    
    # Save results to JSON
    results_path = os.path.join(ARTIFACTS_DIR, "demo_results.json")
    with open(results_path, 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for category, results in all_results.items():
            serializable_results[category] = {}
            for k, v in results.items():
                serializable_results[category][k] = str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    run_full_demo() 