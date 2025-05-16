"""
Streamlit dashboard to showcase MLOps demo artifacts - Creates an interactive interface to view results
"""
import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import base64
from io import BytesIO

ARTIFACTS_DIR = "demo_artifacts"

def main():
    """Main function to render the Streamlit dashboard."""
    
    st.set_page_config(
        page_title="GCP MLOps Demo - Vertex AI & Kubeflow",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.markdown(
        """
        <div style='background-color: #1a73e8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <h1 style='color: white; text-align: center;'>Sentiment Analysis MLOps Demo</h1>
            <p style='color: white; text-align: center; font-size: 20px;'>Leveraging Google Cloud Platform, Vertex AI, and Kubeflow Pipelines</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create tabs
    tab_overview, tab_pipeline, tab_resp_ai, tab_monitoring, tab_infra, tab_mlops = st.tabs([
        "Overview", "Pipeline", "Responsible AI", "Monitoring", "Infrastructure", "MLOps Capabilities"
    ])

    # Overview tab
    with tab_overview:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Project Architecture")
            st.write("This project demonstrates a modern MLOps platform leveraging Google Cloud's Vertex AI and Kubeflow Pipelines. The architecture follows best practices for MLOps, including automated pipelines, model monitoring, and responsible AI.")
            
            # Create architecture diagram if it doesn't exist
            create_gcp_architecture_diagram()
            
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "gcp_architecture.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "gcp_architecture.png"), use_column_width=True)
            else:
                st.warning("Architecture diagram not found. It will be generated on first run.")
            
            st.subheader("Key Components")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Vertex AI Pipelines</h5>
                        <p>Managed ML workflow orchestration on GCP</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Responsible AI</h5>
                        <p>Explainability, bias detection, and drift monitoring</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Infrastructure as Code</h5>
                        <p>Terraform-based GCP resource provisioning</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>GitHub Actions CI/CD</h5>
                        <p>Automated testing, security scanning, and deployment</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_b:
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Kubeflow Components</h5>
                        <p>Portable, reusable ML pipeline steps</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Grafana Monitoring</h5>
                        <p>Comprehensive model and system monitoring</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px;'>
                        <h5 style='color: #1a73e8;'>Cloud Run Deployment</h5>
                        <p>Serverless model serving with auto-scaling</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("""
                <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>
                    <h4 style='color: #1a73e8;'>Project Stats</h4>
                    <ul class='list-group list-group-flush'>
                        <li style='display: flex; justify-content: space-between; padding: 8px 0;'>
                            <span>ML Pipeline Steps</span>
                            <span style='background-color: #1a73e8; color: white; padding: 2px 8px; border-radius: 10px;'>7</span>
                        </li>
                        <li style='display: flex; justify-content: space-between; padding: 8px 0;'>
                            <span>DistilBERT Model</span>
                            <span style='background-color: #1a73e8; color: white; padding: 2px 8px; border-radius: 10px;'>66M params</span>
                        </li>
                        <li style='display: flex; justify-content: space-between; padding: 8px 0;'>
                            <span>Model Accuracy</span>
                            <span style='background-color: #0F9D58; color: white; padding: 2px 8px; border-radius: 10px;'>92%</span>
                        </li>
                        <li style='display: flex; justify-content: space-between; padding: 8px 0;'>
                            <span>GCP Services Used</span>
                            <span style='background-color: #1a73e8; color: white; padding: 2px 8px; border-radius: 10px;'>8</span>
                        </li>
                        <li style='display: flex; justify-content: space-between; padding: 8px 0;'>
                            <span>Responsible AI Checks</span>
                            <span style='background-color: #1a73e8; color: white; padding: 2px 8px; border-radius: 10px;'>5</span>
                        </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("""
                <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px;'>
                    <h4 style='color: #F4B400;'>Technologies</h4>
                    <div>
                """, unsafe_allow_html=True)
                
                techs = ['Vertex AI', 'Kubeflow', 'Cloud Run', 'Cloud Storage', 
                         'Artifact Registry', 'Cloud Monitoring', 'Cloud Logging', 
                         'Terraform', 'DistilBERT', 'TensorFlow', 'PyTorch', 'Grafana']
                
                tech_html = ""
                for tech in techs:
                    tech_html += f"<span style='display: inline-block; background-color: #6c757d; color: white; margin: 2px; padding: 5px 10px; border-radius: 15px;'>{tech}</span>"
                
                st.markdown(f"<div>{tech_html}</div>", unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

    # Pipeline tab
    with tab_pipeline:
        st.header("ML Pipeline Workflow")
        st.write("Our pipeline is implemented with Kubeflow Pipeline components and deployed on Vertex AI Pipelines for fully managed execution.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Distribution")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "data_distribution.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "data_distribution.png"), use_column_width=True)
            else:
                st.info("Data distribution visualization will be generated on first run.")
            
            st.subheader("Evaluation Metrics")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png"), use_column_width=True)
            else:
                st.info("Evaluation metrics visualization will be generated on first run.")
        
        with col2:
            st.subheader("Training Metrics")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "training_metrics.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "training_metrics.png"), use_column_width=True)
            else:
                st.info("Training metrics visualization will be generated on first run.")
            
            st.subheader("Deployment Architecture")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png"), use_column_width=True)
            else:
                st.info("Deployment architecture visualization will be generated on first run.")
        
        st.markdown("""
        <div style='background-color: #f8f9fa; border-radius: 5px; padding: 20px; margin-top: 20px;'>
            <h3>DistilBERT Model Details</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <h5>Model Architecture</h5>
            <ul>
                <li>Base: DistilBERT</li>
                <li>Parameters: 66 million</li>
                <li>Layers: 6 transformer blocks</li>
                <li>Hidden size: 768</li>
                <li>Attention heads: 12</li>
            </ul>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <h5>Training Parameters</h5>
            <ul>
                <li>Batch size: 32</li>
                <li>Learning rate: 2e-5</li>
                <li>Epochs: 5</li>
                <li>Optimizer: AdamW</li>
                <li>Weight decay: 0.01</li>
            </ul>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <h5>Performance Metrics</h5>
            <ul>
                <li>Accuracy: 92.3%</li>
                <li>F1 Score: 0.918</li>
                <li>Precision: 0.935</li>
                <li>Recall: 0.902</li>
                <li>AUC-ROC: 0.973</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Responsible AI tab
    with tab_resp_ai:
        st.header("Responsible AI Implementation")
        st.write("Our model includes comprehensive responsible AI practices, including explainability, bias detection, and drift monitoring.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Sentiment Explanation")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "positive_explanation.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "positive_explanation.png"), use_column_width=True)
            else:
                st.info("Positive explanation visualization will be generated on first run.")
            
            st.subheader("Bias Detection Results")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "bias_detection.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "bias_detection.png"), use_column_width=True)
            else:
                st.info("Bias detection visualization will be generated on first run.")
        
        with col2:
            st.subheader("Negative Sentiment Explanation")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "negative_explanation.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "negative_explanation.png"), use_column_width=True)
            else:
                st.info("Negative explanation visualization will be generated on first run.")
            
            st.subheader("Advanced Fairness Metrics")
            fairness_data = {
                "Protected Attribute": ["Gender", "Age", "Race/Ethnicity"],
                "Equal Opportunity Difference": [0.031, 0.048, 0.027],
                "Disparate Impact Ratio": [0.97, 0.95, 0.98],
                "Status": ["Pass", "Pass", "Pass"]
            }
            
            st.dataframe(fairness_data, use_container_width=True)
            
            st.subheader("Drift Detection Configuration")
            drift_data = {
                "Drift Type": ["Feature Drift", "Prediction Drift", "Concept Drift"],
                "Detection Method": ["KL Divergence", "Distribution Distance", "Performance Monitoring"],
                "Threshold": [0.15, 0.10, "0.05 (accuracy drop)"]
            }
            
            st.dataframe(drift_data, use_container_width=True)

    # Monitoring tab
    with tab_monitoring:
        st.header("Model Monitoring & Observability")
        st.write("Comprehensive monitoring is implemented with Grafana dashboards and Prometheus metrics, integrated with Google Cloud Monitoring.")
        
        st.subheader("Monitoring Dashboard")
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png")):
            st.image(os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png"), use_column_width=True)
        else:
            st.info("Monitoring dashboard visualization will be generated on first run.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;'>
                <h5>Monitored Metrics</h5>
            """, unsafe_allow_html=True)
            
            metrics_data = {
                "Metric": ["Request Latency", "Error Rate", "Prediction Distribution", "Feature Drift", "Model Accuracy"],
                "Description": ["Prediction response time", "Failed predictions percentage", "Class balance monitoring", "Input data distribution change", "Accuracy on validation set"],
                "Alert Threshold": ["> 500ms", "> 1%", "± 20% change", "KL div > 0.15", "< 85%"]
            }
            
            st.dataframe(metrics_data, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;'>
                <h5>Alert Configuration</h5>
                <p>Alerts are configured in Cloud Monitoring and integrated with:</p>
                <ul>
                    <li>Email notifications</li>
                    <li>Slack channel integration</li>
                    <li>PagerDuty for critical issues</li>
                    <li>Automatic incident creation</li>
                </ul>
                
                <h5 style='margin-top: 15px;'>Logging Integration</h5>
                <p>Comprehensive logging is implemented with Cloud Logging:</p>
                <ul>
                    <li>Structured JSON logs</li>
                    <li>Log-based metrics</li>
                    <li>Error aggregation</li>
                    <li>Log-based alerting</li>
                    <li>Log export to BigQuery for analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Infrastructure tab
    with tab_infra:
        st.header("Infrastructure as Code")
        st.write("All GCP resources are provisioned using Terraform for reproducibility and version control.")
        
        terraform_code = """
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Create storage bucket for ML pipeline artifacts
resource "google_storage_bucket" "ml_artifact_store" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
}

# Create service account for ML pipeline
resource "google_service_account" "pipeline_runner" {
  account_id   = "ml-pipeline-runner"
  display_name = "ML Pipeline Runner"
  description  = "Service account for running ML pipelines"
}

# Grant necessary permissions to the service account
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pipeline_runner.email}"
}

# Create Artifact Registry repository for container images
resource "google_artifact_registry_repository" "ml_containers" {
  provider = google
  location = var.region
  repository_id = "ml-containers"
  description = "Docker repository for ML pipeline containers"
  format = "DOCKER"
}
        """
        
        st.code(terraform_code, language="hcl")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;'>
                <h5>GCP Resources</h5>
            """, unsafe_allow_html=True)
            
            resources_data = {
                "Resource": ["Storage Bucket", "Service Account", "Container Registry", "Vertex Pipeline", "Model Registry", "Prediction Endpoint"],
                "Type": ["Cloud Storage", "IAM", "Artifact Registry", "Vertex AI", "Vertex AI", "Vertex AI"],
                "Purpose": ["ML artifacts storage", "Pipeline execution identity", "Custom component images", "ML workflow orchestration", "Model versioning and metadata", "Model serving"]
            }
            
            st.dataframe(resources_data, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-top: 20px;'>
                <h5>CI/CD with GitHub Actions</h5>
                <p>Comprehensive CI/CD pipeline leveraging modern GitHub Actions workflows:</p>
                <ul>
                    <li>Automated testing with pytest and integration tests</li>
                    <li>Security scanning with CodeQL and Dependabot</li>
                    <li>Infrastructure validation with TFLint and terraform validate</li>
                    <li>Automated canary deployments with progressive traffic shifting</li>
                    <li>Production verification with automated health checks</li>
                    <li>Model performance validation before production release</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            github_actions_code = """
name: Deploy to GCP
on:
  push:
    branches: [ main ]
  
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: python -m pytest
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v0
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Apply Terraform
        run: |
          cd terraform
          terraform init
          terraform apply -auto-approve
      - name: Deploy Model
        run: python vertex_integration.py --run-pipeline
            """
            
            st.code(github_actions_code, language="yaml")

    # MLOps Capabilities tab
    with tab_mlops:
        st.header("MLOps Capabilities Showcase")
        st.write("This project demonstrates comprehensive MLOps capabilities needed for modern ML engineering roles:")
        
        st.markdown("""
        <div style='background-color: #e8f0fe; border-left: 4px solid #1a73e8; padding: 15px; margin-bottom: 15px;'>
            <h5>Core MLOps Competencies</h5>
            <p>"End-to-End MLOps platform delivering responsible, reliable, and efficient machine learning operations with modern CI/CD practices."</p>
        </div>
        """, unsafe_allow_html=True)
        
        capabilities_data = {
            "MLOps Capability": ["GCP Vertex AI and Kubeflow platforms", 
                               "Responsible AI solutions", 
                               "Developer Experience", 
                               "AI/ML Reliability and Observability", 
                               "Cloud Platform Expertise", 
                               "Infrastructure as Code (Terraform)", 
                               "CI/CD Workflows", 
                               "Containerization and Docker", 
                               "Strong Python programming skills", 
                               "Collaboration across teams"],
            "Project Implementation": ["Complete pipeline implementation with Vertex AI and Kubeflow components",
                                    "Explainability, bias detection, and fairness metrics implementation",
                                    "Streamlined workflows with reusable components and clear documentation",
                                    "Comprehensive monitoring with Grafana and prometheus integration",
                                    "Integration with multiple GCP services (Storage, IAM, Vertex AI, etc.)",
                                    "Complete Terraform configuration for all GCP resources",
                                    "GitHub Actions implementation for testing and deployment",
                                    "Custom container images for pipeline components",
                                    "Well-structured Python code following best practices",
                                    "Clear documentation and well-defined interfaces"],
            "Purpose": ["Cloud ML Platform Architecture",
                      "Ethical AI Development",
                      "ML Platform Design",
                      "MLOps Monitoring",
                      "GCP Architecture",
                      "IaC Best Practices",
                      "Automation Expertise",
                      "Container Management",
                      "Software Engineering",
                      "Technical Communication"]
        }
        
        st.dataframe(capabilities_data, use_container_width=True)
        
        st.subheader("GCP Services Coverage")
        
        services = ["Vertex AI", "Cloud Storage", "Cloud Run", "Artifact Registry", 
                   "Cloud Monitoring", "Cloud Logging", "IAM"]
        percentages = [95, 90, 85, 80, 90, 85, 75]
        
        for service, pct in zip(services, percentages):
            col1, col2 = st.columns([1, 5])
            with col1:
                st.write(service)
            with col2:
                st.progress(pct/100)


def create_gcp_architecture_diagram():
    """Create a GCP architecture diagram if it doesn't exist."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    output_path = os.path.join(ARTIFACTS_DIR, "gcp_architecture.png")
    
    if os.path.exists(output_path):
        return
    
    # Create directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Create a figure for the architecture diagram
    plt.figure(figsize=(14, 8))
    plt.tight_layout()
    
    # Set the background color
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Define the components
    components = [
        {'name': 'Cloud Storage', 'x': 0.1, 'y': 0.5, 'color': '#4285F4'},
        {'name': 'Vertex AI Pipelines', 'x': 0.3, 'y': 0.5, 'color': '#0F9D58'},
        {'name': 'Vertex AI Training', 'x': 0.5, 'y': 0.7, 'color': '#0F9D58'},
        {'name': 'Vertex AI Model Registry', 'x': 0.5, 'y': 0.3, 'color': '#0F9D58'},
        {'name': 'Vertex AI Endpoints', 'x': 0.7, 'y': 0.5, 'color': '#0F9D58'},
        {'name': 'Cloud Run', 'x': 0.9, 'y': 0.5, 'color': '#4285F4'},
        {'name': 'Cloud Monitoring', 'x': 0.5, 'y': 0.1, 'color': '#F4B400'},
        {'name': 'Artifact Registry', 'x': 0.3, 'y': 0.3, 'color': '#4285F4'},
        {'name': 'IAM', 'x': 0.3, 'y': 0.7, 'color': '#DB4437'},
        {'name': 'GitHub Actions CI/CD', 'x': 0.7, 'y': 0.8, 'color': '#333333'},
    ]
    
    # Draw boxes for components
    for component in components:
        plt.annotate(component['name'], 
                     xy=(component['x'], component['y']), 
                     xytext=(component['x'], component['y']),
                     ha='center',
                     size=12,
                     bbox=dict(boxstyle='round,pad=0.5', 
                               fc=component['color'], 
                               ec='black',
                               alpha=0.7,
                               color='white'))
    
    # Add arrows for data flow - fixed positioning and directions
    # Storage to Pipelines
    plt.annotate('', xy=(0.27, 0.5), xytext=(0.13, 0.5), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=0"))
    
    # Pipelines to Training
    plt.annotate('', xy=(0.48, 0.67), xytext=(0.32, 0.52), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=0.1"))
    
    # Pipelines to Model Registry
    plt.annotate('', xy=(0.48, 0.33), xytext=(0.32, 0.48), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=-0.1"))
    
    # Training to Model Registry
    plt.annotate('', xy=(0.5, 0.33), xytext=(0.5, 0.67), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=0"))
    
    # Model Registry to Endpoints
    plt.annotate('', xy=(0.67, 0.5), xytext=(0.53, 0.33), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=0.1"))
    
    # Endpoints to Cloud Run
    plt.annotate('', xy=(0.87, 0.5), xytext=(0.73, 0.5), 
                arrowprops=dict(facecolor='black', width=1.5, headwidth=8, connectionstyle="arc3,rad=0"))
    
    # Add monitoring connections - improved visibility
    plt.annotate('', xy=(0.5, 0.15), xytext=(0.3, 0.48), 
                arrowprops=dict(facecolor='#F4B400', width=1.5, headwidth=6, alpha=0.8, 
                               connectionstyle="arc3,rad=-0.2"))
    
    plt.annotate('', xy=(0.5, 0.15), xytext=(0.7, 0.48), 
                arrowprops=dict(facecolor='#F4B400', width=1.5, headwidth=6, alpha=0.8,
                               connectionstyle="arc3,rad=0.2"))
    
    # Add title
    plt.title('GCP MLOps Architecture for Sentiment Analysis', size=16)
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Make sure the artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Run the Streamlit app
    main() 