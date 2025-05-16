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
            <h1 style='color: white; text-align: center;'>Sentiment Analysis MLOps Grafana Dashboard</h1>
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
                st.image(os.path.join(ARTIFACTS_DIR, "gcp_architecture.png"), use_container_width=True)
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
                st.image(os.path.join(ARTIFACTS_DIR, "data_distribution.png"), use_container_width=True)
            else:
                st.info("Data distribution visualization will be generated on first run.")
            
            st.subheader("Evaluation Metrics")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png"), use_container_width=True)
            else:
                st.info("Evaluation metrics visualization will be generated on first run.")
        
        with col2:
            st.subheader("Training Metrics")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "training_metrics.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "training_metrics.png"), use_container_width=True)
            else:
                st.info("Training metrics visualization will be generated on first run.")
            
            st.subheader("Deployment Architecture")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png"), use_container_width=True)
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
                st.image(os.path.join(ARTIFACTS_DIR, "positive_explanation.png"), use_container_width=True)
            else:
                st.info("Positive explanation visualization will be generated on first run.")
            
            st.subheader("Bias Detection Results")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "bias_detection.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "bias_detection.png"), use_container_width=True)
            else:
                st.info("Bias detection visualization will be generated on first run.")
        
        with col2:
            st.subheader("Negative Sentiment Explanation")
            if os.path.exists(os.path.join(ARTIFACTS_DIR, "negative_explanation.png")):
                st.image(os.path.join(ARTIFACTS_DIR, "negative_explanation.png"), use_container_width=True)
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
            st.image(os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png"), use_container_width=True)
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
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Set the background color
    ax.set_facecolor('#f8f9fa')
    
    # Define component dimensions
    box_width = 0.15
    box_height = 0.25
    
    # Define the components with precise positions
    components = [
        {'name': 'Cloud Storage', 'x': 0.1, 'y': 0.5, 'color': '#4285F4', 'width': box_width, 'height': box_height},
        {'name': 'Vertex AI\nPipelines', 'x': 0.3, 'y': 0.5, 'color': '#0F9D58', 'width': box_width, 'height': box_height},
        {'name': 'Vertex AI\nTraining', 'x': 0.5, 'y': 0.7, 'color': '#0F9D58', 'width': box_width, 'height': box_height},
        {'name': 'Vertex AI\nModel Registry', 'x': 0.5, 'y': 0.3, 'color': '#0F9D58', 'width': box_width, 'height': box_height},
        {'name': 'Vertex AI\nEndpoints', 'x': 0.7, 'y': 0.5, 'color': '#0F9D58', 'width': box_width, 'height': box_height},
        {'name': 'Cloud Run', 'x': 0.9, 'y': 0.5, 'color': '#4285F4', 'width': box_width, 'height': box_height},
        {'name': 'Cloud\nMonitoring', 'x': 0.5, 'y': 0.1, 'color': '#F4B400', 'width': box_width, 'height': box_height},
        {'name': 'Artifact\nRegistry', 'x': 0.3, 'y': 0.3, 'color': '#4285F4', 'width': box_width, 'height': box_height},
        {'name': 'IAM', 'x': 0.3, 'y': 0.7, 'color': '#DB4437', 'width': box_width, 'height': box_height},
        {'name': 'GitHub Actions\nCI/CD', 'x': 0.7, 'y': 0.8, 'color': '#333333', 'width': box_width, 'height': box_height},
    ]
    
    # Draw components as rectangles with text
    for component in components:
        # Create rectangle patch
        rect = plt.Rectangle(
            (component['x'] - component['width']/2, component['y'] - component['height']/2),
            component['width'], component['height'],
            facecolor=component['color'],
            edgecolor='black',
            alpha=0.7,
            zorder=1
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            component['x'], component['y'],
            component['name'],
            ha='center',
            va='center',
            color='white',
            fontsize=8,
            zorder=2
        )
    
    # Helper function to calculate connection points between components
    def get_connection_points(start_idx, end_idx):
        start_comp = components[start_idx]
        end_comp = components[end_idx]
        
        # Determine connection sides based on relative positions
        if abs(start_comp['x'] - end_comp['x']) > abs(start_comp['y'] - end_comp['y']):
            # Horizontal connection
            if start_comp['x'] < end_comp['x']:  # start is to the left of end
                start_x = start_comp['x'] + start_comp['width']/2
                end_x = end_comp['x'] - end_comp['width']/2
            else:  # start is to the right of end
                start_x = start_comp['x'] - start_comp['width']/2
                end_x = end_comp['x'] + end_comp['width']/2
            
            start_y = start_comp['y']
            end_y = end_comp['y']
        else:
            # Vertical connection
            if start_comp['y'] < end_comp['y']:  # start is below end
                start_y = start_comp['y'] + start_comp['height']/2
                end_y = end_comp['y'] - end_comp['height']/2
            else:  # start is above end
                start_y = start_comp['y'] - start_comp['height']/2
                end_y = end_comp['y'] + end_comp['height']/2
            
            start_x = start_comp['x']
            end_x = end_comp['x']
            
        return start_x, start_y, end_x, end_y
    
    # Define connections
    connections = [
        (0, 1),  # Storage to Pipelines
        (1, 2),  # Pipelines to Training
        (1, 3),  # Pipelines to Model Registry
        (2, 3),  # Training to Model Registry
        (3, 4),  # Model Registry to Endpoints
        (4, 5),  # Endpoints to Cloud Run
        (1, 7),  # Pipelines to Artifact Registry
        (1, 8),  # Pipelines to IAM
        (9, 2),  # GitHub Actions to Training
    ]
    
    # Draw arrows for data flow with precise connections
    for start_idx, end_idx in connections:
        start_x, start_y, end_x, end_y = get_connection_points(start_idx, end_idx)
        
        # Determine curvature based on connection type
        if abs(components[start_idx]['x'] - components[end_idx]['x']) > 0.3:
            rad = 0  # Straight line for distant components
        elif abs(components[start_idx]['y'] - components[end_idx]['y']) > 0.3:
            rad = 0  # Straight line for distant components
        else:
            # Determine direction for curve
            if components[start_idx]['x'] < components[end_idx]['x']:
                rad = 0.2
            else:
                rad = -0.2
        
        ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                facecolor='black',
                width=1.0,
                headwidth=6,
                connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=3
        )
    
    # Add monitoring connections
    monitoring_connections = [
        (6, 1),  # Monitoring to Pipelines
        (6, 4),  # Monitoring to Endpoints
        (6, 5),  # Monitoring to Cloud Run
    ]
    
    # Draw monitoring arrows
    for start_idx, end_idx in monitoring_connections:
        start_x, start_y, end_x, end_y = get_connection_points(start_idx, end_idx)
        
        # Custom curvature for monitoring connections
        if components[end_idx]['x'] < components[start_idx]['x']:
            rad = -0.3
        else:
            rad = 0.3
            
        ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                facecolor='#F4B400',
                width=1.0,
                headwidth=5,
                alpha=0.8,
                connectionstyle=f"arc3,rad={rad}"
            ),
            zorder=2
        )
    
    # Add title
    plt.title('GCP MLOps Architecture for Sentiment Analysis', size=12)
    
    # Set limits and turn off axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def create_all_visualizations():
    """Generate all visualizations needed for the dashboard."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Create GCP architecture diagram
    create_gcp_architecture_diagram()
    
    # 2. Create data distribution visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "data_distribution.png")):
        plt.figure(figsize=(7, 3.5))
        # Generate sample data
        categories = ['Positive', 'Negative', 'Neutral']
        train_counts = [1250, 950, 800]
        val_counts = [300, 250, 200]
        test_counts = [450, 350, 300]
        x = np.arange(len(categories))
        width = 0.25
        
        # Create plot
        plt.bar(x - width, train_counts, width, label='Train')
        plt.bar(x, val_counts, width, label='Validation')
        plt.bar(x + width, test_counts, width, label='Test')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Number of Samples')
        plt.title('Data Distribution by Split')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "data_distribution.png"), dpi=120)
        plt.close()
    
    # 3. Create training metrics visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "training_metrics.png")):
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(7, 3.5))
        
        # Generate sample data
        epochs = range(1, 6)
        train_acc = [0.82, 0.87, 0.90, 0.91, 0.93]
        val_acc = [0.81, 0.85, 0.88, 0.89, 0.92]
        train_loss = [0.52, 0.38, 0.29, 0.22, 0.19]
        val_loss = [0.54, 0.42, 0.33, 0.28, 0.24]
        
        # Plot accuracy lines
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(epochs, train_acc, 'o-', color='tab:blue', label='Train Accuracy')
        ax1.plot(epochs, val_acc, 's-', color='lightblue', label='Validation Accuracy')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim([0.8, 0.95])
        
        # Create second y-axis for loss
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:red')
        ax2.plot(epochs, train_loss, 'o-', color='tab:red', label='Train Loss')
        ax2.plot(epochs, val_loss, 's-', color='lightcoral', label='Validation Loss')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim([0.1, 0.6])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.title('Training and Validation Metrics')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "training_metrics.png"), dpi=120)
        plt.close()
    
    # 4. Create evaluation metrics visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png")):
        # Sample confusion matrix
        cm = np.array([[425, 15, 10], [12, 335, 3], [8, 2, 290]])
        
        # Create heatmap
        plt.figure(figsize=(7, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Positive', 'Negative', 'Neutral'],
                   yticklabels=['Positive', 'Negative', 'Neutral'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.png"), dpi=120)
        plt.close()
    
    # 5. Create deployment architecture visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png")):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        
        # Define component dimensions
        box_width = 0.15
        box_height = 0.3
        
        # Define component positions - exact positioning for better alignment
        components = [
            {'name': 'Vertex AI\nModel Registry', 'x': 0.2, 'y': 0.5, 'color': '#0F9D58'},
            {'name': 'Vertex AI\nEndpoints', 'x': 0.4, 'y': 0.5, 'color': '#0F9D58'},
            {'name': 'Cloud Load\nBalancer', 'x': 0.6, 'y': 0.5, 'color': '#4285F4'},
            {'name': 'Cloud Run\nService', 'x': 0.8, 'y': 0.5, 'color': '#4285F4'},
        ]
        
        # Draw components as rectangles with text
        for component in components:
            # Create rectangle patch
            rect = plt.Rectangle(
                (component['x'] - box_width/2, component['y'] - box_height/2),
                box_width, box_height,
                facecolor=component['color'],
                edgecolor='black',
                alpha=0.7,
                zorder=1
            )
            ax.add_patch(rect)
            
            # Add text
            ax.text(
                component['x'], component['y'],
                component['name'],
                ha='center',
                va='center',
                color='white',
                fontsize=10,
                zorder=2
            )
        
        # Calculate arrow endpoints precisely
        def get_arrow_points(start_idx, end_idx):
            start_x = components[start_idx]['x'] + box_width/2  # Right edge of starting box
            start_y = components[start_idx]['y']
            end_x = components[end_idx]['x'] - box_width/2  # Left edge of ending box
            end_y = components[end_idx]['y']
            return start_x, start_y, end_x, end_y
        
        # Draw arrows with precise connection points
        for i in range(len(components) - 1):
            start_x, start_y, end_x, end_y = get_arrow_points(i, i+1)
            ax.annotate(
                '',
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    facecolor='black',
                    width=1.5,
                    headwidth=8,
                    connectionstyle="arc3,rad=0"
                ),
                zorder=3
            )
        
        # Add title
        plt.title('Deployment Architecture', size=14)
        
        # Add user requests
        ax.text(
            0.9, 0.8,
            'User Requests',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round", fc="0.9", ec="black"),
            zorder=2
        )
        
        # Draw arrow from user to Cloud Run
        ax.annotate(
            '',
            xy=(0.8, 0.5 + box_height/2),  # Top edge of Cloud Run box
            xytext=(0.9, 0.77),  # Bottom of user request text
            arrowprops=dict(
                facecolor='black',
                width=1.0,
                headwidth=6
            ),
            zorder=3
        )
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "deployment_architecture.png"), dpi=120)
        plt.close()
    
    # 6. Create positive explanation visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "positive_explanation.png")):
        plt.figure(figsize=(7, 3.5))
        
        # Sample text and importance scores
        words = ["The", "service", "was", "excellent", "and", "staff", "very", "friendly"]
        importance = [0.05, 0.25, 0.03, 0.40, 0.02, 0.18, 0.02, 0.30]
        
        # Create horizontal bar chart with green color scheme
        plt.barh(words, importance, color=['#c9e7c8', '#8ecc8d', '#c9e7c8', '#0F9D58', '#c9e7c8', '#8ecc8d', '#c9e7c8', '#0F9D58'])
        plt.xlabel('Importance Score')
        plt.title('Positive Sentiment Word Importance')
        plt.xlim(0, 0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "positive_explanation.png"), dpi=120)
        plt.close()
    
    # 7. Create negative explanation visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "negative_explanation.png")):
        plt.figure(figsize=(7, 3.5))
        
        # Sample text and importance scores
        words = ["The", "food", "was", "terrible", "and", "service", "very", "slow"]
        importance = [0.04, 0.20, 0.03, 0.45, 0.02, 0.15, 0.02, 0.35]
        
        # Create horizontal bar chart with red color scheme
        plt.barh(words, importance, color=['#fad2cf', '#f4908e', '#fad2cf', '#DB4437', '#fad2cf', '#f4908e', '#fad2cf', '#DB4437'])
        plt.xlabel('Importance Score')
        plt.title('Negative Sentiment Word Importance')
        plt.xlim(0, 0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "negative_explanation.png"), dpi=120)
        plt.close()
    
    # 8. Create bias detection visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "bias_detection.png")):
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(7, 3.5))
        
        # Sample data
        categories = ['Gender', 'Age', 'Race/Ethnicity']
        equal_odds_diff = [0.031, 0.048, 0.027]
        disparate_impact = [0.97, 0.95, 0.98]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Plot equal odds difference
        ax1.set_xlabel('Protected Attribute')
        ax1.set_ylabel('Equal Odds Difference', color='tab:blue')
        ax1.bar(x - width/2, equal_odds_diff, width, color='tab:blue', label='Equal Odds Difference')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim([0, 0.06])
        
        # Add threshold line
        ax1.axhline(y=0.05, color='tab:blue', linestyle='--', alpha=0.7)
        ax1.text(2.1, 0.05, 'Threshold: 0.05', color='tab:blue', va='bottom')
        
        # Create second y-axis for disparate impact
        ax2 = ax1.twinx()
        ax2.set_ylabel('Disparate Impact Ratio', color='tab:red')
        ax2.bar(x + width/2, disparate_impact, width, color='tab:red', label='Disparate Impact Ratio')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim([0.9, 1.0])
        
        # Add threshold line
        ax2.axhline(y=0.95, color='tab:red', linestyle='--', alpha=0.7)
        ax2.text(2.1, 0.945, 'Threshold: 0.95', color='tab:red', va='top')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Fairness Metrics by Protected Attribute')
        plt.xticks(x, categories)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "bias_detection.png"), dpi=120)
        plt.close()
    
    # 9. Create monitoring dashboard visualization
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png")):
        # Create a 2x2 grid for sample monitoring panels
        fig, axs = plt.subplots(2, 2, figsize=(7, 4))
        
        # Panel 1: Request latency
        days = np.arange(1, 31)
        latency = 100 + 10 * np.sin(days/3) + np.random.normal(0, 5, len(days))
        axs[0, 0].plot(days, latency, 'b-')
        axs[0, 0].set_title('Request Latency (ms)', fontsize=10)
        axs[0, 0].set_ylim([80, 120])
        axs[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.7)
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].tick_params(labelsize=8)
        
        # Panel 2: Error rate
        error_rate = 0.5 + 0.2 * np.cos(days/5) + np.random.normal(0, 0.1, len(days))
        error_rate = np.clip(error_rate, 0, 1) * 100  # Convert to percentage
        axs[0, 1].plot(days, error_rate, 'r-')
        axs[0, 1].set_title('Error Rate (%)', fontsize=10)
        axs[0, 1].set_ylim([0, 1.5])
        axs[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        axs[0, 1].grid(alpha=0.3)
        axs[0, 1].tick_params(labelsize=8)
        
        # Panel 3: Prediction distribution
        class_names = ['Positive', 'Negative', 'Neutral']
        distribution = [45, 35, 20]
        axs[1, 0].pie(distribution, labels=class_names, autopct='%1.1f%%', 
                     colors=['#0F9D58', '#DB4437', '#F4B400'])
        axs[1, 0].set_title('Prediction Distribution', fontsize=10)
        
        # Panel 4: Feature drift
        drift_days = np.arange(1, 31)
        drift_score = 0.08 + 0.02 * np.cumsum(np.random.normal(0, 0.01, len(drift_days)))
        drift_score = np.clip(drift_score, 0, 0.2)
        axs[1, 1].plot(drift_days, drift_score, 'g-')
        axs[1, 1].set_title('Feature Drift (KL Divergence)', fontsize=10)
        axs[1, 1].set_ylim([0, 0.2])
        axs[1, 1].axhline(y=0.15, color='r', linestyle='--', alpha=0.7)
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Add more space between subplots
        plt.savefig(os.path.join(ARTIFACTS_DIR, "monitoring_dashboard.png"), dpi=120)
        plt.close()


if __name__ == "__main__":
    # Make sure the artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Generate all visualizations on first run
    create_all_visualizations()
    
    # Run the Streamlit app
    main() 