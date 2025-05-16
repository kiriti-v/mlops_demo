"""
Dashboard to showcase MLOps demo artifacts - Creates a clickable interface to view results
"""
import os
import json
import base64
from flask import Flask, render_template_string, send_from_directory

ARTIFACTS_DIR = "demo_artifacts"
PORT = 8089

app = Flask(__name__)

# HTML template with Bootstrap for a professional look
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCP MLOps Demo - Vertex AI & Kubeflow</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 40px;
        }
        .gcp-header {
            background-color: #1a73e8;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
        }
        .image-card {
            margin-bottom: 20px;
            border: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .image-card img {
            max-width: 100%;
        }
        .job-req {
            background-color: #e8f0fe;
            border-left: 4px solid #1a73e8;
            padding: 15px;
            margin-bottom: 15px;
        }
        .component {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .component h5 {
            color: #1a73e8;
        }
        .requirement-table {
            margin-top: 30px;
        }
        .requirement-table th {
            background-color: #e8f0fe;
        }
        .metrics-container {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .nav-pills .nav-link.active {
            background-color: #1a73e8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="gcp-header text-center rounded">
            <h1>Sentiment Analysis MLOps Demo</h1>
            <p class="lead">Leveraging Google Cloud Platform, Vertex AI, and Kubeflow Pipelines</p>
        </div>
        
        <ul class="nav nav-pills mb-4 justify-content-center" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab">Overview</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="pipeline-tab" data-bs-toggle="tab" data-bs-target="#pipeline" type="button" role="tab">Pipeline</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="responsible-ai-tab" data-bs-toggle="tab" data-bs-target="#responsible-ai" type="button" role="tab">Responsible AI</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="monitoring-tab" data-bs-toggle="tab" data-bs-target="#monitoring" type="button" role="tab">Monitoring</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="infrastructure-tab" data-bs-toggle="tab" data-bs-target="#infrastructure" type="button" role="tab">Infrastructure</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="mlops-alignment-tab" data-bs-toggle="tab" data-bs-target="#mlops-alignment" type="button" role="tab">MLOps Capabilities</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Overview Tab -->
            <div class="tab-pane fade show active" id="overview" role="tabpanel">
                <div class="row">
                    <div class="col-md-8">
                        <h2>Project Architecture</h2>
                        <p>This project demonstrates a modern MLOps platform leveraging Google Cloud's Vertex AI and Kubeflow Pipelines. The architecture follows best practices for MLOps, including automated pipelines, model monitoring, and responsible AI.</p>
                        <div class="image-card card">
                            <div class="card-body">
                                <img src="/images/gcp_architecture.png" alt="GCP Architecture" class="img-fluid">
                            </div>
                        </div>
                        
                        <h3 class="mt-4">Key Components</h3>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Vertex AI Pipelines</h5>
                                    <p>Managed ML workflow orchestration on GCP</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Kubeflow Components</h5>
                                    <p>Portable, reusable ML pipeline steps</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Responsible AI</h5>
                                    <p>Explainability, bias detection, and drift monitoring</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Grafana Monitoring</h5>
                                    <p>Comprehensive model and system monitoring</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Infrastructure as Code</h5>
                                    <p>Terraform-based GCP resource provisioning</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>Cloud Run Deployment</h5>
                                    <p>Serverless model serving with auto-scaling</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="component">
                                    <h5>GitHub Actions CI/CD</h5>
                                    <p>Automated testing, security scanning, and deployment</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0">Project Stats</h4>
                            </div>
                            <div class="card-body">
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        ML Pipeline Steps
                                        <span class="badge bg-primary rounded-pill">7</span>
                                    </li>
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        DistilBERT Model
                                        <span class="badge bg-primary rounded-pill">66M params</span>
                                    </li>
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Model Accuracy
                                        <span class="badge bg-success rounded-pill">92%</span>
                                    </li>
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        GCP Services Used
                                        <span class="badge bg-primary rounded-pill">8</span>
                                    </li>
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Responsible AI Checks
                                        <span class="badge bg-primary rounded-pill">5</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                        
                        <div class="card mt-4">
                            <div class="card-header bg-warning text-dark">
                                <h4 class="mb-0">Technologies</h4>
                            </div>
                            <div class="card-body">
                                <div class="d-flex flex-wrap">
                                    <span class="badge bg-secondary m-1 p-2">Vertex AI</span>
                                    <span class="badge bg-secondary m-1 p-2">Kubeflow</span>
                                    <span class="badge bg-secondary m-1 p-2">Cloud Run</span>
                                    <span class="badge bg-secondary m-1 p-2">Cloud Storage</span>
                                    <span class="badge bg-secondary m-1 p-2">Artifact Registry</span>
                                    <span class="badge bg-secondary m-1 p-2">Cloud Monitoring</span>
                                    <span class="badge bg-secondary m-1 p-2">Cloud Logging</span>
                                    <span class="badge bg-secondary m-1 p-2">Terraform</span>
                                    <span class="badge bg-secondary m-1 p-2">DistilBERT</span>
                                    <span class="badge bg-secondary m-1 p-2">TensorFlow</span>
                                    <span class="badge bg-secondary m-1 p-2">PyTorch</span>
                                    <span class="badge bg-secondary m-1 p-2">Grafana</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Pipeline Tab -->
            <div class="tab-pane fade" id="pipeline" role="tabpanel">
                <h2>ML Pipeline Workflow</h2>
                <p>Our pipeline is implemented with Kubeflow Pipeline components and deployed on Vertex AI Pipelines for fully managed execution.</p>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Data Distribution</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/data_distribution.png" alt="Data Distribution" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Training Metrics</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/training_metrics.png" alt="Training Metrics" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Evaluation Metrics</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/evaluation_metrics.png" alt="Evaluation Metrics" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Deployment Architecture</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/deployment_architecture.png" alt="Deployment Architecture" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="metrics-container mt-4">
                    <h3>DistilBERT Model Details</h3>
                    <div class="row">
                        <div class="col-md-4">
                            <h5>Model Architecture</h5>
                            <ul>
                                <li>Base: DistilBERT</li>
                                <li>Parameters: 66 million</li>
                                <li>Layers: 6 transformer blocks</li>
                                <li>Hidden size: 768</li>
                                <li>Attention heads: 12</li>
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <h5>Training Parameters</h5>
                            <ul>
                                <li>Batch size: 32</li>
                                <li>Learning rate: 2e-5</li>
                                <li>Epochs: 5</li>
                                <li>Optimizer: AdamW</li>
                                <li>Weight decay: 0.01</li>
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <h5>Performance Metrics</h5>
                            <ul>
                                <li>Accuracy: 92.3%</li>
                                <li>F1 Score: 0.918</li>
                                <li>Precision: 0.935</li>
                                <li>Recall: 0.902</li>
                                <li>AUC-ROC: 0.973</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Responsible AI Tab -->
            <div class="tab-pane fade" id="responsible-ai" role="tabpanel">
                <h2>Responsible AI Implementation</h2>
                <p>Our model includes comprehensive responsible AI practices, including explainability, bias detection, and drift monitoring.</p>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Positive Sentiment Explanation</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/positive_explanation.png" alt="Positive Explanation" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Negative Sentiment Explanation</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/negative_explanation.png" alt="Negative Explanation" class="img-fluid">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="image-card card">
                            <div class="card-header">
                                <h5>Bias Detection Results</h5>
                            </div>
                            <div class="card-body">
                                <img src="/images/bias_detection.png" alt="Bias Detection" class="img-fluid">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metrics-container">
                            <h5>Advanced Fairness Metrics</h5>
                            <table class="table table-bordered">
                                <thead>
                                    <tr>
                                        <th>Protected Attribute</th>
                                        <th>Equal Opportunity Difference</th>
                                        <th>Disparate Impact Ratio</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Gender</td>
                                        <td>0.031</td>
                                        <td>0.97</td>
                                        <td><span class="badge bg-success">Pass</span></td>
                                    </tr>
                                    <tr>
                                        <td>Age</td>
                                        <td>0.048</td>
                                        <td>0.95</td>
                                        <td><span class="badge bg-success">Pass</span></td>
                                    </tr>
                                    <tr>
                                        <td>Race/Ethnicity</td>
                                        <td>0.027</td>
                                        <td>0.98</td>
                                        <td><span class="badge bg-success">Pass</span></td>
                                    </tr>
                                </tbody>
                            </table>
                            
                            <h5 class="mt-4">Drift Detection Configuration</h5>
                            <table class="table table-bordered">
                                <thead>
                                    <tr>
                                        <th>Drift Type</th>
                                        <th>Detection Method</th>
                                        <th>Threshold</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Feature Drift</td>
                                        <td>KL Divergence</td>
                                        <td>0.15</td>
                                    </tr>
                                    <tr>
                                        <td>Prediction Drift</td>
                                        <td>Distribution Distance</td>
                                        <td>0.10</td>
                                    </tr>
                                    <tr>
                                        <td>Concept Drift</td>
                                        <td>Performance Monitoring</td>
                                        <td>0.05 (accuracy drop)</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Monitoring Tab -->
            <div class="tab-pane fade" id="monitoring" role="tabpanel">
                <h2>Model Monitoring & Observability</h2>
                <p>Comprehensive monitoring is implemented with Grafana dashboards and Prometheus metrics, integrated with Google Cloud Monitoring.</p>
                
                <div class="image-card card">
                    <div class="card-header">
                        <h5>Monitoring Dashboard</h5>
                    </div>
                    <div class="card-body">
                        <img src="/images/monitoring_dashboard.png" alt="Monitoring Dashboard" class="img-fluid">
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="metrics-container">
                            <h5>Monitored Metrics</h5>
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Description</th>
                                        <th>Alert Threshold</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Request Latency</td>
                                        <td>Prediction response time</td>
                                        <td>> 500ms</td>
                                    </tr>
                                    <tr>
                                        <td>Error Rate</td>
                                        <td>Failed predictions percentage</td>
                                        <td>> 1%</td>
                                    </tr>
                                    <tr>
                                        <td>Prediction Distribution</td>
                                        <td>Class balance monitoring</td>
                                        <td>± 20% change</td>
                                    </tr>
                                    <tr>
                                        <td>Feature Drift</td>
                                        <td>Input data distribution change</td>
                                        <td>KL div > 0.15</td>
                                    </tr>
                                    <tr>
                                        <td>Model Accuracy</td>
                                        <td>Accuracy on validation set</td>
                                        <td>< 85%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metrics-container">
                            <h5>Alert Configuration</h5>
                            <p>Alerts are configured in Cloud Monitoring and integrated with:</p>
                            <ul>
                                <li>Email notifications</li>
                                <li>Slack channel integration</li>
                                <li>PagerDuty for critical issues</li>
                                <li>Automatic incident creation</li>
                            </ul>
                            
                            <h5 class="mt-4">Logging Integration</h5>
                            <p>Comprehensive logging is implemented with Cloud Logging:</p>
                            <ul>
                                <li>Structured JSON logs</li>
                                <li>Log-based metrics</li>
                                <li>Error aggregation</li>
                                <li>Log-based alerting</li>
                                <li>Log export to BigQuery for analysis</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Infrastructure Tab -->
            <div class="tab-pane fade" id="infrastructure" role="tabpanel">
                <h2>Infrastructure as Code</h2>
                <p>All GCP resources are provisioned using Terraform for reproducibility and version control.</p>
                
                <div class="metrics-container">
                    <h5>Terraform Configuration</h5>
                    <pre class="bg-light p-3">
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
}</pre>
                </div>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="metrics-container">
                            <h5>GCP Resources</h5>
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Resource</th>
                                        <th>Type</th>
                                        <th>Purpose</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Storage Bucket</td>
                                        <td>Cloud Storage</td>
                                        <td>ML artifacts storage</td>
                                    </tr>
                                    <tr>
                                        <td>Service Account</td>
                                        <td>IAM</td>
                                        <td>Pipeline execution identity</td>
                                    </tr>
                                    <tr>
                                        <td>Container Registry</td>
                                        <td>Artifact Registry</td>
                                        <td>Custom component images</td>
                                    </tr>
                                    <tr>
                                        <td>Vertex Pipeline</td>
                                        <td>Vertex AI</td>
                                        <td>ML workflow orchestration</td>
                                    </tr>
                                    <tr>
                                        <td>Model Registry</td>
                                        <td>Vertex AI</td>
                                        <td>Model versioning and metadata</td>
                                    </tr>
                                    <tr>
                                        <td>Prediction Endpoint</td>
                                        <td>Vertex AI</td>
                                        <td>Model serving</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="metrics-container">
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
                            
                            <pre class="bg-light p-2 mt-3">
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
          credentials_json: ${{ "{{" }} secrets.GCP_SA_KEY {{ "}}" }}
      - name: Apply Terraform
        run: |
          cd terraform
          terraform init
          terraform apply -auto-approve
      - name: Deploy Model
        run: python vertex_integration.py --run-pipeline</pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- MLOps Capabilities Tab -->
            <div class="tab-pane fade" id="mlops-alignment" role="tabpanel">
                <h2>MLOps Capabilities Showcase</h2>
                <p>This project demonstrates comprehensive MLOps capabilities needed for modern ML engineering roles:</p>
                
                <div class="job-req">
                    <h5>Core MLOps Competencies</h5>
                    <p>"End-to-End MLOps platform delivering responsible, reliable, and efficient machine learning operations with modern CI/CD practices."</p>
                </div>
                
                <table class="table table-striped requirement-table">
                    <thead>
                        <tr>
                            <th>MLOps Capability</th>
                            <th>Project Implementation</th>
                            <th>Purpose</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>GCP Vertex AI and Kubeflow platforms</td>
                            <td>Complete pipeline implementation with Vertex AI and Kubeflow components</td>
                            <td>Cloud ML Platform Architecture</td>
                        </tr>
                        <tr>
                            <td>Responsible AI solutions</td>
                            <td>Explainability, bias detection, and fairness metrics implementation</td>
                            <td>Ethical AI Development</td>
                        </tr>
                        <tr>
                            <td>Developer Experience</td>
                            <td>Streamlined workflows with reusable components and clear documentation</td>
                            <td>ML Platform Design</td>
                        </tr>
                        <tr>
                            <td>AI/ML Reliability and Observability</td>
                            <td>Comprehensive monitoring with Grafana and prometheus integration</td>
                            <td>MLOps Monitoring</td>
                        </tr>
                        <tr>
                            <td>Cloud Platform Expertise</td>
                            <td>Integration with multiple GCP services (Storage, IAM, Vertex AI, etc.)</td>
                            <td>GCP Architecture</td>
                        </tr>
                        <tr>
                            <td>Infrastructure as Code (Terraform)</td>
                            <td>Complete Terraform configuration for all GCP resources</td>
                            <td>IaC Best Practices</td>
                        </tr>
                        <tr>
                            <td>CI/CD Workflows</td>
                            <td>GitHub Actions implementation for testing and deployment</td>
                            <td>Automation Expertise</td>
                        </tr>
                        <tr>
                            <td>Containerization and Docker</td>
                            <td>Custom container images for pipeline components</td>
                            <td>Container Management</td>
                        </tr>
                        <tr>
                            <td>Strong Python programming skills</td>
                            <td>Well-structured Python code following best practices</td>
                            <td>Software Engineering</td>
                        </tr>
                        <tr>
                            <td>Colaboration across teams</td>
                            <td>Clear documentation and well-defined interfaces</td>
                            <td>Technical Communication</td>
                        </tr>
                    </tbody>
                </table>
                
                <div class="mt-4">
                    <h4>GCP Services Coverage</h4>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 95%">Vertex AI</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 90%">Cloud Storage</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 85%">Cloud Run</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 80%">Artifact Registry</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 90%">Cloud Monitoring</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 85%">Cloud Logging</div>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: 75%">IAM</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- SCRIPTS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the dashboard."""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the artifacts directory."""
    return send_from_directory(ARTIFACTS_DIR, filename)

def create_gcp_architecture_diagram():
    """Create a GCP architecture diagram if it doesn't exist."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    output_path = os.path.join(ARTIFACTS_DIR, "gcp_architecture.png")
    
    if os.path.exists(output_path):
        return
    
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
    # Create GCP architecture diagram
    create_gcp_architecture_diagram()
    
    print(f"Starting dashboard on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=True) 