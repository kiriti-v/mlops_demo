import subprocess
import json
import time
import requests
from pathlib import Path

def load_config():
    """Load GCP configuration"""
    try:
        with open('tidal-fusion-399118-9bf0691a3825.json') as f:
            config = json.load(f)
            return config['project_id']
    except FileNotFoundError:
        raise Exception("Please place your GCP service account key JSON file in this directory")

def deploy_to_cloud_run(project_id):
    """Deploy the service to Cloud Run"""
    service_name = "sentiment-analysis-api"
    region = "us-central1"
    
    print("\n🚀 Deploying to Cloud Run...")
    print("This may take a few minutes...")
    
    try:
        # Deploy using gcloud
        result = subprocess.run(
            [
                "gcloud", "run", "deploy", service_name,
                "--source", ".",
                "--platform", "managed",
                "--region", region,
                "--allow-unauthenticated",
                "--memory", "2Gi",
                "--timeout", "300",
                "--quiet"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Get service URL using a separate command
        url_result = subprocess.run(
            [
                "gcloud", "run", "services", "describe", service_name,
                "--region", region,
                "--format=get(status.url)"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        service_url = url_result.stdout.strip()
        if not service_url:
            raise Exception("Failed to get service URL")
        
        return service_url
                
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Deployment failed: {e.stderr}")
        raise

def test_deployment(service_url):
    """Test the deployed service"""
    print("\n🔍 Testing deployment...")
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{service_url}/health")
        print(f"Health check status: {health_response.status_code}")
        print(f"Health check response: {health_response.json()}")
    except Exception as e:
        print(f"⚠️  Health check failed: {str(e)}")
        return False

    # Test prediction endpoint
    try:
        test_data = {"text": "This is a test deployment and it's working great!"}
        predict_response = requests.post(f"{service_url}/predict", json=test_data)
        print(f"\nPrediction test status: {predict_response.status_code}")
        print(f"Prediction response: {predict_response.json()}")
    except Exception as e:
        print(f"⚠️  Prediction test failed: {str(e)}")
        return False

    return True

def main():
    print("🚀 Starting deployment process...")
    
    # Load project configuration
    project_id = load_config()
    print(f"📂 Loaded configuration for project: {project_id}")
    
    # Deploy to Cloud Run
    service_url = deploy_to_cloud_run(project_id)
    print(f"\n✅ Deployment successful!")
    print(f"🌐 Service URL: {service_url}")
    
    # Wait for service to be ready
    print("\n⏳ Waiting for service to be fully ready...")
    time.sleep(30)
    
    # Test the deployment
    if test_deployment(service_url):
        print("\n✨ Deployment and testing completed successfully!")
        print(f"\nYour API is now available at: {service_url}")
        print("You can access the API documentation at: {service_url}/docs")
    else:
        print("\n⚠️  Deployment testing failed. Please check the Cloud Run logs for more details.")

if __name__ == "__main__":
    main() 