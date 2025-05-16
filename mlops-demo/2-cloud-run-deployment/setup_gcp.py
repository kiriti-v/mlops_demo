from google.cloud import storage
from google.api_core import exceptions
import subprocess
import json
import os

def load_config():
    """Load GCP configuration"""
    try:
        with open('tidal-fusion-399118-9bf0691a3825.json') as f:
            config = json.load(f)
            return config['project_id']
    except FileNotFoundError:
        raise Exception("Please place your GCP service account key JSON file in this directory")

def enable_apis():
    """Enable required GCP APIs"""
    print("\n📡 Enabling required Google Cloud APIs...")
    required_apis = [
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
        "artifactregistry.googleapis.com",
        "storage.googleapis.com"
    ]
    
    for api in required_apis:
        try:
            subprocess.run(
                ["gcloud", "services", "enable", api],
                check=True,
                capture_output=True
            )
            print(f"✅ Enabled {api}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Error enabling {api}: {e.stderr.decode()}")
            raise

def create_storage_bucket(project_id):
    """Create a Cloud Storage bucket for artifacts"""
    print("\n🪣 Setting up Cloud Storage bucket...")
    bucket_name = f"{project_id}-mlops-artifacts"
    
    storage_client = storage.Client()
    try:
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")
        print(f"✅ Created bucket: {bucket.name}")
    except exceptions.Conflict:
        print(f"✅ Bucket {bucket_name} already exists")
    return bucket_name

def setup_artifact_registry(project_id):
    """Set up Artifact Registry for Docker images"""
    print("\n📦 Setting up Artifact Registry repository...")
    repo_name = "mlops-images"
    
    try:
        subprocess.run(
            [
                "gcloud", "artifacts", "repositories", "create", repo_name,
                "--repository-format=docker",
                "--location=us-central1",
                "--description=MLOps Docker images"
            ],
            check=True,
            capture_output=True
        )
        print(f"✅ Created Artifact Registry repository: {repo_name}")
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr.decode():
            print(f"✅ Repository {repo_name} already exists")
        else:
            print(f"⚠️  Error creating repository: {e.stderr.decode()}")
            raise

def main():
    print("🚀 Starting GCP infrastructure setup...")
    
    # Load project configuration
    project_id = load_config()
    print(f"📂 Loaded configuration for project: {project_id}")
    
    # Set project in gcloud
    subprocess.run(["gcloud", "config", "set", "project", project_id], check=True)
    
    # Enable required APIs
    enable_apis()
    
    # Create storage bucket
    bucket_name = create_storage_bucket(project_id)
    
    # Setup Artifact Registry
    setup_artifact_registry(project_id)
    
    print("\n✨ GCP infrastructure setup complete!")
    print("\nNext steps:")
    print("1. Run deploy_to_cloud_run.py to deploy your model")
    print(f"2. Your artifacts will be stored in: gs://{bucket_name}")
    print("3. Docker images will be stored in Artifact Registry")

if __name__ == "__main__":
    main() 