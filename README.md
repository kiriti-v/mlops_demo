# MLOps Sentiment Analysis Project

This project implements a complete MLOps pipeline for sentiment analysis using DistilBERT, deployed on Google Cloud Platform.

## Project Structure

```
mlops-demo/
├── 1-model/               # Core model implementation
├── 2-cloud-run-deployment/# Cloud Run deployment configuration
├── 3-pipeline/           # ML pipeline components
├── 4-responsible-ai/     # Responsible AI implementations
├── 5-monitoring/         # Monitoring and observability
└── 6-cicd/              # CI/CD pipeline configuration
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mlops-demo
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r 1-model/requirements.txt
   pip install -r 2-cloud-run-deployment/requirements.txt
   pip install -r 6-cicd/requirements.txt
   ```

4. Download model:
   ```bash
   python 1-model/download_model.py
   ```

## Model Files

The model files are not included in the repository due to size constraints. They will be downloaded automatically when running the download script above.

## Development Workflow

1. Make changes to the code
2. Run tests locally:
   ```bash
   cd 6-cicd
   python test_model.py
   python test_api.py
   ```
3. Commit and push changes
4. The CI/CD pipeline will automatically:
   - Run all tests
   - Build the Docker container
   - Deploy to Cloud Run
   - Verify the deployment

## API Endpoints

The service is deployed at: https://sentiment-analysis-api-monitored-240846069363.us-central1.run.app

Available endpoints:
- GET /health - Health check
- POST /predict - Sentiment prediction

## Monitoring

Access the monitoring dashboard at:
[Cloud Monitoring Dashboard](https://console.cloud.google.com/monitoring)

## Responsible AI

The project includes:
- Data drift detection
- Bias analysis
- Model explanations

## Contributing

1. Create a new branch
2. Make your changes
3. Run tests locally
4. Create a pull request

## License

MIT License 