# MLOps Sentiment Analysis with Kubeflow and Vertex AI

![Pipeline Overview](pipeline_visualization.png)

This project implements a complete MLOps pipeline for sentiment analysis using DistilBERT, with deployment paths for both Kubeflow Pipelines and Google Cloud Vertex AI.

## 🌟 [Live Demo](#live-demo) | [Architecture](#architecture) | [Setup](#setup-instructions) | [Kubeflow to Vertex AI](#kubeflow-to-vertex-ai)

## Architecture

This project demonstrates a modern ML platform architecture:

```
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                │     │                 │     │                 │     │                 │
│ Data Pipeline  │────▶│  Model Training │────▶│  Evaluation     │────▶│  Deployment     │
│                │     │                 │     │                 │     │                 │
└────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                      │                       │                       │
        ▼                      ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                               Kubeflow Pipeline Components                           │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │                      │                       │                       │
        ▼                      ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                             Vertex AI Pipeline Components                            │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points:
- **Kubeflow Pipelines**: Component organization, dependencies, and artifacts
- **Vertex AI**: Managed ML pipeline orchestration on Google Cloud
- **Cloud Run**: Model serving API deployment
- **Cloud Monitoring**: Performance tracking and alerting

## Live Demo

View the interactive pipeline demonstration:
- [Pipeline Visualization](pipeline_visualization.png)
- [Evaluation Metrics](evaluation_metrics.png)
- [Monitoring Dashboard](monitoring_dashboard.png)

## Kubeflow to Vertex AI

This project demonstrates the migration path from Kubeflow Pipelines to Vertex AI:

| Kubeflow Component | Vertex AI Equivalent |
|-------------------|----------------------|
| ContainerOp | PipelineComponent |
| pipeline.yaml | VertexPipeline |
| kfp.compiler | google.cloud.aiplatform |
| KFP UI | Vertex AI Pipelines UI |

Migration benefits:
- Managed infrastructure 
- Integrated with Google Cloud ecosystem
- Simplified security model
- Enterprise-grade SLAs

## Project Structure

```
mlops-demo/
├── component_impl.py         # Core component implementations
├── components.py             # Kubeflow component definitions
├── component_specs/          # Component specifications
├── pipeline_runner.py        # Pipeline execution utilities
├── integration_demo.py       # End-to-end workflow demo
├── explainer.py              # Model explanation utilities
└── tests/                    # Test suite
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
   pip install -r requirements.txt
   ```

4. Run the demo:
   ```bash
   python integration_demo.py
   ```

## Model Metrics

![Evaluation Metrics](evaluation_metrics.png)

The model achieves:
- Accuracy: 91.2%
- F1 Score: 0.89
- Precision: 0.87
- Recall: 0.92

## Monitoring

![Monitoring Dashboard](monitoring_dashboard.png)

## Responsible AI

The project includes:
- Data drift detection
- Bias analysis
- Model explanations

![Explanation Example](positive_explanation.png)

## License

MIT License 