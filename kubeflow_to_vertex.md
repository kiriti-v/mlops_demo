# Migration Guide: Kubeflow Pipelines to Vertex AI

This document outlines the migration strategy from Kubeflow Pipelines to Google Cloud's Vertex AI Pipelines.

## Architecture Comparison

| Kubeflow Pipelines | Vertex AI Pipelines |
|-------------------|---------------------|
| Self-managed infrastructure | Fully managed by Google Cloud |
| Manual scaling | Auto-scaling infrastructure |
| Custom auth integration | Integrated with IAM |
| Open-source | Commercial offering with SLAs |

## Component Migration Map

### Pipeline Definition

**Kubeflow:**
```python
@dsl.pipeline(
    name="sentiment-analysis-pipeline",
    description="A pipeline that analyzes sentiment in text data"
)
def sentiment_analysis_pipeline(data_path: str):
    data_prep_op = data_prep_component(data_path=data_path)
    train_op = train_component(train_path=data_prep_op.outputs['train_path'])
    # ...
```

**Vertex AI:**
```python
@pipeline(
    name="sentiment-analysis-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def sentiment_analysis_pipeline(data_path: str):
    data_prep_op = data_prep_component(data_path=data_path)
    train_op = train_component(train_path=data_prep_op.outputs['train_path'])
    # ...
```

### Component Definitions

**Kubeflow:**
```python
@kfp.dsl.component
def prepare_data(data_path: str, test_size: float) -> dict:
    # Implementation
```

**Vertex AI:**
```python
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def prepare_data(data_path: str, test_size: float) -> dict:
    # Implementation
```

## Artifact Management

| Kubeflow | Vertex AI |
|---------|-----------|
| `InputPath`/`OutputPath` | Google Cloud Storage paths |
| Local storage | Cloud Storage |
| Custom metadata store | Vertex ML Metadata |

## Compilation Process

**Kubeflow:**
```python
kfp.compiler.Compiler().compile(
    pipeline_func=sentiment_analysis_pipeline,
    package_path="pipeline.yaml"
)
```

**Vertex AI:**
```python
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

compiler.Compiler().compile(
    pipeline_func=sentiment_analysis_pipeline,
    package_path="pipeline.json"
)

pipeline_job = pipeline_jobs.PipelineJob(
    display_name="sentiment-analysis",
    template_path="pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={"data_path": "gs://my-bucket/data.csv"}
)
pipeline_job.submit()
```

## Benefits of Migration

1. **Scalability**: Native integration with Google Cloud's infrastructure
2. **Reliability**: Enterprise-grade SLAs and managed components
3. **Security**: Integrated with IAM and Google Cloud security controls
4. **Ecosystem**: Seamless integration with other Google Cloud services
5. **Managed Infrastructure**: Reduced operational overhead

## Migration Checklist

- [ ] Update component definitions to use Vertex AI format
- [ ] Change artifact storage to Google Cloud Storage
- [ ] Update pipeline compiler imports
- [ ] Configure pipeline_root to use GCS bucket
- [ ] Update component container images (if needed)
- [ ] Set up Google Cloud service accounts and permissions
- [ ] Configure monitoring and logging

## Monitoring and Observability

| Kubeflow | Vertex AI |
|---------|-----------|
| Custom Prometheus/Grafana | Cloud Monitoring integration |
| Manual log aggregation | Cloud Logging |
| Custom alerting | Cloud Monitoring alerts |

Our project demonstrates this migration path with components designed to work in both environments. 