from google.cloud import monitoring_v3
from google.api import label_pb2
import time
from typing import Optional

class CloudMonitoring:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
        
        # Initialize metric descriptors
        self._setup_metric_descriptors()
    
    def _setup_metric_descriptors(self):
        """Setup custom metric descriptors"""
        metrics = {
            "sentiment_analysis/latency": {
                "display_name": "Sentiment Analysis Latency",
                "description": "Processing time for sentiment analysis requests",
                "unit": "ms",
                "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
                "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            },
            "sentiment_analysis/request_count": {
                "display_name": "Request Count",
                "description": "Number of sentiment analysis requests",
                "unit": "1",
                "value_type": monitoring_v3.MetricDescriptor.ValueType.INT64,
                "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            },
            "sentiment_analysis/confidence": {
                "display_name": "Prediction Confidence",
                "description": "Confidence scores for sentiment predictions",
                "unit": "1",
                "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
                "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            }
        }
        
        for path, config in metrics.items():
            descriptor = monitoring_v3.MetricDescriptor(
                type=f"custom.googleapis.com/{path}",
                display_name=config["display_name"],
                description=config["description"],
                unit=config["unit"],
                value_type=config["value_type"],
                metric_kind=config["metric_kind"],
                labels=[
                    label_pb2.LabelDescriptor(
                        key="sentiment",
                        value_type=label_pb2.LabelDescriptor.ValueType.STRING,
                        description="Predicted sentiment (positive/negative)"
                    )
                ]
            )
            
            try:
                self.client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=descriptor
                )
                print(f"Created metric descriptor: {path}")
            except Exception as e:
                if "Already exists" not in str(e):
                    print(f"Error creating metric descriptor {path}: {e}")
    
    def write_latency_metric(self, latency_ms: float, sentiment: str):
        """Write latency metric to Cloud Monitoring"""
        self._write_metric(
            "sentiment_analysis/latency",
            latency_ms,
            {"sentiment": sentiment}
        )
    
    def write_request_metric(self, sentiment: str):
        """Write request count metric to Cloud Monitoring"""
        self._write_metric(
            "sentiment_analysis/request_count",
            1,
            {"sentiment": sentiment}
        )
    
    def write_confidence_metric(self, confidence: float, sentiment: str):
        """Write confidence metric to Cloud Monitoring"""
        self._write_metric(
            "sentiment_analysis/confidence",
            confidence,
            {"sentiment": sentiment}
        )
    
    def _write_metric(self, metric_path: str, value: float, labels: Optional[dict] = None):
        """Write a metric value to Cloud Monitoring"""
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_path}"
        
        if labels:
            series.metric.labels.update(labels)
        
        # Add resource information
        series.resource.type = "cloud_run_revision"
        series.resource.labels.update({
            "project_id": self.project_id,
            "service_name": "sentiment-analysis-api",
            "location": "us-central1",
        })
        
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        
        point = monitoring_v3.Point({
            "value": monitoring_v3.TypedValue(
                double_value=float(value) if isinstance(value, (int, float)) else value
            ),
            "interval": monitoring_v3.TimeInterval({
                "end_time": {"seconds": seconds, "nanos": nanos},
            }),
        })
        
        series.points = [point]
        
        try:
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
        except Exception as e:
            print(f"Error writing metric {metric_path}: {e}") 