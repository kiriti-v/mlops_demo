from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboard_v1 import DashboardsServiceClient
from google.cloud.monitoring_dashboard_v1.types import Dashboard, Widget, XyChart, Text, TimeSeriesQuery
import json

class MonitoringDashboard:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = DashboardsServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def create_dashboard(self):
        """Create a monitoring dashboard for sentiment analysis metrics"""
        # Create a dashboard definition
        dashboard = Dashboard()
        dashboard.display_name = "Sentiment Analysis Monitoring"
        
        # Set up the grid layout
        dashboard.grid_layout.columns = 2
        
        # Create latency chart widget
        latency_widget = Widget()
        latency_widget.title = "Request Latency"
        
        latency_chart = XyChart()
        latency_query = TimeSeriesQuery()
        latency_query.time_series_filter.filter = 'metric.type="custom.googleapis.com/sentiment_analysis/latency"'
        latency_query.time_series_filter.aggregation.alignment_period.seconds = 60
        latency_query.time_series_filter.aggregation.per_series_aligner = 4  # ALIGN_MEAN
        
        latency_data_set = XyChart.DataSet()
        latency_data_set.time_series_query = latency_query
        latency_chart.data_sets.append(latency_data_set)
        
        latency_widget.xy_chart = latency_chart
        dashboard.grid_layout.widgets.append(latency_widget)
        
        # Create request count widget
        count_widget = Widget()
        count_widget.title = "Request Count by Sentiment"
        
        count_chart = XyChart()
        count_query = TimeSeriesQuery()
        count_query.time_series_filter.filter = 'metric.type="custom.googleapis.com/sentiment_analysis/request_count"'
        count_query.time_series_filter.aggregation.alignment_period.seconds = 60
        count_query.time_series_filter.aggregation.per_series_aligner = 1  # ALIGN_SUM
        
        count_data_set = XyChart.DataSet()
        count_data_set.time_series_query = count_query
        count_chart.data_sets.append(count_data_set)
        
        count_widget.xy_chart = count_chart
        dashboard.grid_layout.widgets.append(count_widget)
        
        # Create confidence widget
        confidence_widget = Widget()
        confidence_widget.title = "Prediction Confidence"
        
        confidence_chart = XyChart()
        confidence_query = TimeSeriesQuery()
        confidence_query.time_series_filter.filter = 'metric.type="custom.googleapis.com/sentiment_analysis/confidence"'
        confidence_query.time_series_filter.aggregation.alignment_period.seconds = 60
        confidence_query.time_series_filter.aggregation.per_series_aligner = 4  # ALIGN_MEAN
        
        confidence_data_set = XyChart.DataSet()
        confidence_data_set.time_series_query = confidence_query
        confidence_chart.data_sets.append(confidence_data_set)
        
        confidence_widget.xy_chart = confidence_chart
        dashboard.grid_layout.widgets.append(confidence_widget)
        
        try:
            # Create the dashboard
            response = self.client.create_dashboard(
                parent=self.project_name,
                dashboard=dashboard
            )
            
            dashboard_id = response.name.split('/')[-1]
            print(f"✅ Created dashboard: {response.name}")
            print(f"Dashboard URL: https://console.cloud.google.com/monitoring/dashboards/custom/{dashboard_id}?project={self.project_id}")
            
            return response.name
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            return None

def main():
    # Load project configuration
    try:
        with open('tidal-fusion-399118-9bf0691a3825.json') as f:
            config = json.load(f)
            project_id = config['project_id']
    except FileNotFoundError:
        raise Exception("Please place your GCP service account key JSON file in this directory")
    
    # Create and setup dashboard
    dashboard = MonitoringDashboard(project_id)
    dashboard.create_dashboard()

if __name__ == "__main__":
    main() 