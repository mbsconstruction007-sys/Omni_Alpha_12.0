
"""
Dependency fallbacks for optional libraries
"""
import warnings

# Plotly fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    warnings.warn("Plotly not available, using fallback")
    PLOTLY_AVAILABLE = False
    
    class MockPlotly:
        def to_json(self):
            return '{"error": "Plotly not available"}'
    
    class go:
        @staticmethod
        def Figure(*args, **kwargs):
            return MockPlotly()
        
        @staticmethod
        def Scatter(*args, **kwargs):
            return MockPlotly()

# H2O fallback
try:
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    warnings.warn("H2O not available, using fallback")
    H2O_AVAILABLE = False
    
    class h2o:
        @staticmethod
        def init():
            pass

# ClickHouse fallback
try:
    import clickhouse_driver
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    warnings.warn("ClickHouse not available, using fallback")
    CLICKHOUSE_AVAILABLE = False
    
    class clickhouse_driver:
        class Client:
            def __init__(self, *args, **kwargs):
                pass
            
            def execute(self, *args, **kwargs):
                return []

# InfluxDB fallback
try:
    import influxdb_client
    INFLUXDB_AVAILABLE = True
except ImportError:
    warnings.warn("InfluxDB not available, using fallback")
    INFLUXDB_AVAILABLE = False
    
    class influxdb_client:
        class InfluxDBClient:
            def __init__(self, *args, **kwargs):
                pass
