"""
OMNI ALPHA 5.0 - DISTRIBUTED TRACING & OBSERVABILITY
====================================================
Production-ready distributed tracing with OpenTelemetry and Jaeger backend
"""

import os
import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FASTAPI_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    FASTAPI_INSTRUMENTATION_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'tracing')

class DistributedTracing:
    """Production distributed tracing with Jaeger backend"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.tracer = None
        self.is_initialized = False
        
        # Configuration
        self.service_name = os.getenv('SERVICE_NAME', 'omni-alpha-trading')
        self.service_version = os.getenv('SERVICE_VERSION', '5.0.0')
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.hostname = os.getenv('HOSTNAME', 'unknown')
        self.instance_id = os.getenv('INSTANCE_ID', f'{self.hostname}-{os.getpid()}')
        
        # Jaeger configuration
        self.jaeger_host = os.getenv('JAEGER_HOST', 'localhost')
        self.jaeger_port = int(os.getenv('JAEGER_PORT', '14268'))
        self.jaeger_endpoint = os.getenv('JAEGER_ENDPOINT', f'http://{self.jaeger_host}:{self.jaeger_port}/api/traces')
        
        # Tracing configuration
        self.trace_enabled = os.getenv('TRACING_ENABLED', 'true').lower() == 'true'
        self.trace_sample_rate = float(os.getenv('TRACE_SAMPLE_RATE', '1.0'))
        self.trace_max_queue_size = int(os.getenv('TRACE_MAX_QUEUE_SIZE', '2048'))
        self.trace_batch_size = int(os.getenv('TRACE_BATCH_SIZE', '512'))
        self.trace_export_interval = int(os.getenv('TRACE_EXPORT_INTERVAL', '5000'))
        
        # Console export for development
        self.console_export = os.getenv('TRACE_CONSOLE_EXPORT', 'false').lower() == 'true'
    
    def initialize(self) -> bool:
        """Initialize OpenTelemetry with Jaeger backend"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available - distributed tracing disabled")
            return False
        
        if not self.trace_enabled:
            logger.info("Distributed tracing disabled by configuration")
            return False
        
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version,
                "service.instance.id": self.instance_id,
                "deployment.environment": self.environment,
                "host.name": self.hostname,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": "1.20.0"
            })
            
            # Set up tracer provider
            provider = TracerProvider(
                resource=resource,
                sampler=trace.sampling.TraceIdRatioBasedSampler(self.trace_sample_rate)
            )
            trace.set_tracer_provider(provider)
            
            # Configure exporters
            exporters = []
            
            # Jaeger exporter
            try:
                jaeger_exporter = JaegerExporter(
                    endpoint=self.jaeger_endpoint,
                    max_tag_value_length=1024,
                    udp_split_oversized_batches=True
                )
                exporters.append(jaeger_exporter)
                logger.info(f"Jaeger exporter configured: {self.jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {e}")
            
            # Console exporter for development
            if self.console_export:
                console_exporter = ConsoleSpanExporter()
                exporters.append(console_exporter)
                logger.info("Console exporter enabled")
            
            # Add span processors
            for exporter in exporters:
                span_processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.trace_max_queue_size,
                    max_export_batch_size=self.trace_batch_size,
                    schedule_delay_millis=self.trace_export_interval
                )
                provider.add_span_processor(span_processor)
            
            # Set global text map propagator
            set_global_textmap(TraceContextTextMapPropagator())
            
            # Auto-instrument libraries
            self._setup_auto_instrumentation()
            
            # Get tracer
            self.tracer = trace.get_tracer(
                __name__,
                version=self.service_version,
                schema_url="https://opentelemetry.io/schemas/1.20.0"
            )
            
            self.is_initialized = True
            logger.info("Distributed tracing initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed tracing: {e}")
            return False
    
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        try:
            # Database instrumentation
            AsyncPGInstrumentor().instrument()
            logger.debug("AsyncPG instrumentation enabled")
            
            # Redis instrumentation
            RedisInstrumentor().instrument()
            logger.debug("Redis instrumentation enabled")
            
            # HTTP requests instrumentation
            RequestsInstrumentor().instrument()
            logger.debug("Requests instrumentation enabled")
            
            # Logging instrumentation
            LoggingInstrumentor().instrument(
                set_logging_format=True,
                log_correlation=True
            )
            logger.debug("Logging instrumentation enabled")
            
            # System metrics instrumentation
            SystemMetricsInstrumentor().instrument()
            logger.debug("System metrics instrumentation enabled")
            
            # FastAPI instrumentation (if available)
            if FASTAPI_INSTRUMENTATION_AVAILABLE:
                FastAPIInstrumentor().instrument()
                logger.debug("FastAPI instrumentation enabled")
                
        except Exception as e:
            logger.error(f"Error setting up auto-instrumentation: {e}")
    
    def trace_method(self, span_name: str = None, attributes: Dict[str, Any] = None):
        """Decorator for tracing methods with automatic error handling"""
        def decorator(func: Callable) -> Callable:
            if not self.is_initialized:
                return func
                
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__qualname__}"
                
                with self.tracer.start_as_current_span(
                    name,
                    kind=trace.SpanKind.INTERNAL
                ) as span:
                    # Set basic attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.qualname", func.__qualname__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    # Add method arguments (be careful with sensitive data)
                    if args and len(args) > 0:
                        # Skip 'self' parameter
                        start_idx = 1 if hasattr(args[0], func.__name__) else 0
                        for i, arg in enumerate(args[start_idx:], start_idx):
                            if not self._is_sensitive_arg(arg):
                                span.set_attribute(f"arg.{i}", str(arg)[:100])
                    
                    # Add keyword arguments (filter sensitive ones)
                    for key, value in kwargs.items():
                        if not self._is_sensitive_key(key) and not self._is_sensitive_arg(value):
                            span.set_attribute(f"kwarg.{key}", str(value)[:100])
                    
                    try:
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        
                        # Record success metrics
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        # Add result information (if not sensitive)
                        if result is not None and not self._is_sensitive_arg(result):
                            span.set_attribute("result.type", type(result).__name__)
                            if hasattr(result, '__len__'):
                                span.set_attribute("result.length", len(result))
                        
                        return result
                        
                    except Exception as e:
                        # Record error information
                        span.set_status(
                            trace.Status(
                                trace.StatusCode.ERROR, 
                                f"{type(e).__name__}: {str(e)[:200]}"
                            )
                        )
                        span.record_exception(e)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e)[:200])
                        raise
                        
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__qualname__}"
                
                with self.tracer.start_as_current_span(
                    name,
                    kind=trace.SpanKind.INTERNAL
                ) as span:
                    # Set basic attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.qualname", func.__qualname__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Record success metrics
                        duration = time.time() - start_time
                        span.set_attribute("duration_ms", duration * 1000)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error information
                        span.set_status(
                            trace.Status(
                                trace.StatusCode.ERROR, 
                                f"{type(e).__name__}: {str(e)[:200]}"
                            )
                        )
                        span.record_exception(e)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e)[:200])
                        raise
                        
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a parameter key contains sensitive information"""
        sensitive_keys = {
            'password', 'secret', 'token', 'key', 'auth', 'credential',
            'api_key', 'secret_key', 'private_key', 'access_token'
        }
        return any(sensitive in key.lower() for sensitive in sensitive_keys)
    
    def _is_sensitive_arg(self, arg: Any) -> bool:
        """Check if an argument value might contain sensitive information"""
        if not isinstance(arg, (str, bytes)):
            return False
        
        arg_str = str(arg).lower()
        return (
            len(arg_str) > 20 and any(char.isalnum() for char in arg_str) and
            ('secret' in arg_str or 'token' in arg_str or 'key' in arg_str)
        )
    
    @contextmanager
    def trace_span(self, name: str, attributes: Dict[str, Any] = None, kind: trace.SpanKind = trace.SpanKind.INTERNAL):
        """Context manager for manual span creation"""
        if not self.is_initialized:
            yield None
            return
        
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
            except Exception as e:
                span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR, 
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )
                )
                span.record_exception(e)
                raise
    
    def add_baggage(self, key: str, value: str):
        """Add baggage to current context"""
        if not self.is_initialized:
            return
        
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current context"""
        if not self.is_initialized:
            return None
        
        return baggage.get_baggage(key)
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        if not self.is_initialized:
            return None
        
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID"""
        if not self.is_initialized:
            return None
        
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None
    
    def inject_context(self, carrier: Dict[str, str]):
        """Inject tracing context into carrier (for HTTP headers, etc.)"""
        if not self.is_initialized:
            return
        
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        TraceContextTextMapPropagator().inject(carrier)
    
    def extract_context(self, carrier: Dict[str, str]):
        """Extract tracing context from carrier and set as current"""
        if not self.is_initialized:
            return
        
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        ctx = TraceContextTextMapPropagator().extract(carrier)
        context.attach(ctx)
    
    def shutdown(self):
        """Shutdown tracing and flush remaining spans"""
        if not self.is_initialized:
            return
        
        try:
            # Get the tracer provider and shutdown
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
            
            logger.info("Distributed tracing shutdown completed")
        except Exception as e:
            logger.error(f"Error during tracing shutdown: {e}")

# ===================== DECORATORS AND UTILITIES =====================

def trace_async(span_name: str = None, attributes: Dict[str, Any] = None):
    """Convenience decorator for async functions"""
    def decorator(func):
        tracer = get_distributed_tracing()
        return tracer.trace_method(span_name, attributes)(func)
    return decorator

def trace_sync(span_name: str = None, attributes: Dict[str, Any] = None):
    """Convenience decorator for sync functions"""
    def decorator(func):
        tracer = get_distributed_tracing()
        return tracer.trace_method(span_name, attributes)(func)
    return decorator

def trace_trading_operation(operation_type: str):
    """Specialized decorator for trading operations"""
    def decorator(func):
        attributes = {
            "trading.operation": operation_type,
            "trading.component": "engine"
        }
        tracer = get_distributed_tracing()
        return tracer.trace_method(attributes=attributes)(func)
    return decorator

def trace_data_operation(data_source: str, operation: str):
    """Specialized decorator for data operations"""
    def decorator(func):
        attributes = {
            "data.source": data_source,
            "data.operation": operation,
            "data.component": "collector"
        }
        tracer = get_distributed_tracing()
        return tracer.trace_method(attributes=attributes)(func)
    return decorator

# ===================== GLOBAL INSTANCE =====================

_distributed_tracing = None

def get_distributed_tracing() -> DistributedTracing:
    """Get global distributed tracing instance"""
    global _distributed_tracing
    if _distributed_tracing is None:
        _distributed_tracing = DistributedTracing()
    return _distributed_tracing

def initialize_tracing():
    """Initialize distributed tracing"""
    tracing = get_distributed_tracing()
    success = tracing.initialize()
    
    if success:
        # Register health check
        from infrastructure.monitoring import get_health_monitor
        health_monitor = get_health_monitor()
        
        async def tracing_health_check():
            if tracing.is_initialized:
                return {
                    'status': 'healthy',
                    'message': 'Distributed tracing operational',
                    'metrics': {
                        'service_name': tracing.service_name,
                        'jaeger_endpoint': tracing.jaeger_endpoint,
                        'sample_rate': tracing.trace_sample_rate
                    }
                }
            else:
                return {
                    'status': 'degraded',
                    'message': 'Distributed tracing not initialized',
                    'metrics': {}
                }
        
        health_monitor.register_health_check('distributed_tracing', tracing_health_check)
    
    return success
