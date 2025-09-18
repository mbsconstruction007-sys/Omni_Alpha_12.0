"""
OMNI ALPHA 5.0 - CIRCUIT BREAKER SYSTEM
=======================================
Production-ready circuit breaker implementation with state management and monitoring
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from config.settings import get_settings
from config.logging_config import get_logger

# ===================== CIRCUIT BREAKER STATES =====================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open" # Testing recovery

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorEvent:
    """Error event data"""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    message: str
    component: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    monitoring_window: int = 300
    max_half_open_requests: int = 5

# ===================== CIRCUIT BREAKER IMPLEMENTATION =====================

class CircuitBreaker:
    """Circuit breaker implementation with state management"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None, 
                 settings=None):
        if settings is None:
            settings = get_settings()
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=settings.monitoring.max_consecutive_errors,
                recovery_timeout=settings.monitoring.error_cooldown_seconds
            )
        
        self.name = name
        self.config = config
        self.settings = settings
        self.logger = get_logger(__name__, 'circuit_breaker')
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.state_transitions = deque(maxlen=100)
        
        # Threading
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_state_change: List[Callable] = []
        self.on_failure: List[Callable] = []
        self.on_success: List[Callable] = []
    
    def register_state_change_callback(self, callback: Callable):
        """Register callback for state changes"""
        self.on_state_change.append(callback)
    
    def register_failure_callback(self, callback: Callable):
        """Register callback for failures"""
        self.on_failure.append(callback)
    
    def register_success_callback(self, callback: Callable):
        """Register callback for successes"""
        self.on_success.append(callback)
    
    def record_success(self, context: Dict[str, Any] = None):
        """Record successful operation"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
            
            # Trigger success callbacks
            for callback in self.on_success:
                try:
                    callback(self, context or {})
                except Exception as e:
                    self.logger.error(f"Success callback error: {e}")
    
    def record_failure(self, error_type: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      message: str = "", component: str = "", context: Dict[str, Any] = None):
        """Record failed operation"""
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            message=message,
            component=component,
            context=context or {}
        )
        
        with self._lock:
            # Add to history
            self.error_history.append(error_event)
            self.error_counts[error_type] += 1
            
            # Update failure tracking
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                # Check if we should open the circuit
                if self._should_open_circuit(severity):
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
            
            # Trigger failure callbacks
            for callback in self.on_failure:
                try:
                    callback(self, error_event)
                except Exception as e:
                    self.logger.error(f"Failure callback error: {e}")
    
    def _should_open_circuit(self, severity: ErrorSeverity) -> bool:
        """Determine if circuit should open based on failures"""
        # Critical errors open circuit immediately
        if severity == ErrorSeverity.CRITICAL:
            return True
        
        # High severity errors have lower threshold
        if severity == ErrorSeverity.HIGH:
            return self.failure_count >= max(1, self.config.failure_threshold // 2)
        
        # Normal threshold for medium/low severity
        return self.failure_count >= self.config.failure_threshold
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now()
        self.half_open_requests = 0
        
        self._log_state_transition(old_state, self.state)
        self._trigger_state_change_callbacks(old_state, self.state)
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self.half_open_requests = 0
        
        self._log_state_transition(old_state, self.state)
        self._trigger_state_change_callbacks(old_state, self.state)
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        
        self._log_state_transition(old_state, self.state)
        self._trigger_state_change_callbacks(old_state, self.state)
    
    def _log_state_transition(self, old_state: CircuitState, new_state: CircuitState):
        """Log state transition"""
        transition = {
            'timestamp': datetime.now().isoformat(),
            'old_state': old_state.value,
            'new_state': new_state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        }
        
        self.state_transitions.append(transition)
        
        self.logger.warning(
            f"Circuit breaker {self.name}: {old_state.value} → {new_state.value}",
            extra={
                'circuit_breaker': self.name,
                'old_state': old_state.value,
                'new_state': new_state.value,
                'failure_count': self.failure_count
            }
        )
    
    def _trigger_state_change_callbacks(self, old_state: CircuitState, new_state: CircuitState):
        """Trigger state change callbacks"""
        for callback in self.on_state_change:
            try:
                callback(self, old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout)):
                    self._transition_to_half_open()
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self.half_open_requests < self.config.max_half_open_requests:
                    self.half_open_requests += 1
                    return True
                return False
            
            return False
    
    def execute_with_breaker(self, operation: Callable, *args, **kwargs):
        """Execute operation with circuit breaker protection"""
        if not self.can_execute():
            raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = operation(*args, **kwargs)
            self.record_success()
            return result
        
        except Exception as e:
            # Determine error severity
            severity = self._classify_error(e)
            
            self.record_failure(
                error_type=type(e).__name__,
                severity=severity,
                message=str(e),
                context={'args': str(args), 'kwargs': str(kwargs)}
            )
            raise
    
    async def execute_async_with_breaker(self, operation: Callable, *args, **kwargs):
        """Execute async operation with circuit breaker protection"""
        if not self.can_execute():
            raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await operation(*args, **kwargs)
            self.record_success()
            return result
        
        except Exception as e:
            # Determine error severity
            severity = self._classify_error(e)
            
            self.record_failure(
                error_type=type(e).__name__,
                severity=severity,
                message=str(e),
                context={'args': str(args), 'kwargs': str(kwargs)}
            )
            raise
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in [
            'connection refused', 'timeout', 'unauthorized', 'forbidden'
        ]):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in [
            'server error', 'internal error', 'database error'
        ]):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_message for keyword in [
            'not found', 'bad request', 'invalid'
        ]):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            old_state = self.state
            self._transition_to_closed()
            
            self.logger.info(
                f"Circuit breaker {self.name} manually reset",
                extra={'circuit_breaker': self.name, 'manual_reset': True}
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            recent_errors = [
                {
                    'timestamp': err.timestamp.isoformat(),
                    'type': err.error_type,
                    'severity': err.severity.value,
                    'message': err.message,
                    'component': err.component
                }
                for err in list(self.error_history)[-10:]  # Last 10 errors
            ]
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold
                },
                'recent_errors': recent_errors,
                'error_counts': dict(self.error_counts),
                'state_transitions': list(self.state_transitions)[-5:]  # Last 5 transitions
            }

# ===================== CIRCUIT BREAKER MANAGER =====================

class CircuitBreakerManager:
    """Manage multiple circuit breakers"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'circuit_breaker')
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create new circuit breaker"""
        with self._lock:
            if name in self.breakers:
                return self.breakers[name]
            
            breaker = CircuitBreaker(name, config, self.settings)
            self.breakers[name] = breaker
            
            self.logger.info(f"Created circuit breaker: {name}")
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker"""
        return self.breakers.get(name)
    
    def get_or_create_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        breaker = self.get_breaker(name)
        if breaker is None:
            breaker = self.create_breaker(name, config)
        return breaker
    
    def record_success(self, breaker_name: str, context: Dict[str, Any] = None):
        """Record success for specific breaker"""
        breaker = self.get_breaker(breaker_name)
        if breaker:
            breaker.record_success(context)
    
    def record_failure(self, breaker_name: str, error_type: str, 
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      message: str = "", component: str = "", 
                      context: Dict[str, Any] = None):
        """Record failure for specific breaker"""
        breaker = self.get_breaker(breaker_name)
        if breaker:
            breaker.record_failure(error_type, severity, message, component, context)
    
    def can_execute(self, breaker_name: str) -> bool:
        """Check if operation can be executed"""
        breaker = self.get_breaker(breaker_name)
        if breaker:
            return breaker.can_execute()
        return True  # If no breaker exists, allow execution
    
    def reset_breaker(self, breaker_name: str):
        """Reset specific circuit breaker"""
        breaker = self.get_breaker(breaker_name)
        if breaker:
            breaker.reset()
    
    def reset_all_breakers(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()
        
        self.logger.info("All circuit breakers reset")
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self.breakers.items()
            }
    
    def get_critical_breakers(self) -> List[str]:
        """Get list of circuit breakers in OPEN state"""
        with self._lock:
            return [
                name for name, breaker in self.breakers.items()
                if breaker.state == CircuitState.OPEN
            ]
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health based on circuit breakers"""
        with self._lock:
            if not self.breakers:
                return 1.0
            
            # Score based on circuit breaker states
            state_scores = {
                CircuitState.CLOSED: 1.0,
                CircuitState.HALF_OPEN: 0.5,
                CircuitState.OPEN: 0.0
            }
            
            total_score = sum(
                state_scores[breaker.state] 
                for breaker in self.breakers.values()
            )
            
            return total_score / len(self.breakers)

# ===================== DECORATORS =====================

def circuit_breaker(breaker_name: str, config: CircuitBreakerConfig = None):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            breaker = manager.get_or_create_breaker(breaker_name, config)
            
            return breaker.execute_with_breaker(func, *args, **kwargs)
        
        async def async_wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            breaker = manager.get_or_create_breaker(breaker_name, config)
            
            return await breaker.execute_async_with_breaker(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

# ===================== EXCEPTIONS =====================

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerException(Exception):
    """Base exception for circuit breaker errors"""
    pass

# ===================== GLOBAL MANAGER =====================

_circuit_breaker_manager = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager

def create_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Create circuit breaker"""
    return get_circuit_breaker_manager().create_breaker(name, config)

def record_success(breaker_name: str, context: Dict[str, Any] = None):
    """Record success for breaker"""
    get_circuit_breaker_manager().record_success(breaker_name, context)

def record_failure(breaker_name: str, error_type: str, 
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  message: str = "", component: str = "", 
                  context: Dict[str, Any] = None):
    """Record failure for breaker"""
    get_circuit_breaker_manager().record_failure(
        breaker_name, error_type, severity, message, component, context
    )

def can_execute(breaker_name: str) -> bool:
    """Check if operation can be executed"""
    return get_circuit_breaker_manager().can_execute(breaker_name)
