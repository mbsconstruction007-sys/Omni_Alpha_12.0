import asyncio
import time
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()

class RecoveryStrategy(Enum):
    RETRY = "retry"
    CIRCUIT_BREAK = "circuit_break"
    FALLBACK = "fallback"
    ESCALATE = "escalate"

@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0

class ErrorRecoverySystem:
    """
    Production-grade error recovery with circuit breakers
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay_ms = 100
        self.circuit_break_threshold = 5
        self.circuit_break_timeout = 60  # seconds
        self.half_open_success_threshold = 3
        
    async def execute_with_recovery(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic recovery
        """
        # Check circuit breaker
        if self._is_circuit_open(operation_name):
            return await self._handle_circuit_open(operation_name)
            
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Execute operation
                result = await operation(*args, **kwargs)
                
                # Mark success
                self._record_success(operation_name)
                
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                # Record failure
                self._record_failure(operation_name, e)
                
                # Log with context
                logger.warning(
                    "Operation failed, attempting recovery",
                    operation=operation_name,
                    attempt=retries,
                    error=str(e)
                )
                
                # Determine recovery strategy
                strategy = self._get_recovery_strategy(operation_name, e, retries)
                
                if strategy == RecoveryStrategy.RETRY:
                    await self._exponential_backoff(retries)
                    continue
                    
                elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                    self._open_circuit(operation_name)
                    return await self._handle_circuit_open(operation_name)
                    
                elif strategy == RecoveryStrategy.FALLBACK:
                    return await self._execute_fallback(operation_name, *args, **kwargs)
                    
                elif strategy == RecoveryStrategy.ESCALATE:
                    await self._escalate_error(operation_name, e)
                    raise
                    
        # Max retries exceeded
        logger.error(
            "Max retries exceeded",
            operation=operation_name,
            attempts=retries,
            last_error=str(last_error)
        )
        
        # Open circuit breaker
        self._open_circuit(operation_name)
        
        raise last_error
        
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open"""
        if operation_name not in self.circuit_breakers:
            return False
            
        breaker = self.circuit_breakers[operation_name]
        
        if breaker.state == "closed":
            return False
            
        if breaker.state == "open":
            # Check if timeout has passed
            if time.time() - breaker.last_failure_time > self.circuit_break_timeout:
                # Move to half-open state
                breaker.state = "half_open"
                breaker.success_count = 0
                logger.info(f"Circuit breaker half-open for {operation_name}")
                return False
            return True
            
        # Half-open state
        return False
        
    def _open_circuit(self, operation_name: str):
        """Open circuit breaker"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
            
        breaker = self.circuit_breakers[operation_name]
        breaker.state = "open"
        breaker.last_failure_time = time.time()
        
        logger.error(f"Circuit breaker opened for {operation_name}")
        
    def _record_success(self, operation_name: str):
        """Record successful operation"""
        if operation_name not in self.circuit_breakers:
            return
            
        breaker = self.circuit_breakers[operation_name]
        
        if breaker.state == "half_open":
            breaker.success_count += 1
            
            if breaker.success_count >= self.half_open_success_threshold:
                # Close circuit breaker
                breaker.state = "closed"
                breaker.failure_count = 0
                breaker.success_count = 0
                logger.info(f"Circuit breaker closed for {operation_name}")
                
    def _record_failure(self, operation_name: str, error: Exception):
        """Record failed operation"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreakerState()
            
        breaker = self.circuit_breakers[operation_name]
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        if breaker.state == "half_open":
            # Immediately re-open on failure in half-open state
            self._open_circuit(operation_name)
            
    async def _exponential_backoff(self, attempt: int):
        """Exponential backoff delay"""
        delay = self.retry_delay_ms * (2 ** (attempt - 1)) / 1000
        await asyncio.sleep(delay)
        
    def _get_recovery_strategy(
        self,
        operation_name: str,
        error: Exception,
        attempt: int
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on error and context"""
        
        # Check configured strategy
        if operation_name in self.recovery_strategies:
            return self.recovery_strategies[operation_name]
            
        # Default strategies based on error type
        if isinstance(error, asyncio.TimeoutError):
            return RecoveryStrategy.RETRY if attempt < 2 else RecoveryStrategy.CIRCUIT_BREAK
            
        if isinstance(error, ConnectionError):
            return RecoveryStrategy.RETRY if attempt < 3 else RecoveryStrategy.FALLBACK
            
        if isinstance(error, PermissionError):
            return RecoveryStrategy.ESCALATE
            
        # Check failure count for circuit breaking
        if operation_name in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_name]
            if breaker.failure_count >= self.circuit_break_threshold:
                return RecoveryStrategy.CIRCUIT_BREAK
                
        return RecoveryStrategy.RETRY
        
    async def _handle_circuit_open(self, operation_name: str):
        """Handle open circuit breaker"""
        if operation_name in self.fallback_handlers:
            return await self.fallback_handlers[operation_name]()
            
        raise Exception(f"Circuit breaker open for {operation_name}")
        
    async def _execute_fallback(self, operation_name: str, *args, **kwargs):
        """Execute fallback handler"""
        if operation_name in self.fallback_handlers:
            return await self.fallback_handlers[operation_name](*args, **kwargs)
            
        return None
        
    async def _escalate_error(self, operation_name: str, error: Exception):
        """Escalate error to monitoring/alerting system"""
        logger.critical(
            "Error escalated",
            operation=operation_name,
            error=str(error),
            error_type=type(error).__name__
        )
        
        # TODO: Send alert to PagerDuty/Slack/Email
        
    def register_fallback(self, operation_name: str, handler: Callable):
        """Register fallback handler for an operation"""
        self.fallback_handlers[operation_name] = handler
        
    def configure_strategy(self, operation_name: str, strategy: RecoveryStrategy):
        """Configure recovery strategy for an operation"""
        self.recovery_strategies[operation_name] = strategy

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
                'last_failure_time': breaker.last_failure_time,
                'is_open': self._is_circuit_open(name)
            }
        return status

    def reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker to closed state"""
        if operation_name in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_name]
            breaker.state = "closed"
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
            logger.info(f"Circuit breaker reset for {operation_name}")

# Global instance
error_recovery = ErrorRecoverySystem()
