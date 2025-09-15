"""
Circuit Breaker Module
Advanced circuit breaker system for risk management
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import json

logger = structlog.get_logger()

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if conditions improved

class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    VAR_BREACH = "var_breach"
    POSITION_SIZE = "position_size"
    PORTFOLIO_RISK = "portfolio_risk"
    BLACK_SWAN = "black_swan"
    CUSTOM = "custom"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    name: str
    breaker_type: CircuitBreakerType
    threshold: float
    time_window: int  # seconds
    cooldown_period: int  # seconds
    half_open_requests: int
    enabled: bool = True
    auto_reset: bool = True
    escalation_level: int = 1

@dataclass
class CircuitBreakerEvent:
    """Circuit breaker event"""
    timestamp: datetime
    breaker_name: str
    event_type: str  # "triggered", "reset", "half_open"
    threshold_value: float
    actual_value: float
    reason: str
    actions_taken: List[str]

class CircuitBreaker:
    """Advanced circuit breaker system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.breakers = {}
        self.events = []
        self.state_history = []
        self.initialize_circuit_breakers()
    
    def initialize_circuit_breakers(self):
        """Initialize all circuit breakers"""
        logger.info("Initializing circuit breaker system")
        
        # Daily loss circuit breaker
        self.breakers["daily_loss"] = CircuitBreakerConfig(
            name="Daily Loss Breaker",
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            threshold=self.config.get("DAILY_LOSS_CIRCUIT_BREAKER_PERCENT", 3.0),
            time_window=86400,  # 24 hours
            cooldown_period=3600,  # 1 hour
            half_open_requests=3,
            enabled=True,
            auto_reset=True,
            escalation_level=1
        )
        
        # Drawdown circuit breaker
        self.breakers["drawdown"] = CircuitBreakerConfig(
            name="Drawdown Breaker",
            breaker_type=CircuitBreakerType.DRAWDOWN,
            threshold=self.config.get("MAX_DRAWDOWN_PERCENT", 20.0),
            time_window=86400,
            cooldown_period=7200,  # 2 hours
            half_open_requests=5,
            enabled=True,
            auto_reset=True,
            escalation_level=2
        )
        
        # Volatility circuit breaker
        self.breakers["volatility"] = CircuitBreakerConfig(
            name="Volatility Breaker",
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=self.config.get("MAX_VOLATILITY_PERCENT", 50.0),
            time_window=3600,  # 1 hour
            cooldown_period=1800,  # 30 minutes
            half_open_requests=3,
            enabled=True,
            auto_reset=True,
            escalation_level=1
        )
        
        # VaR breach circuit breaker
        self.breakers["var_breach"] = CircuitBreakerConfig(
            name="VaR Breach Breaker",
            breaker_type=CircuitBreakerType.VAR_BREACH,
            threshold=self.config.get("MAX_VAR_PERCENT", 10.0),
            time_window=3600,
            cooldown_period=1800,
            half_open_requests=3,
            enabled=True,
            auto_reset=True,
            escalation_level=2
        )
        
        # Position size circuit breaker
        self.breakers["position_size"] = CircuitBreakerConfig(
            name="Position Size Breaker",
            breaker_type=CircuitBreakerType.POSITION_SIZE,
            threshold=self.config.get("MAX_POSITION_SIZE_PERCENT", 5.0),
            time_window=300,  # 5 minutes
            cooldown_period=600,  # 10 minutes
            half_open_requests=2,
            enabled=True,
            auto_reset=True,
            escalation_level=1
        )
        
        # Portfolio risk circuit breaker
        self.breakers["portfolio_risk"] = CircuitBreakerConfig(
            name="Portfolio Risk Breaker",
            breaker_type=CircuitBreakerType.PORTFOLIO_RISK,
            threshold=self.config.get("MAX_PORTFOLIO_RISK_PERCENT", 15.0),
            time_window=1800,  # 30 minutes
            cooldown_period=3600,
            half_open_requests=3,
            enabled=True,
            auto_reset=True,
            escalation_level=3
        )
        
        # Black swan circuit breaker
        self.breakers["black_swan"] = CircuitBreakerConfig(
            name="Black Swan Breaker",
            breaker_type=CircuitBreakerType.BLACK_SWAN,
            threshold=self.config.get("BLACK_SWAN_THRESHOLD", 0.8),
            time_window=300,
            cooldown_period=1800,
            half_open_requests=1,
            enabled=True,
            auto_reset=False,  # Manual reset required
            escalation_level=4
        )
        
        # Initialize breaker states
        for breaker_name in self.breakers:
            self.state_history.append({
                "breaker_name": breaker_name,
                "state": CircuitBreakerState.CLOSED,
                "timestamp": datetime.utcnow(),
                "trigger_count": 0,
                "last_trigger": None
            })
        
        logger.info("Circuit breaker system initialized", n_breakers=len(self.breakers))
    
    async def check_all_breakers(self) -> Dict[str, bool]:
        """Check all circuit breakers"""
        results = {}
        
        for breaker_name, breaker_config in self.breakers.items():
            if not breaker_config.enabled:
                results[breaker_name] = False
                continue
            
            is_triggered = await self._check_breaker(breaker_name, breaker_config)
            results[breaker_name] = is_triggered
            
            if is_triggered:
                await self._trigger_breaker(breaker_name, breaker_config)
        
        return results
    
    async def _check_breaker(self, breaker_name: str, breaker_config: CircuitBreakerConfig) -> bool:
        """Check if a specific circuit breaker should be triggered"""
        current_state = self._get_breaker_state(breaker_name)
        
        # Don't check if already open (unless in half-open state)
        if current_state == CircuitBreakerState.OPEN:
            return True
        
        # Get current value for the breaker
        current_value = await self._get_breaker_value(breaker_name, breaker_config)
        
        # Check if threshold is breached
        if current_value > breaker_config.threshold:
            logger.warning(
                "Circuit breaker threshold breached",
                breaker=breaker_name,
                current_value=current_value,
                threshold=breaker_config.threshold
            )
            return True
        
        return False
    
    async def _get_breaker_value(self, breaker_name: str, breaker_config: CircuitBreakerConfig) -> float:
        """Get current value for a circuit breaker"""
        if breaker_config.breaker_type == CircuitBreakerType.DAILY_LOSS:
            return await self._get_daily_loss()
        elif breaker_config.breaker_type == CircuitBreakerType.DRAWDOWN:
            return await self._get_current_drawdown()
        elif breaker_config.breaker_type == CircuitBreakerType.VOLATILITY:
            return await self._get_current_volatility()
        elif breaker_config.breaker_type == CircuitBreakerType.VAR_BREACH:
            return await self._get_current_var()
        elif breaker_config.breaker_type == CircuitBreakerType.POSITION_SIZE:
            return await self._get_max_position_size()
        elif breaker_config.breaker_type == CircuitBreakerType.PORTFOLIO_RISK:
            return await self._get_portfolio_risk()
        elif breaker_config.breaker_type == CircuitBreakerType.BLACK_SWAN:
            return await self._get_black_swan_threat()
        else:
            return 0.0
    
    async def _trigger_breaker(self, breaker_name: str, breaker_config: CircuitBreakerConfig):
        """Trigger a circuit breaker"""
        current_state = self._get_breaker_state(breaker_name)
        
        if current_state == CircuitBreakerState.OPEN:
            return  # Already open
        
        # Update state
        self._update_breaker_state(breaker_name, CircuitBreakerState.OPEN)
        
        # Record event
        event = CircuitBreakerEvent(
            timestamp=datetime.utcnow(),
            breaker_name=breaker_name,
            event_type="triggered",
            threshold_value=breaker_config.threshold,
            actual_value=await self._get_breaker_value(breaker_name, breaker_config),
            reason=f"Threshold breached: {breaker_config.threshold}",
            actions_taken=[]
        )
        
        # Take actions based on escalation level
        actions = await self._take_breaker_actions(breaker_name, breaker_config)
        event.actions_taken = actions
        
        self.events.append(event)
        
        logger.critical(
            "Circuit breaker triggered",
            breaker=breaker_name,
            escalation_level=breaker_config.escalation_level,
            actions=actions
        )
        
        # Schedule auto-reset if enabled
        if breaker_config.auto_reset:
            asyncio.create_task(self._schedule_auto_reset(breaker_name, breaker_config))
    
    async def _take_breaker_actions(self, breaker_name: str, breaker_config: CircuitBreakerConfig) -> List[str]:
        """Take actions when circuit breaker is triggered"""
        actions = []
        
        if breaker_config.escalation_level == 1:
            # Level 1: Basic actions
            actions.extend([
                "Stop new order placement",
                "Send alert to risk team",
                "Log incident"
            ])
        
        elif breaker_config.escalation_level == 2:
            # Level 2: Moderate actions
            actions.extend([
                "Stop new order placement",
                "Cancel pending orders",
                "Send alert to risk team",
                "Notify portfolio manager",
                "Log incident"
            ])
        
        elif breaker_config.escalation_level == 3:
            # Level 3: Severe actions
            actions.extend([
                "Stop all trading",
                "Cancel all pending orders",
                "Send emergency alert",
                "Notify senior management",
                "Activate emergency procedures",
                "Log incident"
            ])
        
        elif breaker_config.escalation_level == 4:
            # Level 4: Critical actions
            actions.extend([
                "Emergency stop all systems",
                "Cancel all orders immediately",
                "Send critical alert to all stakeholders",
                "Activate crisis management team",
                "Prepare for emergency liquidation",
                "Log critical incident"
            ])
        
        # Execute actions
        await self._execute_actions(actions)
        
        return actions
    
    async def _execute_actions(self, actions: List[str]):
        """Execute circuit breaker actions"""
        for action in actions:
            try:
                if action == "Stop new order placement":
                    await self._stop_new_orders()
                elif action == "Cancel pending orders":
                    await self._cancel_pending_orders()
                elif action == "Stop all trading":
                    await self._stop_all_trading()
                elif action == "Send alert to risk team":
                    await self._send_risk_alert()
                elif action == "Send emergency alert":
                    await self._send_emergency_alert()
                elif action == "Send critical alert to all stakeholders":
                    await self._send_critical_alert()
                elif action == "Activate emergency procedures":
                    await self._activate_emergency_procedures()
                elif action == "Activate crisis management team":
                    await self._activate_crisis_team()
                elif action == "Prepare for emergency liquidation":
                    await self._prepare_emergency_liquidation()
                
                logger.info("Circuit breaker action executed", action=action)
                
            except Exception as e:
                logger.error("Failed to execute circuit breaker action", action=action, error=str(e))
    
    async def reset_breaker(self, breaker_name: str, force: bool = False) -> bool:
        """Reset a circuit breaker"""
        if breaker_name not in self.breakers:
            logger.error("Unknown circuit breaker", breaker_name=breaker_name)
            return False
        
        breaker_config = self.breakers[breaker_name]
        current_state = self._get_breaker_state(breaker_name)
        
        if current_state == CircuitBreakerState.CLOSED:
            logger.info("Circuit breaker already closed", breaker_name=breaker_name)
            return True
        
        # Check if enough time has passed for cooldown
        if not force:
            last_trigger = self._get_last_trigger_time(breaker_name)
            if last_trigger:
                time_since_trigger = (datetime.utcnow() - last_trigger).total_seconds()
                if time_since_trigger < breaker_config.cooldown_period:
                    logger.warning(
                        "Circuit breaker cooldown period not elapsed",
                        breaker_name=breaker_name,
                        time_remaining=breaker_config.cooldown_period - time_since_trigger
                    )
                    return False
        
        # Reset breaker
        self._update_breaker_state(breaker_name, CircuitBreakerState.CLOSED)
        
        # Record event
        event = CircuitBreakerEvent(
            timestamp=datetime.utcnow(),
            breaker_name=breaker_name,
            event_type="reset",
            threshold_value=breaker_config.threshold,
            actual_value=await self._get_breaker_value(breaker_name, breaker_config),
            reason="Manual reset" if force else "Auto reset after cooldown",
            actions_taken=["Reset circuit breaker"]
        )
        
        self.events.append(event)
        
        logger.info("Circuit breaker reset", breaker_name=breaker_name, force=force)
        return True
    
    async def _schedule_auto_reset(self, breaker_name: str, breaker_config: CircuitBreakerConfig):
        """Schedule automatic reset of circuit breaker"""
        await asyncio.sleep(breaker_config.cooldown_period)
        
        # Check if still open
        current_state = self._get_breaker_state(breaker_name)
        if current_state == CircuitBreakerState.OPEN:
            await self.reset_breaker(breaker_name, force=False)
    
    def _get_breaker_state(self, breaker_name: str) -> CircuitBreakerState:
        """Get current state of a circuit breaker"""
        for state_info in self.state_history:
            if state_info["breaker_name"] == breaker_name:
                return state_info["state"]
        return CircuitBreakerState.CLOSED
    
    def _update_breaker_state(self, breaker_name: str, new_state: CircuitBreakerState):
        """Update state of a circuit breaker"""
        for state_info in self.state_history:
            if state_info["breaker_name"] == breaker_name:
                state_info["state"] = new_state
                state_info["timestamp"] = datetime.utcnow()
                
                if new_state == CircuitBreakerState.OPEN:
                    state_info["trigger_count"] += 1
                    state_info["last_trigger"] = datetime.utcnow()
                break
    
    def _get_last_trigger_time(self, breaker_name: str) -> Optional[datetime]:
        """Get last trigger time for a circuit breaker"""
        for state_info in self.state_history:
            if state_info["breaker_name"] == breaker_name:
                return state_info["last_trigger"]
        return None
    
    async def get_breaker_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers"""
        status = {}
        
        for breaker_name, breaker_config in self.breakers.items():
            current_state = self._get_breaker_state(breaker_name)
            current_value = await self._get_breaker_value(breaker_name, breaker_config)
            
            status[breaker_name] = {
                "name": breaker_config.name,
                "type": breaker_config.breaker_type.value,
                "state": current_state.value,
                "enabled": breaker_config.enabled,
                "threshold": breaker_config.threshold,
                "current_value": current_value,
                "threshold_breached": current_value > breaker_config.threshold,
                "escalation_level": breaker_config.escalation_level,
                "trigger_count": self._get_trigger_count(breaker_name),
                "last_trigger": self._get_last_trigger_time(breaker_name).isoformat() if self._get_last_trigger_time(breaker_name) else None
            }
        
        return status
    
    def _get_trigger_count(self, breaker_name: str) -> int:
        """Get trigger count for a circuit breaker"""
        for state_info in self.state_history:
            if state_info["breaker_name"] == breaker_name:
                return state_info["trigger_count"]
        return 0
    
    async def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Get recent circuit breaker events"""
        recent_events = self.events[-limit:] if self.events else []
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "breaker_name": event.breaker_name,
                "event_type": event.event_type,
                "threshold_value": event.threshold_value,
                "actual_value": event.actual_value,
                "reason": event.reason,
                "actions_taken": event.actions_taken
            }
            for event in recent_events
        ]
    
    async def create_custom_breaker(
        self,
        name: str,
        breaker_type: CircuitBreakerType,
        threshold: float,
        time_window: int = 3600,
        cooldown_period: int = 1800,
        escalation_level: int = 1
    ) -> bool:
        """Create a custom circuit breaker"""
        try:
            breaker_config = CircuitBreakerConfig(
                name=name,
                breaker_type=breaker_type,
                threshold=threshold,
                time_window=time_window,
                cooldown_period=cooldown_period,
                half_open_requests=3,
                enabled=True,
                auto_reset=True,
                escalation_level=escalation_level
            )
            
            self.breakers[name.lower().replace(" ", "_")] = breaker_config
            
            # Initialize state
            self.state_history.append({
                "breaker_name": name.lower().replace(" ", "_"),
                "state": CircuitBreakerState.CLOSED,
                "timestamp": datetime.utcnow(),
                "trigger_count": 0,
                "last_trigger": None
            })
            
            logger.info("Custom circuit breaker created", name=name, threshold=threshold)
            return True
            
        except Exception as e:
            logger.error("Failed to create custom circuit breaker", error=str(e))
            return False
    
    # Data access methods (placeholders)
    
    async def _get_daily_loss(self) -> float:
        """Get daily loss percentage"""
        return 2.5  # Placeholder
    
    async def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        return 15.0  # Placeholder
    
    async def _get_current_volatility(self) -> float:
        """Get current volatility percentage"""
        return 25.0  # Placeholder
    
    async def _get_current_var(self) -> float:
        """Get current VaR percentage"""
        return 8.0  # Placeholder
    
    async def _get_max_position_size(self) -> float:
        """Get maximum position size percentage"""
        return 3.0  # Placeholder
    
    async def _get_portfolio_risk(self) -> float:
        """Get portfolio risk percentage"""
        return 12.0  # Placeholder
    
    async def _get_black_swan_threat(self) -> float:
        """Get black swan threat level"""
        return 0.3  # Placeholder
    
    # Action execution methods (placeholders)
    
    async def _stop_new_orders(self):
        """Stop new order placement"""
        logger.info("Stopping new order placement")
    
    async def _cancel_pending_orders(self):
        """Cancel pending orders"""
        logger.info("Cancelling pending orders")
    
    async def _stop_all_trading(self):
        """Stop all trading"""
        logger.info("Stopping all trading")
    
    async def _send_risk_alert(self):
        """Send risk alert"""
        logger.info("Sending risk alert")
    
    async def _send_emergency_alert(self):
        """Send emergency alert"""
        logger.info("Sending emergency alert")
    
    async def _send_critical_alert(self):
        """Send critical alert"""
        logger.info("Sending critical alert")
    
    async def _activate_emergency_procedures(self):
        """Activate emergency procedures"""
        logger.info("Activating emergency procedures")
    
    async def _activate_crisis_team(self):
        """Activate crisis management team"""
        logger.info("Activating crisis management team")
    
    async def _prepare_emergency_liquidation(self):
        """Prepare for emergency liquidation"""
        logger.info("Preparing for emergency liquidation")
