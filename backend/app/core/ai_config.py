"""
AI Configuration - Configuration for the AI Brain and Execution System
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConsciousnessLevel(Enum):
    """AI Consciousness Levels"""
    BASIC = "basic"
    AWARE = "aware"
    INTELLIGENT = "intelligent"
    ADVANCED = "advanced"
    SUPREME = "supreme"
    TRANSCENDENT = "transcendent"
    OMNISCIENT = "omniscient"

@dataclass
class AIConfig:
    """AI Brain and Execution Configuration"""
    
    # Consciousness Settings
    consciousness_level: str = "advanced"
    self_evolution_enabled: bool = True
    quantum_brain_enabled: bool = False
    
    # Brain Settings
    max_thoughts: int = 1000000
    max_memories: int = 100000
    learning_rate: float = 0.01
    adaptation_rate: float = 0.1
    
    # Execution Settings
    max_execution_workers: int = 16
    execution_timeout: float = 30.0
    max_order_size: int = 1000000
    
    # Performance Settings
    target_latency_ms: float = 1.0
    target_slippage_bps: float = 2.0
    target_fill_rate: float = 0.99
    
    # Risk Settings
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    
    # Quantum Settings
    quantum_advantage_threshold: float = 0.7
    entanglement_strength: float = 0.8
    superposition_coherence: float = 0.9
    
    # Learning Settings
    pattern_recognition_enabled: bool = True
    strategy_optimization_enabled: bool = True
    memory_consolidation_enabled: bool = True
    
    # Monitoring Settings
    consciousness_monitoring: bool = True
    execution_monitoring: bool = True
    performance_tracking: bool = True
    
    # Advanced Settings
    meta_cognition_enabled: bool = True
    dream_cycles_enabled: bool = True
    adversarial_detection_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "CONSCIOUSNESS_LEVEL": self.consciousness_level,
            "SELF_EVOLUTION_ENABLED": self.self_evolution_enabled,
            "QUANTUM_BRAIN_ENABLED": self.quantum_brain_enabled,
            "MAX_THOUGHTS": self.max_thoughts,
            "MAX_MEMORIES": self.max_memories,
            "LEARNING_RATE": self.learning_rate,
            "ADAPTATION_RATE": self.adaptation_rate,
            "MAX_EXECUTION_WORKERS": self.max_execution_workers,
            "EXECUTION_TIMEOUT": self.execution_timeout,
            "MAX_ORDER_SIZE": self.max_order_size,
            "TARGET_LATENCY_MS": self.target_latency_ms,
            "TARGET_SLIPPAGE_BPS": self.target_slippage_bps,
            "TARGET_FILL_RATE": self.target_fill_rate,
            "MAX_POSITION_SIZE": self.max_position_size,
            "MAX_DAILY_LOSS": self.max_daily_loss,
            "MAX_DRAWDOWN": self.max_drawdown,
            "QUANTUM_ADVANTAGE_THRESHOLD": self.quantum_advantage_threshold,
            "ENTANGLEMENT_STRENGTH": self.entanglement_strength,
            "SUPERPOSITION_COHERENCE": self.superposition_coherence,
            "PATTERN_RECOGNITION_ENABLED": self.pattern_recognition_enabled,
            "STRATEGY_OPTIMIZATION_ENABLED": self.strategy_optimization_enabled,
            "MEMORY_CONSOLIDATION_ENABLED": self.memory_consolidation_enabled,
            "CONSCIOUSNESS_MONITORING": self.consciousness_monitoring,
            "EXECUTION_MONITORING": self.execution_monitoring,
            "PERFORMANCE_TRACKING": self.performance_tracking,
            "META_COGNITION_ENABLED": self.meta_cognition_enabled,
            "DREAM_CYCLES_ENABLED": self.dream_cycles_enabled,
            "ADVERSARIAL_DETECTION_ENABLED": self.adversarial_detection_enabled
        }

def load_ai_config() -> AIConfig:
    """Load AI configuration from environment variables"""
    try:
        config = AIConfig()
        
        # Override with environment variables
        config.consciousness_level = os.getenv("AI_CONSCIOUSNESS_LEVEL", config.consciousness_level)
        config.self_evolution_enabled = os.getenv("SELF_EVOLUTION_ENABLED", "true").lower() == "true"
        config.quantum_brain_enabled = os.getenv("QUANTUM_BRAIN_ENABLED", "false").lower() == "true"
        
        config.max_thoughts = int(os.getenv("MAX_THOUGHTS", config.max_thoughts))
        config.max_memories = int(os.getenv("MAX_MEMORIES", config.max_memories))
        config.learning_rate = float(os.getenv("LEARNING_RATE", config.learning_rate))
        config.adaptation_rate = float(os.getenv("ADAPTATION_RATE", config.adaptation_rate))
        
        config.max_execution_workers = int(os.getenv("MAX_EXECUTION_WORKERS", config.max_execution_workers))
        config.execution_timeout = float(os.getenv("EXECUTION_TIMEOUT", config.execution_timeout))
        config.max_order_size = int(os.getenv("MAX_ORDER_SIZE", config.max_order_size))
        
        config.target_latency_ms = float(os.getenv("TARGET_LATENCY_MS", config.target_latency_ms))
        config.target_slippage_bps = float(os.getenv("TARGET_SLIPPAGE_BPS", config.target_slippage_bps))
        config.target_fill_rate = float(os.getenv("TARGET_FILL_RATE", config.target_fill_rate))
        
        config.max_position_size = float(os.getenv("MAX_POSITION_SIZE", config.max_position_size))
        config.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", config.max_daily_loss))
        config.max_drawdown = float(os.getenv("MAX_DRAWDOWN", config.max_drawdown))
        
        config.quantum_advantage_threshold = float(os.getenv("QUANTUM_ADVANTAGE_THRESHOLD", config.quantum_advantage_threshold))
        config.entanglement_strength = float(os.getenv("ENTANGLEMENT_STRENGTH", config.entanglement_strength))
        config.superposition_coherence = float(os.getenv("SUPERPOSITION_COHERENCE", config.superposition_coherence))
        
        config.pattern_recognition_enabled = os.getenv("PATTERN_RECOGNITION_ENABLED", "true").lower() == "true"
        config.strategy_optimization_enabled = os.getenv("STRATEGY_OPTIMIZATION_ENABLED", "true").lower() == "true"
        config.memory_consolidation_enabled = os.getenv("MEMORY_CONSOLIDATION_ENABLED", "true").lower() == "true"
        
        config.consciousness_monitoring = os.getenv("CONSCIOUSNESS_MONITORING", "true").lower() == "true"
        config.execution_monitoring = os.getenv("EXECUTION_MONITORING", "true").lower() == "true"
        config.performance_tracking = os.getenv("PERFORMANCE_TRACKING", "true").lower() == "true"
        
        config.meta_cognition_enabled = os.getenv("META_COGNITION_ENABLED", "true").lower() == "true"
        config.dream_cycles_enabled = os.getenv("DREAM_CYCLES_ENABLED", "true").lower() == "true"
        config.adversarial_detection_enabled = os.getenv("ADVERSARIAL_DETECTION_ENABLED", "true").lower() == "true"
        
        logger.info("✅ AI configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load AI configuration: {str(e)}")
        # Return default configuration
        return AIConfig()

def get_ai_config_dict() -> Dict[str, Any]:
    """Get AI configuration as dictionary"""
    config = load_ai_config()
    return config.to_dict()

