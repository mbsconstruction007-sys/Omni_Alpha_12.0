"""
Strategy Configuration - Configuration Management for Strategy Engine
Step 8: World's #1 Strategy Engine
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml

logger = logging.getLogger(__name__)

class StrategyPreset(Enum):
    """Strategy presets"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    QUANTUM = "quantum"
    ML_FOCUSED = "ml_focused"
    ALTERNATIVE_DATA = "alternative_data"

@dataclass
class StrategyConfig:
    """Strategy engine configuration"""
    
    # General settings
    max_strategies: int = 100
    max_active_strategies: int = 10
    execution_frequency: int = 60  # seconds
    signal_retention_hours: int = 24
    
    # Signal aggregation settings
    fusion_method: str = "weighted_ensemble"
    min_signal_confidence: float = 0.3
    max_signals_per_symbol: int = 10
    
    # ML settings
    ml_model_path: str = "models/"
    ml_retrain_frequency: int = 86400  # seconds
    ml_confidence_threshold: float = 0.7
    
    # Discovery settings
    discovery_enabled: bool = True
    max_discovery_strategies: int = 5
    discovery_frequency: int = 3600  # seconds
    
    # Evolution settings
    evolution_enabled: bool = True
    max_evolution_generations: int = 10
    evolution_population_size: int = 20
    
    # Backtesting settings
    backtest_enabled: bool = True
    default_backtest_period: int = 252  # days
    backtest_commission: float = 0.001
    
    # Risk management settings
    max_position_size: float = 0.1  # 10% of portfolio
    max_drawdown: float = 0.2  # 20%
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.1  # 10%
    
    # Performance monitoring settings
    performance_update_frequency: int = 300  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'drawdown': 0.15,
        'loss': 0.1,
        'volatility': 0.3
    })
    
    # Alternative data settings
    alternative_data_enabled: bool = True
    news_sources: List[str] = field(default_factory=lambda: ['reuters', 'bloomberg', 'cnbc'])
    social_media_sources: List[str] = field(default_factory=lambda: ['twitter', 'reddit', 'stocktwits'])
    economic_indicators: List[str] = field(default_factory=lambda: ['gdp', 'inflation', 'unemployment'])
    
    # Quantum computing settings
    quantum_enabled: bool = False
    quantum_backend: str = "simulator"
    quantum_shots: int = 1000
    
    # Database settings
    database_url: str = "sqlite:///strategy_engine.db"
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "strategy_engine.log"
    log_rotation: str = "daily"
    
    # API settings
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Monitoring settings
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60  # seconds
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)

def load_strategy_config(config_path: Optional[str] = None) -> StrategyConfig:
    """Load strategy configuration from file or environment variables"""
    try:
        config = StrategyConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
            
            # Update config with file values
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Override with environment variables
        config = _load_from_environment(config)
        
        logger.info("✅ Strategy configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error loading strategy configuration: {e}")
        # Return default config
        return StrategyConfig()

def _load_from_environment(config: StrategyConfig) -> StrategyConfig:
    """Load configuration from environment variables"""
    try:
        # General settings
        config.max_strategies = int(os.getenv("STRATEGY_MAX_STRATEGIES", config.max_strategies))
        config.max_active_strategies = int(os.getenv("STRATEGY_MAX_ACTIVE_STRATEGIES", config.max_active_strategies))
        config.execution_frequency = int(os.getenv("STRATEGY_EXECUTION_FREQUENCY", config.execution_frequency))
        config.signal_retention_hours = int(os.getenv("STRATEGY_SIGNAL_RETENTION_HOURS", config.signal_retention_hours))
        
        # Signal aggregation settings
        config.fusion_method = os.getenv("STRATEGY_FUSION_METHOD", config.fusion_method)
        config.min_signal_confidence = float(os.getenv("STRATEGY_MIN_SIGNAL_CONFIDENCE", config.min_signal_confidence))
        config.max_signals_per_symbol = int(os.getenv("STRATEGY_MAX_SIGNALS_PER_SYMBOL", config.max_signals_per_symbol))
        
        # ML settings
        config.ml_model_path = os.getenv("STRATEGY_ML_MODEL_PATH", config.ml_model_path)
        config.ml_retrain_frequency = int(os.getenv("STRATEGY_ML_RETRAIN_FREQUENCY", config.ml_retrain_frequency))
        config.ml_confidence_threshold = float(os.getenv("STRATEGY_ML_CONFIDENCE_THRESHOLD", config.ml_confidence_threshold))
        
        # Discovery settings
        config.discovery_enabled = os.getenv("STRATEGY_DISCOVERY_ENABLED", "true").lower() == "true"
        config.max_discovery_strategies = int(os.getenv("STRATEGY_MAX_DISCOVERY_STRATEGIES", config.max_discovery_strategies))
        config.discovery_frequency = int(os.getenv("STRATEGY_DISCOVERY_FREQUENCY", config.discovery_frequency))
        
        # Evolution settings
        config.evolution_enabled = os.getenv("STRATEGY_EVOLUTION_ENABLED", "true").lower() == "true"
        config.max_evolution_generations = int(os.getenv("STRATEGY_MAX_EVOLUTION_GENERATIONS", config.max_evolution_generations))
        config.evolution_population_size = int(os.getenv("STRATEGY_EVOLUTION_POPULATION_SIZE", config.evolution_population_size))
        
        # Backtesting settings
        config.backtest_enabled = os.getenv("STRATEGY_BACKTEST_ENABLED", "true").lower() == "true"
        config.default_backtest_period = int(os.getenv("STRATEGY_DEFAULT_BACKTEST_PERIOD", config.default_backtest_period))
        config.backtest_commission = float(os.getenv("STRATEGY_BACKTEST_COMMISSION", config.backtest_commission))
        
        # Risk management settings
        config.max_position_size = float(os.getenv("STRATEGY_MAX_POSITION_SIZE", config.max_position_size))
        config.max_drawdown = float(os.getenv("STRATEGY_MAX_DRAWDOWN", config.max_drawdown))
        config.stop_loss = float(os.getenv("STRATEGY_STOP_LOSS", config.stop_loss))
        config.take_profit = float(os.getenv("STRATEGY_TAKE_PROFIT", config.take_profit))
        
        # Performance monitoring settings
        config.performance_update_frequency = int(os.getenv("STRATEGY_PERFORMANCE_UPDATE_FREQUENCY", config.performance_update_frequency))
        
        # Alternative data settings
        config.alternative_data_enabled = os.getenv("STRATEGY_ALTERNATIVE_DATA_ENABLED", "true").lower() == "true"
        config.news_sources = os.getenv("STRATEGY_NEWS_SOURCES", ",".join(config.news_sources)).split(",")
        config.social_media_sources = os.getenv("STRATEGY_SOCIAL_MEDIA_SOURCES", ",".join(config.social_media_sources)).split(",")
        config.economic_indicators = os.getenv("STRATEGY_ECONOMIC_INDICATORS", ",".join(config.economic_indicators)).split(",")
        
        # Quantum computing settings
        config.quantum_enabled = os.getenv("STRATEGY_QUANTUM_ENABLED", "false").lower() == "true"
        config.quantum_backend = os.getenv("STRATEGY_QUANTUM_BACKEND", config.quantum_backend)
        config.quantum_shots = int(os.getenv("STRATEGY_QUANTUM_SHOTS", config.quantum_shots))
        
        # Database settings
        config.database_url = os.getenv("STRATEGY_DATABASE_URL", config.database_url)
        config.cache_enabled = os.getenv("STRATEGY_CACHE_ENABLED", "true").lower() == "true"
        config.cache_ttl = int(os.getenv("STRATEGY_CACHE_TTL", config.cache_ttl))
        
        # Logging settings
        config.log_level = os.getenv("STRATEGY_LOG_LEVEL", config.log_level)
        config.log_file = os.getenv("STRATEGY_LOG_FILE", config.log_file)
        config.log_rotation = os.getenv("STRATEGY_LOG_ROTATION", config.log_rotation)
        
        # API settings
        config.api_enabled = os.getenv("STRATEGY_API_ENABLED", "true").lower() == "true"
        config.api_host = os.getenv("STRATEGY_API_HOST", config.api_host)
        config.api_port = int(os.getenv("STRATEGY_API_PORT", config.api_port))
        config.api_workers = int(os.getenv("STRATEGY_API_WORKERS", config.api_workers))
        
        # Monitoring settings
        config.monitoring_enabled = os.getenv("STRATEGY_MONITORING_ENABLED", "true").lower() == "true"
        config.metrics_port = int(os.getenv("STRATEGY_METRICS_PORT", config.metrics_port))
        config.health_check_interval = int(os.getenv("STRATEGY_HEALTH_CHECK_INTERVAL", config.health_check_interval))
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error loading configuration from environment: {e}")
        return config

def apply_strategy_preset(config: StrategyConfig, preset: StrategyPreset) -> StrategyConfig:
    """Apply strategy preset to configuration"""
    try:
        if preset == StrategyPreset.CONSERVATIVE:
            config.max_position_size = 0.05  # 5%
            config.max_drawdown = 0.1  # 10%
            config.stop_loss = 0.03  # 3%
            config.take_profit = 0.06  # 6%
            config.min_signal_confidence = 0.7
            config.ml_confidence_threshold = 0.8
            config.fusion_method = "bayesian"
            
        elif preset == StrategyPreset.MODERATE:
            config.max_position_size = 0.1  # 10%
            config.max_drawdown = 0.15  # 15%
            config.stop_loss = 0.05  # 5%
            config.take_profit = 0.1  # 10%
            config.min_signal_confidence = 0.5
            config.ml_confidence_threshold = 0.7
            config.fusion_method = "weighted_ensemble"
            
        elif preset == StrategyPreset.AGGRESSIVE:
            config.max_position_size = 0.2  # 20%
            config.max_drawdown = 0.25  # 25%
            config.stop_loss = 0.08  # 8%
            config.take_profit = 0.15  # 15%
            config.min_signal_confidence = 0.3
            config.ml_confidence_threshold = 0.6
            config.fusion_method = "ml_fusion"
            
        elif preset == StrategyPreset.QUANTUM:
            config.quantum_enabled = True
            config.fusion_method = "quantum_superposition"
            config.min_signal_confidence = 0.4
            config.ml_confidence_threshold = 0.6
            config.max_position_size = 0.15  # 15%
            config.max_drawdown = 0.2  # 20%
            
        elif preset == StrategyPreset.ML_FOCUSED:
            config.ml_confidence_threshold = 0.8
            config.fusion_method = "ml_fusion"
            config.discovery_enabled = True
            config.evolution_enabled = True
            config.max_discovery_strategies = 10
            config.max_evolution_generations = 20
            
        elif preset == StrategyPreset.ALTERNATIVE_DATA:
            config.alternative_data_enabled = True
            config.fusion_method = "weighted_ensemble"
            config.min_signal_confidence = 0.4
            config.news_sources = ['reuters', 'bloomberg', 'cnbc', 'wsj', 'ft']
            config.social_media_sources = ['twitter', 'reddit', 'stocktwits', 'linkedin']
            config.economic_indicators = ['gdp', 'inflation', 'unemployment', 'interest_rates', 'consumer_confidence']
        
        logger.info(f"✅ Applied strategy preset: {preset.value}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error applying strategy preset: {e}")
        return config

def save_strategy_config(config: StrategyConfig, config_path: str):
    """Save strategy configuration to file"""
    try:
        config_dict = {
            'max_strategies': config.max_strategies,
            'max_active_strategies': config.max_active_strategies,
            'execution_frequency': config.execution_frequency,
            'signal_retention_hours': config.signal_retention_hours,
            'fusion_method': config.fusion_method,
            'min_signal_confidence': config.min_signal_confidence,
            'max_signals_per_symbol': config.max_signals_per_symbol,
            'ml_model_path': config.ml_model_path,
            'ml_retrain_frequency': config.ml_retrain_frequency,
            'ml_confidence_threshold': config.ml_confidence_threshold,
            'discovery_enabled': config.discovery_enabled,
            'max_discovery_strategies': config.max_discovery_strategies,
            'discovery_frequency': config.discovery_frequency,
            'evolution_enabled': config.evolution_enabled,
            'max_evolution_generations': config.max_evolution_generations,
            'evolution_population_size': config.evolution_population_size,
            'backtest_enabled': config.backtest_enabled,
            'default_backtest_period': config.default_backtest_period,
            'backtest_commission': config.backtest_commission,
            'max_position_size': config.max_position_size,
            'max_drawdown': config.max_drawdown,
            'stop_loss': config.stop_loss,
            'take_profit': config.take_profit,
            'performance_update_frequency': config.performance_update_frequency,
            'alert_thresholds': config.alert_thresholds,
            'alternative_data_enabled': config.alternative_data_enabled,
            'news_sources': config.news_sources,
            'social_media_sources': config.social_media_sources,
            'economic_indicators': config.economic_indicators,
            'quantum_enabled': config.quantum_enabled,
            'quantum_backend': config.quantum_backend,
            'quantum_shots': config.quantum_shots,
            'database_url': config.database_url,
            'cache_enabled': config.cache_enabled,
            'cache_ttl': config.cache_ttl,
            'log_level': config.log_level,
            'log_file': config.log_file,
            'log_rotation': config.log_rotation,
            'api_enabled': config.api_enabled,
            'api_host': config.api_host,
            'api_port': config.api_port,
            'api_workers': config.api_workers,
            'monitoring_enabled': config.monitoring_enabled,
            'metrics_port': config.metrics_port,
            'health_check_interval': config.health_check_interval,
            'custom_settings': config.custom_settings
        }
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
        
        logger.info(f"✅ Strategy configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving strategy configuration: {e}")
        raise

def validate_strategy_config(config: StrategyConfig) -> bool:
    """Validate strategy configuration"""
    try:
        # Check required fields
        if config.max_strategies <= 0:
            logger.error("max_strategies must be positive")
            return False
        
        if config.max_active_strategies <= 0:
            logger.error("max_active_strategies must be positive")
            return False
        
        if config.execution_frequency <= 0:
            logger.error("execution_frequency must be positive")
            return False
        
        if config.min_signal_confidence < 0 or config.min_signal_confidence > 1:
            logger.error("min_signal_confidence must be between 0 and 1")
            return False
        
        if config.max_position_size <= 0 or config.max_position_size > 1:
            logger.error("max_position_size must be between 0 and 1")
            return False
        
        if config.max_drawdown <= 0 or config.max_drawdown > 1:
            logger.error("max_drawdown must be between 0 and 1")
            return False
        
        if config.stop_loss <= 0 or config.stop_loss > 1:
            logger.error("stop_loss must be between 0 and 1")
            return False
        
        if config.take_profit <= 0 or config.take_profit > 1:
            logger.error("take_profit must be between 0 and 1")
            return False
        
        # Check fusion method
        valid_fusion_methods = ["weighted_ensemble", "ml_fusion", "bayesian", "fuzzy_logic", "quantum_superposition", "simple_average"]
        if config.fusion_method not in valid_fusion_methods:
            logger.error(f"Invalid fusion_method: {config.fusion_method}")
            return False
        
        logger.info("✅ Strategy configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating strategy configuration: {e}")
        return False
