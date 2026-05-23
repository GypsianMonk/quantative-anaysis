"""
System Configuration Management
===============================

Centralized configuration for all AEGIS PRIME components.
Uses Pydantic for validation and environment variable injection.
"""

from pydantic import BaseSettings, Field, validator
from typing import List, Dict, Optional
import os


class DatabaseConfig(BaseSettings):
    """PostgreSQL/TimescaleDB configuration."""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="aegis_prime", env="DB_NAME")
    user: str = Field(default="aegis_user", env="DB_USER")
    password: str = Field(default="changeme", env="DB_PASSWORD")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis cache configuration."""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class KafkaConfig(BaseSettings):
    """Kafka message broker configuration."""
    brokers: List[str] = Field(default=["localhost:9092"], env="KAFKA_BROKERS")
    topic_prefix: str = Field(default="aegis", env="KAFKA_TOPIC_PREFIX")
    consumer_group: str = Field(default="aegis_consumer", env="KAFKA_GROUP")


class RiskConfig(BaseSettings):
    """Risk management parameters."""
    max_drawdown_threshold: float = Field(default=0.08, description="Maximum allowed drawdown (8%)")
    target_volatility: float = Field(default=0.15, description="Target annual volatility (15%)")
    kelly_fraction: float = Field(default=0.5, description="Fraction of Kelly criterion to use")
    var_confidence: float = Field(default=0.99, description="VaR confidence level")
    max_position_size: float = Field(default=0.20, description="Max position size per asset (20%)")
    correlation_threshold: float = Field(default=0.7, description="Correlation limit for diversification")
    
    @validator('max_drawdown_threshold', 'target_volatility', 'kelly_fraction')
    def check_ranges(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Risk parameters must be between 0 and 1")
        return v


class MLConfig(BaseSettings):
    """Machine learning system configuration."""
    model_registry_path: str = Field(default="./models", env="MODEL_REGISTRY")
    hyperparam_trials: int = Field(default=100, description="Number of Bayesian optimization trials")
    ensemble_size: int = Field(default=10, description="Number of models in ensemble")
    drift_detection_window: int = Field(default=252, description="Trading days for drift detection")
    retrain_threshold: float = Field(default=0.1, description="Performance degradation threshold for retraining")


class ExecutionConfig(BaseSettings):
    """Execution engine parameters."""
    default_slippage_bps: float = Field(default=5.0, description="Default slippage assumption in bps")
    latency_budget_ms: int = Field(default=10, description="Maximum allowed latency in milliseconds")
    retry_attempts: int = Field(default=3, description="Order retry attempts on failure")
    twap_duration_minutes: int = Field(default=60, description="Default TWAP duration")


class SystemConfig(BaseSettings):
    """Master configuration orchestrating all subsystems."""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = SystemConfig()
    return _config


def reload_config() -> SystemConfig:
    """Force reload configuration from environment."""
    global _config
    _config = SystemConfig()
    return _config
