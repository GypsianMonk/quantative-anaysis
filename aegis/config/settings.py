"""
Configuration management for AEGIS trading system.

Handles environment variables, database connections, API keys, and system settings.
Uses Pydantic for validation and type safety.
"""

from pydantic import BaseSettings, Field, validator
from typing import Dict, List, Optional
import os
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration for PostgreSQL/TimescaleDB."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="aegis", env="DB_NAME")
    user: str = Field(default="aegis_user", env="DB_USER")
    password: str = Field(default="changeme", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def timescale_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class KafkaSettings(BaseSettings):
    """Kafka message broker configuration."""
    
    bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    topic_prefix: str = Field(default="aegis", env="KAFKA_TOPIC_PREFIX")
    consumer_group: str = Field(default="aegis-consumer", env="KAFKA_CONSUMER_GROUP")
    auto_offset_reset: str = Field(default="earliest", env="KAFKA_AUTO_OFFSET_RESET")


class ExchangeSettings(BaseSettings):
    """Exchange API configuration."""
    
    # Binance
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # Interactive Brokers
    ib_host: str = Field(default="127.0.0.1", env="IB_HOST")
    ib_port: int = Field(default=7497, env="IB_PORT")  # 7497 for paper, 7496 for live
    ib_client_id: int = Field(default=1, env="IB_CLIENT_ID")
    
    # CCXT universal
    ccxt_exchanges: List[str] = Field(default=["binance", "kraken", "ftx"], env="CCXT_EXCHANGES")


class RiskSettings(BaseSettings):
    """Risk management parameters."""
    
    # Position sizing
    max_position_size_pct: float = Field(default=0.20, ge=0, le=1)  # 20% max per asset
    max_portfolio_leverage: float = Field(default=2.0, ge=1)  # Max 2x leverage
    target_annual_volatility: float = Field(default=0.15)  # 15% target vol
    
    # Drawdown controls
    max_drawdown_pct: float = Field(default=0.10, ge=0, le=1)  # 10% max DD
    daily_loss_limit_pct: float = Field(default=0.03, ge=0, le=1)  # 3% daily loss limit
    
    # Kelly criterion
    kelly_fraction: float = Field(default=0.5, ge=0, le=1)  # Half-Kelly for safety
    
    # VaR/CVaR
    var_confidence: float = Field(default=0.99)  # 99% VaR
    cvar_confidence: float = Field(default=0.99)  # 99% CVaR
    
    # Circuit breakers
    circuit_breaker_enabled: bool = Field(default=True)
    consecutive_losses_limit: int = Field(default=5)
    volatility_spike_threshold: float = Field(default=3.0)  # Std dev multiplier


class BacktestSettings(BaseSettings):
    """Backtesting configuration."""
    
    # Data handling
    start_date: str = Field(default="2020-01-01", env="BACKTEST_START_DATE")
    end_date: str = Field(default="2023-12-31", env="BACKTEST_END_DATE")
    initial_capital: float = Field(default=1_000_000, env="INITIAL_CAPITAL")
    
    # Transaction costs
    commission_rate: float = Field(default=0.001)  # 0.1% commission
    slippage_model: str = Field(default="volume_proportional")  # volume_proportional, fixed, random
    slippage_factor: float = Field(default=0.0005)  # 0.05% slippage
    
    # Validation
    walk_forward_windows: int = Field(default=4)
    purged_cv_folds: int = Field(default=5)
    embargo_pct: float = Field(default=0.02)  # 2% embargo for purged CV
    
    # Monte Carlo
    mc_simulations: int = Field(default=1000)
    mc_block_size: int = Field(default=21)  # ~1 month of daily data


class MLSettings(BaseSettings):
    """Machine learning configuration."""
    
    # Model selection
    default_models: List[str] = Field(default=["xgboost", "lightgbm", "lstm"])
    ensemble_method: str = Field(default="weighted_average")  # weighted_average, stacking, voting
    
    # Hyperparameter optimization
    optuna_trials: int = Field(default=100)
    optuna_timeout: int = Field(default=3600)  # 1 hour
    bayesian_opt_iterations: int = Field(default=50)
    
    # Training
    train_test_split: float = Field(default=0.8)
    validation_split: float = Field(default=0.1)
    early_stopping_rounds: int = Field(default=50)
    max_features: int = Field(default=100)
    
    # Drift detection
    drift_detection_enabled: bool = Field(default=True)
    drift_threshold: float = Field(default=0.05)
    retrain_frequency_days: int = Field(default=7)


class ExecutionSettings(BaseSettings):
    """Execution engine configuration."""
    
    # Order routing
    smart_routing_enabled: bool = Field(default=True)
    routing_latency_threshold_ms: int = Field(default=10)
    
    # Execution algorithms
    twap_duration_minutes: int = Field(default=60)
    vwap_participation_rate: float = Field(default=0.1)  # 10% of volume
    
    # Retry logic
    max_retries: int = Field(default=3)
    retry_delay_seconds: float = Field(default=1.0)
    
    # Latency monitoring
    latency_monitoring_enabled: bool = Field(default=True)
    latency_alert_threshold_ms: int = Field(default=100)


class MonitoringSettings(BaseSettings):
    """Monitoring and alerting configuration."""
    
    # Prometheus
    prometheus_port: int = Field(default=9090)
    metrics_enabled: bool = Field(default=True)
    
    # Grafana
    grafana_url: str = Field(default="http://localhost:3000")
    grafana_api_key: Optional[str] = Field(default=None, env="GRAFANA_API_KEY")
    
    # Alerting
    slack_webhook: Optional[str] = Field(default=None, env="SLACK_WEBHOOK")
    email_alerts: bool = Field(default=False)
    alert_email: Optional[str] = Field(default=None, env="ALERT_EMAIL")


class Settings(BaseSettings):
    """Main settings class aggregating all configurations."""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
