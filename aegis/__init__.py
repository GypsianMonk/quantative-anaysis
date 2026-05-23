# AEGIS: Advanced Ensemble Generative Institutional System
# Production-Ready Quantitative Trading Framework

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"
__license__ = "Proprietary"

"""
AEGIS Architecture Overview:
============================

This system implements a modular, event-driven architecture for institutional-grade
quantitative trading across multiple asset classes (Equities, Crypto, Forex, Futures).

Core Design Principles:
1. Modularity: Decoupled components for independent scaling and maintenance
2. Robustness: Comprehensive error handling, circuit breakers, and fail-safes
3. Performance: Low-latency execution with async I/O and optimized data structures
4. Validation: Rigorous backtesting with walk-forward analysis and purged CV
5. Risk-First: Position sizing, exposure limits, and dynamic leverage controls

System Components:
- data/: Market data ingestion, storage, and retrieval
- features/: Feature engineering with lookahead bias prevention
- models/: ML models (XGBoost, LSTM, Transformers, RL)
- strategies/: Alpha generation and portfolio construction
- risk/: Risk management, position sizing, and circuit breakers
- execution/: Order routing, smart execution, and fill handling
- backtest/: Event-driven and vectorized backtesting engines
- pipeline/: Data pipelines and workflow orchestration
- monitoring/: Real-time dashboards and alerting
- utils/: Common utilities, logging, and configuration

Mathematical Foundations:
- Kelly Criterion (Half-Kelly for safety): f* = (bp - q) / b
- Volatility Targeting: Position Size = (Target Vol / Asset Vol) × Equity
- VaR/CVaR: Parametric and historical methods
- Sharpe Ratio: (Return - Risk-Free Rate) / StdDev(Excess Return)
- Maximum Drawdown: Peak-to-trough decline monitoring

Performance Targets:
- Sharpe Ratio > 2.0
- Maximum Drawdown < 10%
- Win Rate > 55%
- Profit Factor > 1.5
- Transaction Cost Awareness: Slippage + Commission modeling
"""

from .config.settings import settings
from .utils.logging import setup_logging

# Initialize logging
setup_logging()

__all__ = [
    "data",
    "features", 
    "models",
    "strategies",
    "risk",
    "execution",
    "backtest",
    "pipeline",
    "monitoring",
    "utils"
]
