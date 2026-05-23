"""
AEGIS PRIME: Autonomous Quantitative Hedge Fund Ecosystem
=========================================================

Copyright (c) 2024 AEGIS Prime Research Lab. All Rights Reserved.

This system implements a self-evolving, regime-adaptive trading architecture
designed for institutional-grade alpha generation across global asset classes.

Core Principles:
1. Probabilistic Decision Making under Uncertainty
2. Adversarial Robustness to Market Regime Shifts
3. Continuous Online Learning and Strategy Evolution
4. Microstructure-Aware Execution
5. Risk-First Capital Allocation

Architecture Layers:
- Data Ingestion & Feature Store (Real-time Stream Processing)
- Alpha Discovery Engine (Symbolic Regression + Deep Learning)
- Regime Detection & State Classification (HMM + Transformers)
- Dynamic Risk Management (Kelly + VaR/CVaR + Circuit Breakers)
- Low-Latency Execution (Smart Order Routing + RL Optimization)
- Self-Improvement Loop (Auto-research, Validation, Deployment, Retirement)

Target Metrics:
- Sharpe Ratio > 3.0
- Maximum Drawdown < 8%
- Calmar Ratio > 2.0
- Tail Risk Resilience (99% CVaR control)
"""

__version__ = "1.0.0"
__author__ = "AEGIS Prime Research Lab"
__status__ = "Production"

from .core.regime import RegimeDetector, MarketState
from .core.entropy import EntropyCalculator, FractalDimension
from .data.pipeline import DataIngestionEngine, FeatureStore
from .alpha.discovery import AlphaMiner, GeneticFactorGenerator
from .risk.engine import DynamicRiskManager, PortfolioOptimizer
from .execution.smart_router import SmartOrderRouter, ExecutionOptimizer
from .infra.config import SystemConfig

__all__ = [
    "RegimeDetector",
    "MarketState",
    "EntropyCalculator",
    "FractalDimension",
    "DataIngestionEngine",
    "FeatureStore",
    "AlphaMiner",
    "GeneticFactorGenerator",
    "DynamicRiskManager",
    "PortfolioOptimizer",
    "SmartOrderRouter",
    "ExecutionOptimizer",
    "SystemConfig",
]
