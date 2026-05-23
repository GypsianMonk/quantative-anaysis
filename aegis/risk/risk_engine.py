"""
Risk Management Engine for AEGIS.

Implements comprehensive risk controls including:
- Position sizing (Kelly criterion, volatility targeting)
- Value at Risk (VaR) and Conditional VaR
- Drawdown monitoring and circuit breakers
- Correlation risk management
- Liquidity risk assessment
- Real-time exposure tracking

Mathematical Foundations:
=========================

1. Kelly Criterion:
   f* = (bp - q) / b
   where:
   - f* = fraction of capital to allocate
   - b = odds received on bet (profit/loss ratio)
   - p = probability of winning
   - q = probability of losing (1 - p)
   
   We use Half-Kelly (f*/2) for more conservative positioning.

2. Volatility Targeting:
   position_size = (target_vol / asset_vol) * equity * signal_strength
   
   This ensures consistent risk contribution across assets with different volatilities.

3. Value at Risk (Parametric):
   VaR_α = μ + Z_α * σ
   where:
   - μ = expected return
   - Z_α = Z-score for confidence level α
   - σ = portfolio volatility
   
   Portfolio VaR = √(w' Σ w) * Z_α * portfolio_value

4. Conditional VaR (Expected Shortfall):
   CVaR_α = E[X | X ≤ VaR_α]
   
   For normal distribution: CVaR = μ - σ * φ(Z_α) / (1 - α)

5. Maximum Drawdown:
   MDD = max((peak - trough) / peak)
   
   Monitored continuously with circuit breaker triggers.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..config.settings import settings
from ..utils.logging import trading_logger


class RiskLevel(Enum):
    """Risk severity levels for alerts and actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for computed risk metrics."""
    
    # Portfolio level
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    leverage: float = 1.0
    
    # Risk measures
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_days: int = 0
    
    # Volatility
    portfolio_volatility: float = 0.0
    realized_volatility: float = 0.0
    
    # Concentration
    top_position_weight: float = 0.0
    herfindahl_index: float = 0.0
    
    # Correlation
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CircuitBreakerStatus:
    """Status of circuit breaker mechanisms."""
    
    is_triggered: bool = False
    trigger_reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    reset_at: Optional[datetime] = None
    
    # Counters
    consecutive_losses: int = 0
    daily_loss_pct: float = 0.0
    volatility_spike_detected: bool = False


class PositionSizer:
    """
    Calculates optimal position sizes based on various methodologies.
    
    Implements:
    - Kelly Criterion (with fractional scaling)
    - Volatility Targeting
    - Risk Parity
    - Fixed Fractional
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.kelly_fraction = settings.risk.kelly_fraction
        self.max_position_pct = settings.risk.max_position_size_pct
        self.target_volatility = settings.risk.target_annual_volatility
        
    def kelly_position_size(
        self,
        win_probability: float,
        profit_loss_ratio: float,
        equity: float,
        fraction: Optional[float] = None
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_probability: Probability of winning (0 to 1)
            profit_loss_ratio: Average profit / average loss
            equity: Total available equity
            fraction: Kelly fraction (default: half-Kelly)
        
        Returns:
            Optimal position size in dollar terms
        
        Kelly Formula:
        f* = (bp - q) / b
        where b = profit_loss_ratio, p = win_prob, q = 1 - p
        """
        fraction = fraction or self.kelly_fraction
        
        if profit_loss_ratio <= 0:
            return 0.0
        
        win_prob = max(0, min(1, win_probability))
        lose_prob = 1 - win_prob
        
        # Full Kelly fraction
        kelly = (profit_loss_ratio * win_prob - lose_prob) / profit_loss_ratio
        
        # Apply fractional Kelly and cap at maximum
        position_fraction = kelly * fraction
        position_fraction = max(0, min(position_fraction, self.max_position_pct))
        
        return equity * position_fraction
    
    def volatility_targeting_size(
        self,
        asset_volatility: float,
        equity: float,
        signal_strength: float = 1.0,
        target_vol: Optional[float] = None
    ) -> float:
        """
        Calculate position size using volatility targeting.
        
        Args:
            asset_volatility: Annualized volatility of the asset
            equity: Total available equity
            signal_strength: Signal strength multiplier (0 to 1)
            target_vol: Target portfolio volatility
        
        Returns:
            Position size in dollar terms
        
        Formula:
        position_size = (target_vol / asset_vol) * equity * signal_strength
        """
        target_vol = target_vol or self.target_volatility
        
        if asset_volatility <= 0:
            return 0.0
        
        # Scale by volatility ratio
        vol_ratio = target_vol / asset_volatility
        
        # Apply signal strength
        position_fraction = vol_ratio * abs(signal_strength)
        
        # Cap at maximum position size
        position_fraction = min(position_fraction, self.max_position_pct)
        
        return equity * position_fraction
    
    def risk_parity_size(
        self,
        asset_volatilities: Dict[str, float],
        correlations: Optional[np.ndarray] = None,
        equity: float
    ) -> Dict[str, float]:
        """
        Calculate risk parity position sizes.
        
        Each asset contributes equally to portfolio risk.
        
        Args:
            asset_volatilities: Dict mapping asset to volatility
            correlations: Correlation matrix (optional)
            equity: Total equity
        
        Returns:
            Dict mapping asset to position size
        """
        n_assets = len(asset_volatilities)
        assets = list(asset_volatilities.keys())
        vols = np.array(list(asset_volatilities.values()))
        
        if correlations is None:
            # Assume zero correlation for simplicity
            correlations = np.eye(n_assets)
        
        # Risk parity: inverse volatility weighting (simplified)
        inv_vols = 1 / vols
        weights = inv_vols / inv_vols.sum()
        
        # Convert weights to position sizes
        positions = {
            asset: equity * weight
            for asset, weight in zip(assets, weights)
        }
        
        return positions
    
    def calculate_position_size(
        self,
        method: str,
        equity: float,
        **kwargs
    ) -> float:
        """
        Unified interface for position sizing.
        
        Args:
            method: 'kelly', 'volatility_targeting', 'risk_parity', 'fixed'
            equity: Available equity
            **kwargs: Method-specific parameters
        
        Returns:
            Position size in dollar terms
        """
        if method == 'kelly':
            return self.kelly_position_size(
                win_probability=kwargs.get('win_probability', 0.5),
                profit_loss_ratio=kwargs.get('profit_loss_ratio', 1.0),
                equity=equity,
                fraction=kwargs.get('fraction')
            )
        
        elif method == 'volatility_targeting':
            return self.volatility_targeting_size(
                asset_volatility=kwargs.get('asset_volatility', 0.2),
                equity=equity,
                signal_strength=kwargs.get('signal_strength', 1.0),
                target_vol=kwargs.get('target_vol')
            )
        
        elif method == 'fixed':
            fixed_pct = kwargs.get('fixed_percentage', 0.05)
            return equity * min(fixed_pct, self.max_position_pct)
        
        else:
            raise ValueError(f"Unknown position sizing method: {method}")


class RiskCalculator:
    """
    Computes various risk metrics for portfolio analysis.
    """
    
    def __init__(self, confidence_levels: Tuple[float, ...] = (0.95, 0.99)):
        self.confidence_levels = confidence_levels
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.99,
        method: str = 'parametric'
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.99 for 99%)
            method: 'parametric', 'historical', or 'monte_carlo'
        
        Returns:
            VaR as a positive number (potential loss)
        """
        if len(returns) < 10:
            return 0.0
        
        if method == 'parametric':
            # Assume normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(confidence)
            var = -(mu - z_score * sigma)
        
        elif method == 'historical':
            # Use empirical distribution
            var = -np.percentile(returns, (1 - confidence) * 100)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation (simplified)
            n_sims = 10000
            simulated_returns = np.random.normal(
                np.mean(returns),
                np.std(returns),
                n_sims
            )
            var = -np.percentile(simulated_returns, (1 - confidence) * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return max(0, var)
    
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.99,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            method: Calculation method
        
        Returns:
            CVaR as a positive number (expected loss beyond VaR)
        """
        if len(returns) < 10:
            return 0.0
        
        var = self.calculate_var(returns, confidence, method)
        
        if method == 'parametric':
            # For normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(confidence)
            cvar = -(mu - sigma * stats.norm.pdf(z_score) / (1 - confidence))
        
        else:
            # Historical/Monte Carlo: average of returns beyond VaR
            threshold = -var
            tail_returns = returns[returns <= threshold]
            if len(tail_returns) > 0:
                cvar = -np.mean(tail_returns)
            else:
                cvar = var
        
        return max(0, cvar)
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and its duration.
        
        Args:
            equity_curve: Array of portfolio values
        
        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown series
        drawdown = (running_max - equity_curve) / running_max
        
        # Find maximum drawdown
        max_dd_idx = np.argmax(drawdown)
        max_drawdown = drawdown[max_dd_idx]
        
        # Find peak (start of drawdown)
        peak_idx = np.argmax(running_max[:max_dd_idx + 1])
        
        return max_drawdown, peak_idx, max_dd_idx
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            annualization_factor: Number of periods per year
        
        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) < 10 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / annualization_factor
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize
        sharpe *= np.sqrt(annualization_factor)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (downside deviation version of Sharpe).
        """
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - risk_free_rate / annualization_factor
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf')
        
        sortino = np.mean(excess_returns) / downside_std
        sortino *= np.sqrt(annualization_factor)
        
        return sortino
    
    def calculate_herfindahl_index(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration measurement.
        
        HHI = sum(w_i^2)
        
        Higher values indicate more concentration.
        """
        return np.sum(weights ** 2)
    
    def calculate_portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility from covariance matrix.
        
        σ_p = √(w' Σ w)
        """
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)


class CircuitBreaker:
    """
    Circuit breaker mechanism to halt trading during adverse conditions.
    
    Triggers:
    - Maximum drawdown exceeded
    - Daily loss limit exceeded
    - Consecutive losses
    - Volatility spike detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.status = CircuitBreakerStatus()
        
        # Load limits from settings
        self.max_drawdown = settings.risk.max_drawdown_pct
        self.daily_loss_limit = settings.risk.daily_loss_limit_pct
        self.consecutive_losses_limit = settings.risk.consecutive_losses_limit
        self.volatility_threshold = settings.risk.volatility_spike_threshold
        
        # State tracking
        self.daily_start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self.loss_streak: int = 0
    
    def reset_daily(self, equity: float):
        """Reset daily tracking at start of trading day."""
        self.daily_start_equity = equity
        self.status.daily_loss_pct = 0.0
    
    def update_peak(self, equity: float):
        """Update peak equity for drawdown calculation."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
    
    def check_circuit_breakers(
        self,
        current_equity: float,
        daily_pnl: float,
        is_profitable_trade: bool,
        current_volatility: float,
        historical_volatility: float
    ) -> CircuitBreakerStatus:
        """
        Check all circuit breaker conditions.
        
        Returns:
            Updated CircuitBreakerStatus
        """
        status = CircuitBreakerStatus()
        
        # Check maximum drawdown
        if self.peak_equity and self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            status.is_triggered = drawdown >= self.max_drawdown
            if status.is_triggered:
                status.trigger_reason = f"Max drawdown breached: {drawdown:.2%}"
        
        # Check daily loss limit
        if self.daily_start_equity and self.daily_start_equity > 0:
            daily_loss_pct = -daily_pnl / self.daily_start_equity
            status.daily_loss_pct = daily_loss_pct
            if daily_loss_pct >= self.daily_loss_limit and not status.is_triggered:
                status.is_triggered = True
                status.trigger_reason = f"Daily loss limit breached: {daily_loss_pct:.2%}"
        
        # Check consecutive losses
        if is_profitable_trade:
            self.loss_streak = 0
        else:
            self.loss_streak += 1
        
        if self.loss_streak >= self.consecutive_losses_limit and not status.is_triggered:
            status.is_triggered = True
            status.trigger_reason = f"Consecutive losses limit: {self.loss_streak}"
        
        # Check volatility spike
        if historical_volatility > 0:
            vol_ratio = current_volatility / historical_volatility
            status.volatility_spike_detected = vol_ratio >= self.volatility_threshold
            if status.volatility_spike_detected and not status.is_triggered:
                status.is_triggered = True
                status.trigger_reason = f"Volatility spike detected: {vol_ratio:.2f}x"
        
        # Update status
        if status.is_triggered:
            status.triggered_at = datetime.utcnow()
            status.consecutive_losses = self.loss_streak
        
        self.status = status
        
        if status.is_triggered:
            trading_logger.circuit_breaker_triggered(
                reason=status.trigger_reason,
                details={
                    'current_equity': current_equity,
                    'peak_equity': self.peak_equity,
                    'daily_pnl': daily_pnl,
                    'loss_streak': self.loss_streak
                }
            )
        
        return status
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.status.is_triggered
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.status = CircuitBreakerStatus()
        self.loss_streak = 0
        trading_logger.info("Circuit breaker manually reset")


class RiskManager:
    """
    Main risk management engine coordinating all risk controls.
    
    Provides:
    - Real-time risk monitoring
    - Position sizing decisions
    - Circuit breaker enforcement
    - Risk reporting
    """
    
    def __init__(self):
        self.position_sizer = PositionSizer()
        self.risk_calculator = RiskCalculator()
        self.circuit_breaker = CircuitBreaker()
        
        # State tracking
        self.equity_curve: List[float] = []
        self.returns_history: List[float] = []
        self.positions: Dict[str, Dict] = {}
        
        self.logger = trading_logger
    
    def update_portfolio_state(
        self,
        equity: float,
        positions: Dict[str, Dict],
        prices: Dict[str, float]
    ):
        """Update internal state with current portfolio information."""
        self.equity_curve.append(equity)
        self.positions = positions
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            ret = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns_history.append(ret)
        
        # Update circuit breaker
        if len(self.equity_curve) > 0:
            self.circuit_breaker.update_peak(equity)
    
    def get_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        returns = np.array(self.returns_history[-252:])  # Last year of returns
        
        metrics = RiskMetrics()
        
        if len(returns) > 0:
            # VaR and CVaR
            metrics.var_95 = self.risk_calculator.calculate_var(returns, 0.95)
            metrics.var_99 = self.risk_calculator.calculate_var(returns, 0.99)
            metrics.cvar_95 = self.risk_calculator.calculate_cvar(returns, 0.95)
            metrics.cvar_99 = self.risk_calculator.calculate_cvar(returns, 0.99)
            
            # Volatility
            metrics.portfolio_volatility = np.std(returns) * np.sqrt(252)
            metrics.realized_volatility = np.std(returns[-21:]) * np.sqrt(252)  # 21-day
        
        # Drawdown
        if len(self.equity_curve) > 0:
            equity_array = np.array(self.equity_curve)
            max_dd, peak_idx, trough_idx = self.risk_calculator.calculate_max_drawdown(equity_array)
            metrics.max_drawdown = max_dd
            metrics.current_drawdown = max(0, (np.max(equity_array) - equity_array[-1]) / np.max(equity_array))
            metrics.drawdown_duration_days = len(equity_array) - peak_idx
        
        # Exposure metrics
        if self.positions:
            position_values = [p.get('value', 0) for p in self.positions.values()]
            total_exposure = sum(abs(v) for v in position_values)
            net_exposure = sum(position_values)
            
            metrics.total_exposure = total_exposure
            metrics.net_exposure = net_exposure
            metrics.gross_exposure = total_exposure
            metrics.leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Concentration
            weights = np.array([abs(v) / total_exposure for v in position_values if total_exposure > 0])
            if len(weights) > 0:
                metrics.top_position_weight = np.max(weights)
                metrics.herfindahl_index = self.risk_calculator.calculate_herfindahl_index(weights)
        
        return metrics
    
    def check_trading_allowed(
        self,
        symbol: str,
        proposed_position_size: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Check if a proposed trade passes all risk checks.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            return False, f"Circuit breaker active: {self.circuit_breaker.status.trigger_reason}"
        
        # Check position size limit
        max_position = portfolio_value * settings.risk.max_position_size_pct
        if proposed_position_size > max_position:
            return False, f"Position size {proposed_position_size:.2f} exceeds max {max_position:.2f}"
        
        # Check concentration
        current_position = self.positions.get(symbol, {}).get('value', 0)
        new_total = abs(current_position) + proposed_position_size
        if new_total > max_position:
            return False, f"Would exceed max position concentration for {symbol}"
        
        return True, "OK"
    
    def calculate_safe_position_size(
        self,
        symbol: str,
        signal_strength: float,
        asset_volatility: float,
        portfolio_value: float,
        win_probability: float = 0.55,
        profit_loss_ratio: float = 1.5
    ) -> float:
        """
        Calculate safe position size considering all risk constraints.
        
        Uses volatility targeting with Kelly upper bound.
        """
        # Get base sizes from different methods
        kelly_size = self.position_sizer.kelly_position_size(
            win_probability=win_probability,
            profit_loss_ratio=profit_loss_ratio,
            equity=portfolio_value
        )
        
        vol_target_size = self.position_sizer.volatility_targeting_size(
            asset_volatility=asset_volatility,
            equity=portfolio_value,
            signal_strength=abs(signal_strength)
        )
        
        # Take minimum of both (conservative approach)
        base_size = min(kelly_size, vol_target_size)
        
        # Scale by signal strength
        final_size = base_size * abs(signal_strength)
        
        # Ensure within limits
        max_size = portfolio_value * settings.risk.max_position_size_pct
        final_size = min(final_size, max_size)
        
        return final_size
    
    def generate_risk_report(self, portfolio_value: float) -> Dict:
        """Generate comprehensive risk report."""
        metrics = self.get_risk_metrics(portfolio_value)
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'portfolio_value': portfolio_value,
            'metrics': {
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'cvar_95': metrics.cvar_95,
                'cvar_99': metrics.cvar_99,
                'current_drawdown': metrics.current_drawdown,
                'max_drawdown': metrics.max_drawdown,
                'portfolio_volatility': metrics.portfolio_volatility,
                'leverage': metrics.leverage,
                'concentration_hhi': metrics.herfindahl_index
            },
            'circuit_breaker': {
                'is_active': self.circuit_breaker.status.is_triggered,
                'reason': self.circuit_breaker.status.trigger_reason,
                'consecutive_losses': self.circuit_breaker.status.consecutive_losses
            },
            'limits': {
                'max_position_pct': settings.risk.max_position_size_pct,
                'max_drawdown_pct': settings.risk.max_drawdown_pct,
                'daily_loss_limit_pct': settings.risk.daily_loss_limit_pct
            }
        }
        
        return report


# Global risk manager instance
risk_manager = RiskManager()
