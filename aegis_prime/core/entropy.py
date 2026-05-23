"""
Entropy and Fractal Analysis Module
====================================

Implements information-theoretic measures and fractal geometry tools
for market complexity analysis, efficiency measurement, and regime detection.

Mathematical Foundations:
1. Shannon Entropy: H(X) = -Σ p(x) * log2(p(x))
   - Measures uncertainty/disorder in return distribution
   - High entropy = high unpredictability

2. Sample Entropy (SampEn):
   - Quantifies regularity/complexity in time series
   - Lower values indicate more predictable patterns

3. Hurst Exponent (H):
   - H < 0.5: Mean-reverting (anti-persistent)
   - H = 0.5: Random walk
   - H > 0.5: Trending (persistent)
   - Calculated via Rescaled Range (R/S) analysis

4. Fractal Dimension (D):
   - D = 2 - H for self-affine processes
   - Measures roughness/complexity of price paths
   - Higher D = more jagged, complex paths

5. Multiscale Entropy:
   - Entropy computed at multiple time scales
   - Detects structure across different frequencies
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ComplexityMetrics:
    """Container for entropy and fractal analysis results."""
    shannon_entropy: float
    sample_entropy: float
    hurst_exponent: float
    fractal_dimension: float
    multiscale_entropy: np.ndarray
    efficiency_ratio: float
    timestamp: int


class EntropyCalculator:
    """
    Comprehensive entropy calculation suite for financial time series.
    
    Applications:
    - Regime change detection (entropy spikes precede transitions)
    - Market efficiency measurement
    - Strategy suitability assessment (mean-reversion vs trend-following)
    - Risk estimation (high entropy = high uncertainty)
    """
    
    def __init__(self, max_scale: int = 10):
        self.max_scale = max_scale
    
    def calculate_shannon_entropy(self, returns: np.ndarray, bins: int = 20) -> float:
        """
        Calculate Shannon entropy of return distribution.
        
        H(X) = -Σ p(x) * log2(p(x))
        
        Interpretation:
        - Low entropy (< 3.0): Predictable, structured returns
        - High entropy (> 4.0): Random, unpredictable returns
        """
        if len(returns) < 10:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        
        # Normalize to probabilities
        probs = hist / np.sum(hist)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    def calculate_sample_entropy(
        self, 
        series: np.ndarray, 
        m: int = 2, 
        r: float = 0.2
    ) -> float:
        """
        Calculate Sample Entropy (SampEn).
        
        Measures the likelihood that similar patterns remain similar
        at the next point, excluding self-matches.
        
        Parameters:
        - m: Embedding dimension (pattern length)
        - r: Tolerance threshold (typically 0.1-0.25 * std)
        
        Lower SampEn = more regular/predictable
        Higher SampEn = more complex/unpredictable
        """
        n = len(series)
        if n < m + 1:
            return 0.0
        
        # Standardize series
        series = (series - np.mean(series)) / (np.std(series) + 1e-10)
        tolerance = r * np.std(series)
        
        # Count matches of length m
        def count_matches(embed_dim: int) -> int:
            matches = 0
            total = 0
            for i in range(n - embed_dim):
                template = series[i:i + embed_dim]
                for j in range(i + 1, n - embed_dim + 1):
                    candidate = series[j:j + embed_dim]
                    if np.max(np.abs(template - candidate)) < tolerance:
                        matches += 1
                    total += 1
            return matches, total
        
        matches_m, total_m = count_matches(m)
        matches_m1, total_m1 = count_matches(m + 1)
        
        # Avoid division by zero
        if matches_m == 0 or matches_m1 == 0:
            return np.inf
        
        # Sample entropy
        phi_m = matches_m / total_m
        phi_m1 = matches_m1 / total_m1
        
        sampen = -np.log(phi_m1 / phi_m + 1e-10)
        
        return sampen if np.isfinite(sampen) else 5.0
    
    def calculate_multiscale_entropy(self, series: np.ndarray) -> np.ndarray:
        """
        Calculate entropy at multiple time scales.
        
        Coarse-graining procedure:
        For scale factor τ, create coarse-grained series:
        y_τ(j) = (1/τ) * Σ x(i) for i from (j-1)τ+1 to jτ
        
        Returns array of entropy values for scales 1 to max_scale.
        """
        n = len(series)
        mse_values = []
        
        for scale in range(1, self.max_scale + 1):
            # Coarse-graining
            n_scaled = n // scale
            coarse_series = np.zeros(n_scaled)
            
            for j in range(n_scaled):
                start_idx = j * scale
                end_idx = min((j + 1) * scale, n)
                coarse_series[j] = np.mean(series[start_idx:end_idx])
            
            # Calculate SampEn on coarse-grained series
            if len(coarse_series) > 10:
                entropy = self.calculate_sample_entropy(coarse_series)
                mse_values.append(entropy if np.isfinite(entropy) else 5.0)
            else:
                mse_values.append(np.nan)
        
        return np.array(mse_values)


class FractalDimension:
    """
    Fractal geometry analysis for financial time series.
    
    Key metrics:
    1. Hurst Exponent (H): Long-term memory indicator
    2. Fractal Dimension (D): Path complexity measure
    3. Efficiency Ratio: Trend vs noise quantification
    """
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst Exponent using Rescaled Range (R/S) analysis.
        
        R/S statistic:
        R(n) = max(S) - min(S) where S(k) = Σ (x(i) - mean(x))
        S(n) = standard deviation
        
        E[R(n)/S(n)] = c * n^H
        
        Interpretation:
        - H < 0.5: Mean-reverting (anti-persistent)
        - H = 0.5: Random walk (geometric Brownian motion)
        - H > 0.5: Trending (persistent, long memory)
        """
        n = len(prices)
        if n < 20:
            return 0.5
        
        # Use log returns for stationarity
        if prices[0] > 0:
            returns = np.diff(np.log(prices))
        else:
            returns = np.diff(prices)
        
        if len(returns) < 20:
            return 0.5
        
        # Lag sizes for R/S analysis
        lags = np.unique(np.logspace(1, np.log10(len(returns) / 2), num=20).astype(int))
        lags = lags[lags >= 10]  # Minimum lag size
        
        if len(lags) < 3:
            return 0.5
        
        rs_values = []
        
        for lag in lags:
            if lag > len(returns):
                continue
            
            # Calculate cumulative deviations
            mean_ret = np.mean(returns[:lag])
            deviations = returns[:lag] - mean_ret
            cum_dev = np.cumsum(deviations)
            
            # Range R
            R = np.max(cum_dev) - np.min(cum_dev)
            
            # Standard deviation S
            S = np.std(returns[:lag])
            
            if S > 1e-10:
                rs_values.append(R / S)
            else:
                rs_values.append(1.0)
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression: log(R/S) = log(c) + H * log(n)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Handle NaN values
        valid_mask = ~(np.isnan(log_lags) | np.isnan(log_rs))
        if np.sum(valid_mask) < 3:
            return 0.5
        
        slope, _ = np.polyfit(log_lags[valid_mask], log_rs[valid_mask], 1)
        
        # Clamp to reasonable range
        hurst = np.clip(slope, 0.0, 1.0)
        
        return hurst
    
    def calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """
        Calculate Fractal Dimension using Hurst exponent relationship.
        
        For self-affine processes: D = 2 - H
        
        Interpretation:
        - D ≈ 1.5: Random walk (Brownian motion)
        - D < 1.5: Smooth, trending paths
        - D > 1.5: Rough, mean-reverting paths
        """
        hurst = self.calculate_hurst_exponent(prices)
        dimension = 2.0 - hurst
        
        return np.clip(dimension, 1.0, 2.0)
    
    def calculate_efficiency_ratio(self, prices: np.ndarray, window: int = 20) -> float:
        """
        Calculate Kaufman's Efficiency Ratio (ER).
        
        ER = Net Change / Sum of Absolute Changes
        
        Interpretation:
        - ER ≈ 1.0: Perfect trend (all movement in one direction)
        - ER ≈ 0.0: Pure noise (equal up/down movements)
        - ER > 0.5: Trending market
        - ER < 0.3: Choppy/ranging market
        """
        if len(prices) < window + 1:
            return 0.5
        
        recent_prices = prices[-window:]
        
        # Net change
        net_change = abs(recent_prices[-1] - recent_prices[0])
        
        # Sum of absolute changes
        changes = np.abs(np.diff(recent_prices))
        total_movement = np.sum(changes)
        
        if total_movement < 1e-10:
            return 0.5
        
        efficiency = net_change / total_movement
        
        return np.clip(efficiency, 0.0, 1.0)
    
    def analyze_market_state(self, prices: np.ndarray) -> dict:
        """
        Comprehensive fractal analysis for market state characterization.
        
        Returns dictionary with:
        - hurst_exponent
        - fractal_dimension
        - efficiency_ratio
        - market_type (trending/mean-reverting/random)
        - recommended_strategy
        """
        hurst = self.calculate_hurst_exponent(prices)
        dimension = self.calculate_fractal_dimension(prices)
        efficiency = self.calculate_efficiency_ratio(prices)
        
        # Classify market type
        if hurst > 0.6:
            market_type = "trending"
            strategy = "momentum/trend-following"
        elif hurst < 0.4:
            market_type = "mean-reverting"
            strategy = "mean-reversion/statistical arbitrage"
        else:
            market_type = "random_walk"
            strategy = "breakout/volatility-based"
        
        return {
            "hurst_exponent": hurst,
            "fractal_dimension": dimension,
            "efficiency_ratio": efficiency,
            "market_type": market_type,
            "recommended_strategy": strategy,
            "persistence": "high" if hurst > 0.6 else ("low" if hurst < 0.4 else "neutral")
        }


def compute_complexity_metrics(
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None
) -> ComplexityMetrics:
    """
    Compute all complexity metrics in a single call.
    
    Convenience function for comprehensive market analysis.
    """
    if returns is None:
        if prices[0] > 0:
            returns = np.diff(np.log(prices))
        else:
            returns = np.diff(prices)
    
    entropy_calc = EntropyCalculator(max_scale=10)
    fractal = FractalDimension()
    
    shannon = entropy_calc.calculate_shannon_entropy(returns)
    sampen = entropy_calc.calculate_sample_entropy(returns)
    mse = entropy_calc.calculate_multiscale_entropy(returns)
    hurst = fractal.calculate_hurst_exponent(prices)
    fd = fractal.calculate_fractal_dimension(prices)
    efficiency = fractal.calculate_efficiency_ratio(prices)
    
    return ComplexityMetrics(
        shannon_entropy=shannon,
        sample_entropy=sampen,
        hurst_exponent=hurst,
        fractal_dimension=fd,
        multiscale_entropy=mse,
        efficiency_ratio=efficiency,
        timestamp=len(prices)
    )
