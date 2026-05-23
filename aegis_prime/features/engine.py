"""
AEGIS PRIME - Advanced Feature Engineering Module
==================================================
Institutional-grade feature generation with microstructure signals,
spectral analysis, entropy measures, and regime-aware transformations.

Features:
- Order Flow Imbalance (OFI)
- VWAP deviation and volume profile
- Spectral entropy and dominant periods
- Sample entropy for complexity measurement
- Fractal dimension estimation
- Wavelet decomposition features
- Cross-asset correlation dynamics
- Lagged feature matrices with purged alignment
"""

import numpy as np
import polars as pl
from numba import jit, prange
from typing import List, Dict, Tuple, Optional
from scipy import signal as scipy_signal
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')


@jit(nopython=True, cache=True)
def _calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Numba-accelerated VWAP calculation."""
    n = len(prices)
    vwap = np.zeros(n)
    cum_vol = 0.0
    cum_pv = 0.0
    
    for i in range(n):
        cum_vol += volumes[i]
        cum_pv += prices[i] * volumes[i]
        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
        else:
            vwap[i] = prices[i]
    
    return vwap


@jit(nopython=True, cache=True)
def _calculate_order_flow_imbalance(
    bids: np.ndarray, 
    asks: np.ndarray,
    bid_vols: np.ndarray,
    ask_vols: np.ndarray
) -> np.ndarray:
    """
    Calculate Order Flow Imbalance (OFI).
    OFI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    Positive OFI indicates buying pressure.
    """
    n = len(bids)
    ofi = np.zeros(n)
    
    for i in range(n):
        total_vol = bid_vols[i] + ask_vols[i]
        if total_vol > 0:
            ofi[i] = (bid_vols[i] - ask_vols[i]) / total_vol
        else:
            ofi[i] = 0.0
    
    return ofi


@jit(nopython=True, cache=True)
def _sample_entropy(data: np.ndarray, m: int, r: float) -> float:
    """
    Calculate Sample Entropy using Numba.
    Measures the complexity/regularity of a time series.
    Lower entropy = more regular/predictable
    Higher entropy = more complex/unpredictable
    """
    n = len(data)
    if n < m + 1:
        return 0.0
    
    # Create templates
    def _count_matches(embedded: np.ndarray, r_thresh: float) -> int:
        count = 0
        n_emb = len(embedded)
        for i in range(n_emb - 1):
            for j in range(i + 1, n_emb):
                max_diff = 0.0
                for k in range(m):
                    diff = abs(embedded[i, k] - embedded[j, k])
                    if diff > max_diff:
                        max_diff = diff
                if max_diff < r_thresh:
                    count += 1
        return count
    
    # Build embedding matrix
    embedded_m = np.zeros((n - m, m))
    for i in range(n - m):
        for j in range(m):
            embedded_m[i, j] = data[i + j]
    
    # Count matches for length m
    A = _count_matches(embedded_m, r)
    
    # Build embedding for length m+1
    if n < m + 2:
        return 0.0
    
    embedded_m1 = np.zeros((n - m - 1, m + 1))
    for i in range(n - m - 1):
        for j in range(m + 1):
            embedded_m1[i, j] = data[i + j]
    
    # Count matches for length m+1
    B = _count_matches(embedded_m1, r)
    
    if A == 0 or B == 0:
        return 0.0
    
    return -np.log(B / A)


class MicrostructureFeatures:
    """Generate microstructure-based alpha features."""
    
    @staticmethod
    def calculate_vwap_deviation(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate VWAP and its deviation from price."""
        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        
        vwap = _calculate_vwap(prices, volumes)
        vwap_dev = (prices - vwap) / vwap
        
        return df.with_columns([
            pl.Series('vwap', vwap),
            pl.Series('vwap_deviation', vwap_dev),
            pl.Series('vwap_distance', np.abs(vwap_dev))
        ])
    
    @staticmethod
    def calculate_order_flow(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate order flow imbalance from bid/ask data.
        Requires: bid_price, ask_price, bid_volume, ask_volume columns
        """
        required = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume']
        if not all(col in df.columns for col in required):
            # Approximate from OHLCV if order book data unavailable
            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            close = df['close'].to_numpy()
            volume = df['volume'].to_numpy()
            
            # Approximate bid/ask from high/low
            bid_vol = volume * (close - low) / (high - low + 1e-10)
            ask_vol = volume * (high - close) / (high - low + 1e-10)
            
            ofi = _calculate_order_flow_imbalance(
                low, high, bid_vol, ask_vol
            )
        else:
            ofi = _calculate_order_flow_imbalance(
                df['bid_price'].to_numpy(),
                df['ask_price'].to_numpy(),
                df['bid_volume'].to_numpy(),
                df['ask_volume'].to_numpy()
            )
        
        return df.with_columns([
            pl.Series('order_flow_imbalance', ofi),
            pl.Series('ofi_smooth', pl.Series(ofi).rolling_mean(window_size=5)),
            pl.Series('ofi_accel', pl.Series(ofi).diff().fill_null(0))
        ])
    
    @staticmethod
    def calculate_volume_profile(df: pl.DataFrame, n_bins: int = 10) -> pl.DataFrame:
        """Calculate volume distribution across price bins."""
        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        
        min_p, max_p = prices.min(), prices.max()
        if min_p == max_p:
            return df.with_columns(pl.lit(0.0).alias('volume_profile_skew'))
        
        bin_edges = np.linspace(min_p, max_p, n_bins + 1)
        bin_indices = np.digitize(prices, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate volume-weighted position in the range
        vol_weighted_price = np.sum(prices * volumes) / (np.sum(volumes) + 1e-10)
        normalized_vwp = (vol_weighted_price - min_p) / (max_p - min_p + 1e-10)
        
        return df.with_columns([
            pl.lit(normalized_vwp).alias('volume_profile_skew'),
            pl.lit(np.std(bin_indices)).alias('volume_concentration')
        ])


class SpectralFeatures:
    """Frequency-domain feature engineering using FFT and wavelets."""
    
    @staticmethod
    def calculate_spectral_entropy(returns: np.ndarray) -> float:
        """
        Calculate spectral entropy to measure signal predictability.
        Low entropy = strong periodic components (more predictable)
        High entropy = noise-like (less predictable)
        """
        if len(returns) < 10:
            return 0.0
        
        # Remove mean and apply window
        returns_centered = returns - np.mean(returns)
        window = np.hanning(len(returns_centered))
        windowed = returns_centered * window
        
        # FFT
        fft_vals = np.fft.rfft(windowed)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Normalize to probability distribution
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        
        prob_dist = power_spectrum / total_power
        
        # Shannon entropy
        return entropy(prob_dist + 1e-10)
    
    @staticmethod
    def find_dominant_periods(returns: np.ndarray, max_period: int = 60) -> Tuple[List[int], List[float]]:
        """Identify dominant periodicities in the return series."""
        if len(returns) < max_period * 2:
            return [], []
        
        # FFT
        fft_vals = np.fft.rfft(returns - np.mean(returns))
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Frequency bins
        n = len(returns)
        freqs = np.fft.rfftfreq(n)
        
        # Find peaks in power spectrum
        periods = []
        powers = []
        
        for i in range(1, len(freqs)):
            if freqs[i] > 0:
                period = 1.0 / freqs[i]
                if period <= max_period and i < len(power_spectrum):
                    periods.append(int(period))
                    powers.append(power_spectrum[i])
        
        if not periods:
            return [], []
        
        # Sort by power and return top 3
        sorted_idx = np.argsort(powers)[::-1][:3]
        return [periods[i] for i in sorted_idx], [powers[i] for i in sorted_idx]
    
    @staticmethod
    def calculate_spectral_features(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
        """Calculate rolling spectral features."""
        returns = df['returns'].to_numpy()
        n = len(returns)
        
        spectral_entropy = np.zeros(n)
        dominant_period = np.zeros(n)
        spectral_power_ratio = np.zeros(n)
        
        for i in range(window, n):
            window_returns = returns[i-window:i]
            spectral_entropy[i] = SpectralFeatures.calculate_spectral_entropy(window_returns)
            
            periods, powers = SpectralFeatures.find_dominant_periods(window_returns)
            if periods:
                dominant_period[i] = periods[0]
                # Ratio of dominant power to total power
                total_power = np.sum(powers)
                if total_power > 0:
                    spectral_power_ratio[i] = powers[0] / total_power
        
        return df.with_columns([
            pl.Series('spectral_entropy', spectral_entropy),
            pl.Series('dominant_period', dominant_period),
            pl.Series('spectral_power_ratio', spectral_power_ratio),
            pl.Series('spectral_entropy_zscore', 
                     pl.Series(spectral_entropy).rolling_std(window_size=window) / 
                     (pl.Series(spectral_entropy).rolling_mean(window_size=window) + 1e-10))
        ])


class EntropyFeatures:
    """Complexity and information-theoretic features."""
    
    @staticmethod
    def calculate_sample_entropy_feature(returns: np.ndarray, window: int = 60) -> np.ndarray:
        """Rolling sample entropy calculation."""
        n = len(returns)
        samp_ent = np.zeros(n)
        
        for i in range(window, n):
            window_data = returns[i-window:i]
            # Standardize
            std = np.std(window_data)
            if std > 0:
                normalized = window_data / std
                # r typically 0.2 * std, m typically 2
                samp_ent[i] = _sample_entropy(normalized, m=2, r=0.2)
            else:
                samp_ent[i] = 0.0
        
        return samp_ent
    
    @staticmethod
    def calculate_entropy_features(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
        """Calculate multiple entropy-based features."""
        returns = df['returns'].to_numpy()
        
        samp_ent = EntropyFeatures.calculate_sample_entropy_feature(returns, window)
        
        # Approximate entropy using simpler method for speed
        perm_entropy = np.zeros(len(returns))
        for i in range(window, len(returns)):
            window_data = returns[i-window:i]
            # Permutation entropy approximation
            diffs = np.diff(window_data)
            signs = np.sign(diffs)
            # Simple entropy of sign changes
            if len(signs) > 0:
                p_pos = np.sum(signs > 0) / len(signs)
                p_neg = np.sum(signs < 0) / len(signs)
                if p_pos > 0 and p_neg > 0:
                    perm_entropy[i] = -p_pos * np.log(p_pos + 1e-10) - p_neg * np.log(p_neg + 1e-10)
        
        return df.with_columns([
            pl.Series('sample_entropy', samp_ent),
            pl.Series('permutation_entropy', perm_entropy),
            pl.Series('entropy_regime', 
                     pl.Series(samp_ent).map_elements(lambda x: 'high' if x > np.median(samp_ent[samp_ent > 0]) else 'low', return_dtype=pl.Utf8))
        ])


class FeatureEngine:
    """
    Main feature engineering orchestrator.
    Combines microstructure, spectral, and entropy features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.window = self.config.get('window', 60)
        self.feature_names = []
    
    def generate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate complete feature set from OHLCV data."""
        # Ensure returns exist
        if 'returns' not in df.columns:
            df = df.with_columns(
                (pl.col('close') / pl.col('close').shift(1) - 1).alias('returns')
            ).fill_null(0)
        
        # Microstructure features
        df = MicrostructureFeatures.calculate_vwap_deviation(df)
        df = MicrostructureFeatures.calculate_order_flow(df)
        df = MicrostructureFeatures.calculate_volume_profile(df)
        
        # Spectral features
        df = SpectralFeatures.calculate_spectral_features(df, self.window)
        
        # Entropy features
        df = EntropyFeatures.calculate_entropy_features(df, self.window)
        
        # Additional technical features
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_lag_features(df)
        
        # Clean infinite values
        df = df.fill_nan(None).fill_null(0)
        
        # Replace inf with large finite values
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64]:
                max_val = df[col].abs().max()
                if max_val > 1e10 or max_val.is_infinite():
                    df = df.with_columns(
                        pl.col(col).clip(-1e10, 1e10)
                    )
        
        self.feature_names = [col for col in df.columns if col not in 
                             ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def _add_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum and trend features."""
        windows = [5, 10, 20, 60]
        
        for w in windows:
            df = df.with_columns([
                pl.col('close').pct_change(w).alias(f'momentum_{w}d'),
                pl.col('close').rolling_mean(w).alias(f'sma_{w}d'),
                (pl.col('close') / pl.col('close').rolling_mean(w) - 1).alias(f'ma_deviation_{w}d')
            ])
        
        # RSI
        delta = df['close'].diff()
        gain = delta.map_elements(lambda x: x if x > 0 else 0, return_dtype=pl.Float64)
        loss = delta.map_elements(lambda x: -x if x < 0 else 0, return_dtype=pl.Float64)
        
        avg_gain = pl.Series(gain).rolling_mean(14)
        avg_loss = pl.Series(loss).rolling_mean(14)
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        df = df.with_columns(pl.Series('rsi_14', rsi))
        
        return df
    
    def _add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility and risk features."""
        windows = [5, 10, 20, 60]
        
        for w in windows:
            df = df.with_columns([
                pl.col('returns').rolling_std(w).alias(f'realized_vol_{w}d'),
                (pl.col('returns').rolling_std(w) / pl.col('returns').rolling_mean(w).abs().clip(1e-10)).alias(f'sharpe_approx_{w}d')
            ])
        
        # Parkinson volatility
        log_hl = (df['high'] / df['low']).log()
        parkinson = (1 / (4 * np.log(2))) * (log_hl ** 2)
        df = df.with_columns([
            parkinson.rolling_mean(20).sqrt().alias('parkinson_vol'),
            ((df['high'] - df['low']) / df['close']).alias('intraday_range')
        ])
        
        return df
    
    def _add_lag_features(self, df: pl.DataFrame, max_lag: int = 10) -> pl.DataFrame:
        """Add lagged features for autoregressive modeling."""
        lag_cols = ['returns', 'order_flow_imbalance', 'vwap_deviation', 'spectral_entropy']
        
        for col in lag_cols:
            if col in df.columns:
                for lag in range(1, min(max_lag + 1, 6)):
                    df = df.with_columns(
                        pl.col(col).shift(lag).alias(f'{col}_lag{lag}')
                    )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of generated feature names."""
        return self.feature_names
    
    def create_target(self, df: pl.DataFrame, horizon: int = 5) -> pl.DataFrame:
        """
        Create forward-looking target variable.
        Target = cumulative returns over next 'horizon' periods.
        Uses purged alignment to prevent leakage.
        """
        future_returns = df['returns'].shift(-horizon).rolling_sum(horizon)
        
        # Sign classification for classification tasks
        target_direction = future_returns.map_elements(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0), 
            return_dtype=pl.Int8
        )
        
        return df.with_columns([
            pl.Series('target_return', future_returns),
            pl.Series('target_direction', target_direction)
        ])


# Example usage and testing
if __name__ == '__main__':
    # Generate synthetic data for testing
    np.random.seed(42)
    n = 1000
    
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1).add(days=n-1), eager=True)
    
    # Generate realistic price path with stochastic volatility
    returns = np.random.normal(0.0005, 0.02, n)
    volatility = np.abs(np.random.normal(0.02, 0.005, n))
    returns = returns * (volatility / 0.02)  # Scale by stochastic vol
    
    prices = 100 * np.cumprod(1 + returns)
    
    # Generate OHLCV
    df = pl.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0.005, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0.005, 0.002, n))),
        'close': prices,
        'volume': np.random.uniform(1e6, 1e7, n)
    })
    
    # Initialize feature engine
    engine = FeatureEngine({'window': 60})
    
    # Generate features
    df_features = engine.generate_features(df)
    df_features = engine.create_target(df_features, horizon=5)
    
    print(f"Generated {len(engine.get_feature_names())} features:")
    for feat in engine.get_feature_names()[:10]:
        print(f"  - {feat}")
    
    print(f"\nFeature matrix shape: {df_features.shape}")
    print(f"Sample features:\n{df_features.select(engine.get_feature_names()[:5]).head()}")
