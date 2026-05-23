"""
Regime Detection Engine
=======================

Identifies market states using Hidden Markov Models, entropy measures,
and volatility clustering to adapt strategy allocation dynamically.

Mathematical Foundation:
- Hidden Markov Models (HMM) for latent state inference
- Entropy-based regime boundary detection
- Volatility persistence modeling via GARCH processes
- Bayesian filtering for real-time state probability updates

Market States:
1. LOW_VOL_BULL: Low volatility, upward trend (risk-on)
2. HIGH_VOL_BULL: High volatility, upward trend (uncertain bullish)
3. LOW_VOL_BEAR: Low volatility, downward trend (slow distribution)
4. HIGH_VOL_BEAR: High volatility, downward trend (panic/crash)
5. TRANSITION: Regime change in progress (high uncertainty)
"""

import numpy as np
from typing import Tuple, List, Optional
from enum import Enum
from dataclasses import dataclass
import warnings

try:
    from hmmlearn import hmm
except ImportError:
    warnings.warn("hmmlearn not installed. Regime detection will use simplified heuristic.")


class MarketState(Enum):
    """Enumerated market regime states."""
    LOW_VOL_BULL = 0
    HIGH_VOL_BULL = 1
    LOW_VOL_BEAR = 2
    HIGH_VOL_BEAR = 3
    TRANSITION = 4
    
    @property
    def risk_multiplier(self) -> float:
        """Risk scaling factor based on regime."""
        mapping = {
            MarketState.LOW_VOL_BULL: 1.0,
            MarketState.HIGH_VOL_BULL: 0.7,
            MarketState.LOW_VOL_BEAR: 0.5,
            MarketState.HIGH_VOL_BEAR: 0.2,
            MarketState.TRANSITION: 0.3
        }
        return mapping[self]
    
    @property
    def description(self) -> str:
        descriptions = {
            MarketState.LOW_VOL_BULL: "Low Volatility Bull (Risk-On)",
            MarketState.HIGH_VOL_BULL: "High Volatility Bull (Uncertain)",
            MarketState.LOW_VOL_BEAR: "Low Volatility Bear (Distribution)",
            MarketState.HIGH_VOL_BEAR: "High Volatility Bear (Panic)",
            MarketState.TRANSITION: "Regime Transition (High Uncertainty)"
        }
        return descriptions[self]


@dataclass
class RegimeSignal:
    """Container for regime detection output."""
    state: MarketState
    confidence: float
    volatility: float
    trend_strength: float
    entropy: float
    timestamp: int
    state_probabilities: np.ndarray


class RegimeDetector:
    """
    Multi-method regime detection combining HMM, entropy, and volatility analysis.
    
    Features used for state classification:
    - Realized volatility (rolling window)
    - Trend strength (linear regression slope / volatility)
    - Skewness of returns
    - Kurtosis of returns
    - Autocorrelation of absolute returns (volatility persistence)
    """
    
    def __init__(
        self,
        n_states: int = 5,
        lookback_window: int = 60,
        volatility_window: int = 20,
        min_confidence: float = 0.6
    ):
        self.n_states = n_states
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.min_confidence = min_confidence
        
        # Initialize HMM if available
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.is_trained = False
        
        # State tracking
        self.current_state: Optional[MarketState] = None
        self.state_history: List[MarketState] = []
        self.confidence_history: List[float] = []
        
    def extract_features(self, returns: np.ndarray) -> np.ndarray:
        """
        Extract regime-discriminative features from return series.
        
        Feature vector: [volatility, trend_strength, skewness, kurtosis, vol_persistence]
        """
        if len(returns) < self.lookback_window:
            raise ValueError(f"Insufficient data: need {self.lookback_window} points")
        
        recent_returns = returns[-self.lookback_window:]
        
        # 1. Realized volatility
        volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        # 2. Trend strength (slope / volatility)
        x = np.arange(len(recent_returns))
        slope, _ = np.polyfit(x, np.cumsum(recent_returns), 1)
        trend_strength = slope / (volatility + 1e-8) * np.sqrt(252)
        
        # 3. Skewness
        skewness = np.mean(((recent_returns - np.mean(recent_returns)) / (np.std(recent_returns) + 1e-8)) ** 3)
        
        # 4. Kurtosis
        kurtosis = np.mean(((recent_returns - np.mean(recent_returns)) / (np.std(recent_returns) + 1e-8)) ** 4) - 3
        
        # 5. Volatility persistence (autocorrelation of absolute returns)
        abs_returns = np.abs(recent_returns)
        if len(abs_returns) > 1:
            vol_persistence = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        else:
            vol_persistence = 0.0
        
        # Handle NaN correlations
        if np.isnan(vol_persistence):
            vol_persistence = 0.0
            
        features = np.array([volatility, trend_strength, skewness, kurtosis, vol_persistence])
        
        # Normalize features
        feature_means = np.array([0.15, 0.0, 0.0, 0.0, 0.5])
        feature_stds = np.array([0.1, 1.0, 1.0, 3.0, 0.3])
        normalized = (features - feature_means) / (feature_stds + 1e-8)
        
        return normalized
    
    def train(self, returns: np.ndarray) -> 'RegimeDetector':
        """
        Train HMM on historical return series.
        
        Uses Gaussian HMM with diagonal covariance for computational efficiency.
        """
        if 'hmm' not in globals():
            print("HMM library not available. Using heuristic regime detection.")
            self.is_trained = False
            return self
        
        features = []
        for i in range(self.lookback_window, len(returns)):
            feat = self.extract_features(returns[:i])
            features.append(feat)
        
        X = np.array(features)
        
        # Initialize and train HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hmm_model.fit(X)
        
        self.is_trained = True
        return self
    
    def detect(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> RegimeSignal:
        """
        Detect current market regime from return series.
        
        If HMM is trained, uses probabilistic inference.
        Otherwise, falls back to heuristic rule-based classification.
        """
        features = self.extract_features(returns)
        volatility = features[0]
        trend_strength = features[1]
        entropy = self._calculate_entropy(returns)
        
        if self.is_trained and self.hmm_model is not None:
            # HMM-based detection
            X = features.reshape(1, -1)
            state_probs = self.hmm_model.predict_proba(X)[0]
            predicted_state_idx = np.argmax(state_probs)
            confidence = state_probs[predicted_state_idx]
            
            # Map HMM states to semantic MarketState
            state = self._map_hmm_to_market_state(predicted_state_idx, volatility, trend_strength)
        else:
            # Heuristic rule-based detection
            state, confidence = self._heuristic_classification(volatility, trend_strength, entropy)
            state_probs = self._generate_heuristic_probs(state, confidence)
        
        # Update history
        self.current_state = state
        self.state_history.append(state)
        self.confidence_history.append(confidence)
        
        return RegimeSignal(
            state=state,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            entropy=entropy,
            timestamp=len(returns),
            state_probabilities=state_probs
        )
    
    def _heuristic_classification(
        self, 
        volatility: float, 
        trend: float, 
        entropy: float
    ) -> Tuple[MarketState, float]:
        """Rule-based regime classification when HMM is unavailable."""
        
        vol_threshold = 0.20  # 20% annualized vol
        trend_threshold = 0.5
        
        # High vs Low Volatility
        is_high_vol = volatility > vol_threshold
        
        # Bull vs Bear
        is_bull = trend > trend_threshold
        is_bear = trend < -trend_threshold
        
        # Transition detection via entropy
        is_transition = entropy > 0.8
        
        if is_transition:
            return MarketState.TRANSITION, min(0.9, entropy)
        
        if is_high_vol:
            if is_bull:
                return MarketState.HIGH_VOL_BULL, 0.75
            elif is_bear:
                return MarketState.HIGH_VOL_BEAR, 0.75
            else:
                # Choppy high vol
                return MarketState.TRANSITION, 0.65
        else:
            if is_bull:
                return MarketState.LOW_VOL_BULL, 0.85
            elif is_bear:
                return MarketState.LOW_VOL_BEAR, 0.85
            else:
                # Neutral low vol
                return MarketState.LOW_VOL_BULL, 0.60
    
    def _map_hmm_to_market_state(self, hmm_idx: int, vol: float, trend: float) -> MarketState:
        """Map HMM internal state index to semantic MarketState."""
        # Simple mapping based on volatility and trend characteristics
        if vol > 0.25:
            return MarketState.HIGH_VOL_BULL if trend > 0 else MarketState.HIGH_VOL_BEAR
        elif vol < 0.10:
            return MarketState.LOW_VOL_BULL if trend > 0 else MarketState.LOW_VOL_BEAR
        else:
            return MarketState.TRANSITION
    
    def _generate_heuristic_probs(self, state: MarketState, confidence: float) -> np.ndarray:
        """Generate probability distribution over states for heuristic method."""
        probs = np.ones(self.n_states) * (1 - confidence) / (self.n_states - 1)
        probs[state.value] = confidence
        return probs
    
    def _calculate_entropy(self, returns: np.ndarray) -> float:
        """Calculate Shannon entropy of return distribution."""
        # Discretize returns into bins
        hist, _ = np.histogram(returns, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        probs = hist / np.sum(hist)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(hist))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        return min(1.0, normalized_entropy)
    
    def get_regime_duration(self) -> int:
        """Calculate how long current regime has persisted."""
        if not self.state_history:
            return 0
        
        current = self.state_history[-1]
        duration = 0
        for state in reversed(self.state_history):
            if state == current:
                duration += 1
            else:
                break
        return duration
    
    def is_regime_change_imminent(self, threshold: float = 0.4) -> bool:
        """Detect if regime change is likely based on probability shifts."""
        if len(self.confidence_history) < 5:
            return False
        
        recent_confidence = np.mean(self.confidence_history[-5:])
        return recent_confidence < threshold
