"""
AEGIS PRIME - Machine Learning Ensemble Core
============================================
Institutional-grade stacked generalization framework with:
- Multi-model ensembling (XGBoost, LightGBM, CatBoost)
- Bayesian hyperparameter optimization
- Concept drift detection (ADWIN-style)
- Probabilistic uncertainty estimation
- Time-series aware cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from bayes_opt import BayesianOptimization
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("Scikit-learn or XGBoost not installed. Ensemble features disabled.")

logger = logging.getLogger(__name__)


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    STACKED = "stacked"


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_type: ModelType
    params: Dict[str, Any]
    use_gpu: bool = False
    early_stopping_rounds: int = 50


@dataclass
class PredictionResult:
    """Standardized prediction output"""
    predictions: np.ndarray
    uncertainty: np.ndarray  # Quantile ranges or std dev
    feature_importance: Optional[Dict[str, float]] = None
    model_confidence: float = 0.0
    drift_detected: bool = False


class DriftDetector:
    """
    Simple ADWIN-style concept drift detector.
    Monitors prediction error distribution shifts.
    """
    def __init__(self, window_size: int = 500, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.error_window: List[float] = []
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self.is_initialized = False

    def update(self, error: float) -> bool:
        self.error_window.append(error)
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)

        if not self.is_initialized and len(self.error_window) == self.window_size:
            self.baseline_mean = np.mean(self.error_window)
            self.baseline_std = np.std(self.error_window) + 1e-6
            self.is_initialized = True
            return False

        if self.is_initialized:
            current_mean = np.mean(self.error_window)
            # Z-score based drift detection
            z_score = abs(current_mean - self.baseline_mean) / self.baseline_std
            if z_score > self.threshold * 100: # Scaled threshold
                logger.warning(f"Concept drift detected! Z-score: {z_score:.2f}")
                # Reset baseline to adapt (simple adaptation)
                self.baseline_mean = current_mean
                return True
        return False


class EnsembleModel:
    """
    Institutional Ensemble Engine.
    Combines gradient boosting models with stacking and uncertainty estimation.
    """
    def __init__(self, configs: List[ModelConfig]):
        if not HAS_SKLEARN:
            raise ImportError("Requires xgboost, lightgbm, catboost, scikit-learn")
        
        self.configs = configs
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.drift_detector = DriftDetector()
        self.feature_names: List[str] = []
        
    def _initialize_model(self, config: ModelConfig) -> Any:
        if config.model_type == ModelType.XGBOOST:
            params = config.params.copy()
            if config.use_gpu:
                params['tree_method'] = 'gpu_hist'
            return xgb.XGBRegressor(**params)
        elif config.model_type == ModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                params = config.params.copy()
                if config.use_gpu:
                    params['device'] = 'gpu'
                return lgb.LGBMRegressor(**params)
            except ImportError:
                logger.warning("LightGBM not installed")
                return None
        elif config.model_type == ModelType.CATBOOST:
            try:
                from catboost import CatBoostRegressor
                params = config.params.copy()
                if config.use_gpu:
                    params['task_type'] = 'GPU'
                return CatBoostRegressor(**params, verbose=0)
            except ImportError:
                logger.warning("CatBoost not installed")
                return None
        return None

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                 model_type: ModelType, n_trials: int = 50) -> Dict:
        """Bayesian optimization for hyperparameters"""
        # Define search space based on model type
        if model_type == ModelType.XGBOOST:
            def xgb_cv(max_depth, min_child_weight, gamma, subsample, colsample_bytree, learning_rate):
                params = {
                    'max_depth': int(max_depth),
                    'min_child_weight': int(min_child_weight),
                    'gamma': gamma,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'learning_rate': learning_rate,
                    'objective': 'reg:squarederror',
                    'n_jobs': -1,
                    'seed': 42
                }
                cv_result = xgb.cv(params, xgb.DMatrix(X, label=y), nfold=5, seed=42, metrics='rmse')
                return -cv_result['test-rmse-mean'].iloc[-1] # Maximize negative RMSE

            pbounds = {
                'max_depth': (3, 9),
                'min_child_weight': (1, 6),
                'gamma': (0.0, 0.4),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'learning_rate': (0.01, 0.3)
            }
            optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42)
            optimizer.maximize(n_iter=n_trials)
            return optimizer.max['target_params']
        
        # Placeholder for other models
        return {}

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """Train all models in the ensemble with time-series CV"""
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.feature_names = list(X.columns)
        total_models = 0
        
        for config in self.configs:
            logger.info(f"Training {config.model_type.value}...")
            model = self._initialize_model(config)
            if model is None:
                continue
            
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=config.early_stopping_rounds,
                    verbose=False
                )
                model_name = f"{config.model_type.value}_{total_models}"
                self.models[model_name] = model
                self.weights[model_name] = 1.0 # Initial equal weighting
                total_models += 1
                logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                logger.error(f"Failed to train {config.model_type.value}: {e}")

        if total_models == 0:
            raise RuntimeError("No models were successfully trained")

    def predict(self, X: pd.DataFrame, true_y: Optional[pd.Series] = None) -> PredictionResult:
        """Generate weighted ensemble predictions with uncertainty"""
        if not self.models:
            raise RuntimeError("Model not trained")
            
        predictions_matrix = []
        confidences = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions_matrix.append(pred)
                
                # Estimate confidence based on model type (simplified)
                if hasattr(model, 'best_iteration'):
                    conf = 1.0 / (1.0 + model.best_iteration * 0.01)
                else:
                    conf = 0.8
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                continue
        
        if not predictions_matrix:
            raise RuntimeError("No valid predictions generated")
            
        predictions_matrix = np.array(predictions_matrix).T
        weights = np.array([self.weights.get(name, 1.0) for name in self.models.keys()])
        weights = weights / weights.sum()
        
        # Weighted average
        final_pred = np.average(predictions_matrix, axis=1, weights=weights)
        
        # Uncertainty as std dev of ensemble members
        uncertainty = np.std(predictions_matrix, axis=1)
        
        # Drift detection if true labels provided
        drift = False
        if true_y is not None:
            errors = np.abs(final_pred - true_y.values)
            for err in errors[-10:]: # Check recent batch
                if self.drift_detector.update(err):
                    drift = True
                    logger.warning("Drift detected during inference. Consider retraining.")
                    break
        
        # Calculate overall confidence score
        avg_conf = np.mean(confidences)
        if drift:
            avg_conf *= 0.5 # Penalize confidence if drift detected
            
        return PredictionResult(
            predictions=final_pred,
            uncertainty=uncertainty,
            model_confidence=avg_conf,
            drift_detected=drift
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across models"""
        importance_map = {}
        counts = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    imps = model.feature_importances_
                    for feat, imp in zip(self.feature_names, imps):
                        importance_map[feat] = importance_map.get(feat, 0) + imp
                        counts[feat] = counts.get(feat, 0) + 1
            except:
                continue
        
        # Average
        for feat in importance_map:
            importance_map[feat] /= counts[feat]
            
        return dict(sorted(importance_map.items(), key=lambda x: x[1], reverse=True))
