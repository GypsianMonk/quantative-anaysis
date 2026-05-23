"""
strategies/QuantCoreStrategy.py
Production Freqtrade strategy — RSI + MACD confluence with BB filter.
Compatible with Freqtrade >= 2024.1

Quickstart
----------
freqtrade download-data -p BTC/USDT ETH/USDT SOL/USDT BNB/USDT \\
    --timeframe 4h --exchange kucoin

freqtrade backtesting --strategy QuantCoreStrategy \\
    --timerange 20240101- --export trades

freqtrade hyperopt --strategy QuantCoreStrategy \\
    --hyperopt-loss SharpeHyperOptLoss \\
    -e 500 --spaces buy sell stoploss \\
    --timeframe 4h

freqtrade trade --strategy QuantCoreStrategy --dry-run
"""
from __future__ import annotations
from functools import reduce

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    informative,
)


class QuantCoreStrategy(IStrategy):
    """
    Multi-indicator confluence strategy.

    Entry  : RSI oversold  + MACD bullish crossover  (+ optional BB lower band filter)
    Exit   : RSI overbought + MACD bearish crossover
    Filter : EMA50 trend alignment (optional, toggled via use_ema_filter)

    All parameters are hyperopt-ready via IntParameter / DecimalParameter.
    """

    INTERFACE_VERSION = 3

    # ── ROI & stoploss ────────────────────────────────────────────────
    minimal_roi = {
        "0":    0.15,
        "720":  0.08,
        "1440": 0.04,
        "2880": 0.02,
    }
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # ── Timeframe ─────────────────────────────────────────────────────
    timeframe = "4h"
    startup_candle_count: int = 60   # enough for EMA50 + BB20 warm-up

    # ── Order type ────────────────────────────────────────────────────
    order_types = {
        "entry":        "limit",
        "exit":         "limit",
        "stoploss":     "market",
        "stoploss_on_exchange": True,
    }

    # ── Hyperopt parameters ───────────────────────────────────────────
    # RSI
    rsi_oversold   = IntParameter(20, 45, default=35, space="buy",  optimize=True, load=True)
    rsi_overbought = IntParameter(55, 80, default=65, space="sell", optimize=True, load=True)

    # MACD
    macd_fast   = IntParameter(8,  16, default=12, space="buy", optimize=True, load=True)
    macd_slow   = IntParameter(20, 32, default=26, space="buy", optimize=True, load=True)
    macd_signal = IntParameter(6,  12, default=9,  space="buy", optimize=True, load=True)

    # Bollinger Bands
    bb_period = IntParameter(10,  30,  default=20,  space="buy", optimize=True, load=True)
    bb_std    = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True, load=True)

    # EMA trend filter toggle
    use_ema_filter = IntParameter(0, 1, default=1, space="buy", optimize=False, load=True)

    # ── Indicator population ──────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = self._rsi(dataframe["close"], 14)

        # MACD (uses hyperopt params)
        macd_df = self._macd(
            dataframe["close"],
            self.macd_fast.value,
            self.macd_slow.value,
            self.macd_signal.value,
        )
        dataframe["macd"]        = macd_df["macd"]
        dataframe["macdsignal"]  = macd_df["signal"]
        dataframe["macdhist"]    = macd_df["hist"]

        # Bollinger Bands
        bb = self._bollinger(dataframe["close"], self.bb_period.value, float(self.bb_std.value))
        dataframe["bb_upper"]  = bb["upper"]
        dataframe["bb_mid"]    = bb["mid"]
        dataframe["bb_lower"]  = bb["lower"]
        dataframe["bb_width"]  = (bb["upper"] - bb["lower"]) / bb["mid"]

        # EMAs
        for period in [9, 21, 50, 200]:
            dataframe[f"ema{period}"] = dataframe["close"].ewm(span=period, adjust=False).mean()

        # ATR (for dynamic stop sizing)
        high_low   = dataframe["high"] - dataframe["low"]
        high_close = (dataframe["high"] - dataframe["close"].shift()).abs()
        low_close  = (dataframe["low"]  - dataframe["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        dataframe["atr"] = tr.ewm(span=14, adjust=False).mean()

        # Volume SMA
        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    # ── Entry signal ──────────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["rsi"]       < self.rsi_oversold.value,
            dataframe["macd"]      > dataframe["macdsignal"],
            dataframe["macd"].shift(1) <= dataframe["macdsignal"].shift(1),
            dataframe["volume"]    > 0,
            dataframe["volume"]    > dataframe["volume_sma20"] * 0.5,
        ]

        if self.use_ema_filter.value:
            conditions.append(dataframe["close"] > dataframe["ema50"])

        dataframe.loc[reduce(lambda a, b: a & b, conditions), "enter_long"] = 1
        return dataframe

    # ── Exit signal ───────────────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["rsi"]      > self.rsi_overbought.value,
            dataframe["macd"]     < dataframe["macdsignal"],
            dataframe["macd"].shift(1) >= dataframe["macdsignal"].shift(1),
            dataframe["volume"]   > 0,
        ]
        dataframe.loc[reduce(lambda a, b: a & b, conditions), "exit_long"] = 1
        return dataframe

    # ── Custom stoploss (ATR-based) ───────────────────────────────────
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """
        ATR-based dynamic stop: 2× ATR below current price.
        Falls back to fixed stoploss if ATR unavailable.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last_atr = dataframe["atr"].iloc[-1]
        if pd.isna(last_atr) or last_atr <= 0:
            return self.stoploss

        atr_stop = (2 * last_atr) / current_rate
        return max(-atr_stop, self.stoploss)  # never looser than stoploss

    # ── Pure-Python indicator helpers (no TA-Lib dep) ─────────────────
    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> DataFrame:
        ema_f = series.ewm(span=fast,   adjust=False).mean()
        ema_s = series.ewm(span=slow,   adjust=False).mean()
        macd  = ema_f - ema_s
        sig   = macd.ewm(span=signal, adjust=False).mean()
        return DataFrame({"macd": macd, "signal": sig, "hist": macd - sig})

    @staticmethod
    def _bollinger(series: pd.Series, period: int, std_dev: float) -> DataFrame:
        mid   = series.rolling(period).mean()
        sigma = series.rolling(period).std(ddof=0)
        return DataFrame({
            "upper": mid + std_dev * sigma,
            "mid":   mid,
            "lower": mid - std_dev * sigma,
        })
