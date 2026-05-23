"""
core/indicators.py
Pure-Python technical indicator library.
No TA-Lib dependency — works in any environment.
"""
from __future__ import annotations
import math
from typing import List, Optional


def ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Exponential Moving Average."""
    k = 2.0 / (period + 1)
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return result
    result[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)  # type: ignore[operator]
    return result


def sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Simple Moving Average."""
    result: List[Optional[float]] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1 : i + 1]) / period
    return result


def rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index (Wilder smoothing)."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    result[period] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss or 1e-9))
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
        result[i] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss or 1e-9))
    return result


def macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """
    MACD indicator.
    Returns dict with keys: macd_line, signal_line, histogram.
    """
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line: List[Optional[float]] = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    _fill = [v if v is not None else 0.0 for v in macd_line]
    signal_line = ema(_fill, signal)
    histogram: List[Optional[float]] = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram}


def bollinger_bands(
    closes: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict:
    """
    Bollinger Bands.
    Returns dict with keys: upper, middle, lower.
    """
    upper: List[Optional[float]] = [None] * len(closes)
    middle: List[Optional[float]] = [None] * len(closes)
    lower: List[Optional[float]] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        avg = sum(window) / period
        std = math.sqrt(sum((x - avg) ** 2 for x in window) / period)
        middle[i] = avg
        upper[i] = avg + std_dev * std
        lower[i] = avg - std_dev * std
    return {"upper": upper, "middle": middle, "lower": lower}


def atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[Optional[float]]:
    """Average True Range."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < 2:
        return result
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return result
    atr_val = sum(trs[:period]) / period
    result[period] = atr_val
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        result[i + 1] = atr_val
    return result


def stochastic(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    k_period: int = 14,
    d_period: int = 3,
) -> dict:
    """Stochastic Oscillator (%K and %D)."""
    k_vals: List[Optional[float]] = [None] * len(closes)
    for i in range(k_period - 1, len(closes)):
        high_max = max(highs[i - k_period + 1 : i + 1])
        low_min = min(lows[i - k_period + 1 : i + 1])
        denom = high_max - low_min
        k_vals[i] = 100.0 * (closes[i] - low_min) / (denom or 1e-9)
    _fill_k = [v if v is not None else 0.0 for v in k_vals]
    d_vals = sma(_fill_k, d_period)
    return {"k": k_vals, "d": d_vals}


def vwap(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
) -> List[float]:
    """Volume-Weighted Average Price (cumulative, resets each session)."""
    result = []
    cum_pv, cum_v = 0.0, 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical = (h + l + c) / 3.0
        cum_pv += typical * v
        cum_v += v
        result.append(cum_pv / cum_v if cum_v else c)
    return result


def enrich_dataframe(candles: List[dict]) -> List[dict]:
    """
    Attach all indicators to a list of OHLCV dicts.
    Each dict must have: open, high, low, close, volume.
    Returns enriched list (new dicts, original untouched).
    """
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)
    e200 = ema(closes, 200)
    rs = rsi(closes, 14)
    mc = macd(closes, 12, 26, 9)
    bb = bollinger_bands(closes, 20, 2.0)
    at = atr(highs, lows, closes, 14)
    sto = stochastic(highs, lows, closes, 14, 3)
    vw = vwap(highs, lows, closes, volumes)

    enriched = []
    for i, c in enumerate(candles):
        enriched.append({
            **c,
            "ema9": e9[i],
            "ema21": e21[i],
            "ema50": e50[i],
            "ema200": e200[i],
            "rsi": rs[i],
            "macd": mc["macd_line"][i],
            "macd_signal": mc["signal_line"][i],
            "macd_hist": mc["histogram"][i],
            "bb_upper": bb["upper"][i],
            "bb_mid": bb["middle"][i],
            "bb_lower": bb["lower"][i],
            "atr": at[i],
            "stoch_k": sto["k"][i],
            "stoch_d": sto["d"][i],
            "vwap": vw[i],
        })
    return enriched
