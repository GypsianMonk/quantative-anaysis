"""
core/scanner.py
Multi-pair real-time signal scanner.
Fetches OHLCV data via ccxt and emits structured signal objects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from core.indicators import enrich_dataframe


@dataclass
class Signal:
    pair: str
    timeframe: str
    timestamp: datetime
    signal: str           # BUY | SELL | HOLD | OVERSOLD | OVERBOUGHT
    strength: int         # 0 = weak, 1 = moderate, 2 = strong
    price: float
    rsi: Optional[float]
    macd_dir: str         # "↑" | "↓" | "–"
    bb_position: str      # "above_upper" | "below_lower" | "middle"
    ema_trend: str        # "bullish" | "bearish" | "mixed"
    reason: str


class Scanner:
    """
    Scans multiple pairs on a given timeframe and produces Signal objects.

    Parameters
    ----------
    strategy_cfg : dict
        Must contain: rsi_oversold, rsi_overbought,
                      macd_fast, macd_slow, macd_signal,
                      bb_period, bb_std
    """

    def __init__(self, strategy_cfg: Optional[dict] = None):
        self.cfg = strategy_cfg or {
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
        }

    def scan_candles(self, pair: str, timeframe: str, candles: List[dict]) -> Signal:
        """
        Evaluate enriched candles and return latest Signal.

        Parameters
        ----------
        pair      : e.g. "BTC/USDT"
        timeframe : e.g. "4h"
        candles   : list of OHLCV dicts with keys open/high/low/close/volume
        """
        data = enrich_dataframe(candles)
        if len(data) < 2:
            return self._neutral(pair, timeframe, 0)

        curr = data[-1]
        prev = data[-2]

        rsi_val = curr.get("rsi")
        macd_val = curr.get("macd")
        macd_sig = curr.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_sig = prev.get("macd_signal")
        close = curr["close"]
        bb_upper = curr.get("bb_upper")
        bb_lower = curr.get("bb_lower")
        ema9 = curr.get("ema9")
        ema21 = curr.get("ema21")
        ema50 = curr.get("ema50")

        if any(v is None for v in [rsi_val, macd_val, macd_sig, prev_macd, prev_sig]):
            return self._neutral(pair, timeframe, close)

        # MACD crossover detection
        macd_cross_up = prev_macd < prev_sig and macd_val > macd_sig  # type: ignore
        macd_cross_dn = prev_macd > prev_sig and macd_val < macd_sig  # type: ignore
        macd_dir = "↑" if macd_val > macd_sig else "↓"  # type: ignore

        # BB position
        if bb_upper and close >= bb_upper * 0.99:
            bb_pos = "above_upper"
        elif bb_lower and close <= bb_lower * 1.01:
            bb_pos = "below_lower"
        else:
            bb_pos = "middle"

        # EMA trend
        if ema9 and ema21 and ema50:
            if ema9 > ema21 > ema50:
                ema_trend = "bullish"
            elif ema9 < ema21 < ema50:
                ema_trend = "bearish"
            else:
                ema_trend = "mixed"
        else:
            ema_trend = "mixed"

        # Signal logic
        os_thresh = self.cfg["rsi_oversold"]
        ob_thresh = self.cfg["rsi_overbought"]

        if macd_cross_up and rsi_val < os_thresh:  # type: ignore
            strength = 2 if bb_pos == "below_lower" and ema_trend != "bearish" else 1
            return Signal(
                pair=pair, timeframe=timeframe, timestamp=datetime.utcnow(),
                signal="BUY", strength=strength, price=close, rsi=round(rsi_val, 1),
                macd_dir=macd_dir, bb_position=bb_pos, ema_trend=ema_trend,
                reason=f"RSI {rsi_val:.1f} oversold · MACD cross↑",
            )

        if macd_cross_dn and rsi_val > ob_thresh:  # type: ignore
            strength = 2 if bb_pos == "above_upper" and ema_trend != "bullish" else 1
            return Signal(
                pair=pair, timeframe=timeframe, timestamp=datetime.utcnow(),
                signal="SELL", strength=strength, price=close, rsi=round(rsi_val, 1),
                macd_dir=macd_dir, bb_position=bb_pos, ema_trend=ema_trend,
                reason=f"RSI {rsi_val:.1f} overbought · MACD cross↓",
            )

        if rsi_val < os_thresh:  # type: ignore
            return Signal(
                pair=pair, timeframe=timeframe, timestamp=datetime.utcnow(),
                signal="OVERSOLD", strength=1, price=close, rsi=round(rsi_val, 1),
                macd_dir=macd_dir, bb_position=bb_pos, ema_trend=ema_trend,
                reason=f"RSI {rsi_val:.1f} — watching for MACD confirmation",
            )

        if rsi_val > ob_thresh:  # type: ignore
            return Signal(
                pair=pair, timeframe=timeframe, timestamp=datetime.utcnow(),
                signal="OVERBOUGHT", strength=1, price=close, rsi=round(rsi_val, 1),
                macd_dir=macd_dir, bb_position=bb_pos, ema_trend=ema_trend,
                reason=f"RSI {rsi_val:.1f} — watching for MACD confirmation",
            )

        return self._neutral(pair, timeframe, close, rsi_val, macd_dir, bb_pos, ema_trend)

    def scan_exchange(self, exchange, pairs: List[str], timeframe: str = "4h") -> List[Signal]:
        """
        Scan live data from a ccxt exchange instance.

        Usage
        -----
        import ccxt
        ex = ccxt.kucoin()
        scanner = Scanner()
        signals = scanner.scan_exchange(ex, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        """
        results = []
        for pair in pairs:
            try:
                raw = exchange.fetch_ohlcv(pair, timeframe, limit=250)
                candles = [
                    {"open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                    for r in raw
                ]
                sig = self.scan_candles(pair, timeframe, candles)
                results.append(sig)
            except Exception as e:
                results.append(self._neutral(pair, timeframe, 0, reason=str(e)))
        return results

    @staticmethod
    def _neutral(
        pair: str,
        timeframe: str,
        price: float,
        rsi: Optional[float] = None,
        macd_dir: str = "–",
        bb_pos: str = "middle",
        ema_trend: str = "mixed",
        reason: str = "No signal",
    ) -> Signal:
        return Signal(
            pair=pair, timeframe=timeframe, timestamp=datetime.utcnow(),
            signal="HOLD", strength=0, price=price, rsi=rsi,
            macd_dir=macd_dir, bb_position=bb_pos, ema_trend=ema_trend,
            reason=reason,
        )
