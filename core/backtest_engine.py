"""
core/backtest_engine.py
Event-driven backtesting engine.
Supports: long/short, stop-loss, take-profit, trailing stop,
          commission, and slippage modelling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from core.indicators import enrich_dataframe
from core.risk_manager import RiskManager, TradeRecord


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.02
    commission: float = 0.001        # 0.1% per side (KuCoin taker)
    slippage: float = 0.0005         # 0.05% market order slippage
    stoploss: float = 0.08           # 8% max loss per trade
    trailing_stop: bool = True
    trailing_stop_offset: float = 0.03   # activate at +3%
    trailing_stop_value: float = 0.015   # trail by 1.5%
    max_open_trades: int = 1
    timeframe_hours: int = 4


@dataclass
class OpenTrade:
    entry_price: float
    size: float
    stop_price: float
    entry_idx: int
    highest_since_entry: float = 0.0
    trailing_active: bool = False


@dataclass
class ClosedTrade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    won: bool


def _apply_costs(price: float, direction: str, commission: float, slippage: float) -> float:
    cost = commission + slippage
    if direction == "buy":
        return price * (1 + cost)
    return price * (1 - cost)


def run_backtest(
    candles: List[dict],
    entry_signal: Callable[[List[dict], int], bool],
    exit_signal: Callable[[List[dict], int], bool],
    cfg: Optional[BacktestConfig] = None,
) -> dict:
    """
    Run a full backtest over enriched candle data.

    Parameters
    ----------
    candles      : list of OHLCV dicts (will be enriched automatically)
    entry_signal : fn(enriched_candles, index) -> bool
    exit_signal  : fn(enriched_candles, index) -> bool
    cfg          : BacktestConfig instance

    Returns
    -------
    dict with keys: trades, equity_curve, stats, config
    """
    if cfg is None:
        cfg = BacktestConfig()

    data = enrich_dataframe(candles)
    rm = RiskManager(
        balance=cfg.initial_balance,
        risk_per_trade=cfg.risk_per_trade,
    )

    balance = cfg.initial_balance
    position: Optional[OpenTrade] = None
    closed_trades: List[ClosedTrade] = []
    equity_curve: List[float] = [balance]

    for i in range(50, len(data)):   # warm-up period for indicators
        candle = data[i]
        close = candle["close"]
        atr_val = candle.get("atr") or close * 0.02

        # ── Manage open position ──────────────────────────────────────
        if position is not None:
            position.highest_since_entry = max(position.highest_since_entry, close)

            # Activate trailing stop
            if (not position.trailing_active and
                    close >= position.entry_price * (1 + cfg.trailing_stop_offset)):
                position.trailing_active = True

            # Update trailing stop price
            if position.trailing_active:
                trail_price = position.highest_since_entry * (1 - cfg.trailing_stop_value)
                position.stop_price = max(position.stop_price, trail_price)

            # Check stop-loss / trailing-stop breach
            exit_reason = None
            if close <= position.stop_price:
                exit_reason = "stop_loss" if not position.trailing_active else "trailing_stop"

            # Check strategy exit signal
            if exit_reason is None and exit_signal(data, i):
                exit_reason = "signal"

            if exit_reason:
                exit_px = _apply_costs(close, "sell", cfg.commission, cfg.slippage)
                pnl = (exit_px - position.entry_price) * position.size
                pnl_pct = (exit_px - position.entry_price) / position.entry_price * 100
                balance += position.size * exit_px
                ct = ClosedTrade(
                    entry_idx=position.entry_idx,
                    exit_idx=i,
                    entry_price=position.entry_price,
                    exit_price=exit_px,
                    size=position.size,
                    pnl=pnl,
                    pnl_pct=round(pnl_pct, 3),
                    exit_reason=exit_reason,
                    won=pnl > 0,
                )
                closed_trades.append(ct)
                rm.record_trade(TradeRecord(
                    entry_price=position.entry_price,
                    exit_price=exit_px,
                    size=position.size,
                ))
                position = None

        # ── Open new position ─────────────────────────────────────────
        if position is None and entry_signal(data, i):
            entry_px = _apply_costs(close, "buy", cfg.commission, cfg.slippage)
            stop_px = entry_px * (1 - cfg.stoploss)
            sizing = rm.position_size(entry_px, stop_px)
            pos_val = min(sizing["position_value"], balance * 0.95)
            if pos_val > 0 and balance > pos_val:
                size = pos_val / entry_px
                balance -= size * entry_px
                position = OpenTrade(
                    entry_price=entry_px,
                    size=size,
                    stop_price=stop_px,
                    entry_idx=i,
                    highest_since_entry=entry_px,
                )

        equity = balance + (position.size * close if position else 0)
        equity_curve.append(equity)

    # Force close remaining position at last close
    if position is not None:
        last = data[-1]["close"]
        exit_px = _apply_costs(last, "sell", cfg.commission, cfg.slippage)
        pnl = (exit_px - position.entry_price) * position.size
        balance += position.size * exit_px
        closed_trades.append(ClosedTrade(
            entry_idx=position.entry_idx,
            exit_idx=len(data) - 1,
            entry_price=position.entry_price,
            exit_price=exit_px,
            size=position.size,
            pnl=pnl,
            pnl_pct=round((exit_px - position.entry_price) / position.entry_price * 100, 3),
            exit_reason="end_of_data",
            won=pnl > 0,
        ))

    stats = rm.portfolio_stats(cfg.initial_balance)

    return {
        "trades": closed_trades,
        "equity_curve": equity_curve,
        "stats": stats,
        "config": cfg,
        "final_balance": round(balance, 2),
    }
