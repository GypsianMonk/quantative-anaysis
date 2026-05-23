"""
core/risk_manager.py
Position sizing, Kelly criterion, drawdown tracking,
and trade-level risk/reward calculations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class TradeRecord:
    entry_price: float
    exit_price: float
    size: float            # units / coins
    direction: str = "long"  # "long" | "short"

    @property
    def pnl(self) -> float:
        if self.direction == "long":
            return (self.exit_price - self.entry_price) * self.size
        return (self.entry_price - self.exit_price) * self.size

    @property
    def pnl_pct(self) -> float:
        ref = self.entry_price * self.size
        return self.pnl / ref * 100 if ref else 0.0

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class PortfolioStats:
    total_return_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    rr_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    num_trades: int
    full_kelly: float
    half_kelly: float
    expectancy_pct: float


class RiskManager:
    """
    Handles position sizing and portfolio-level risk for a single account.

    Parameters
    ----------
    balance : float
        Starting account balance in quote currency (e.g. USDT).
    risk_per_trade : float
        Fraction of balance risked per trade (default 0.02 = 2%).
    max_open_trades : int
        Maximum concurrent open positions.
    use_kelly : bool
        If True, derive position size from Kelly criterion
        (requires prior trade history).
    kelly_fraction : float
        Multiplier on full Kelly — use 0.25–0.5 in live trading.
    """

    def __init__(
        self,
        balance: float = 10_000.0,
        risk_per_trade: float = 0.02,
        max_open_trades: int = 3,
        use_kelly: bool = False,
        kelly_fraction: float = 0.5,
    ):
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_open_trades = max_open_trades
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[float] = [balance]

    # ── Position sizing ───────────────────────────────────────────────

    def position_size(
        self,
        entry_price: float,
        stop_price: float,
        current_kelly: Optional[float] = None,
    ) -> dict:
        """
        Compute recommended position size.

        Returns a dict with:
          risk_amount, stop_distance_pct, position_value,
          units, leverage, tp_1r, tp_2r, tp_3r
        """
        stop_dist = abs(entry_price - stop_price) / entry_price

        if self.use_kelly and current_kelly is not None:
            risk_frac = min(
                current_kelly * self.kelly_fraction,
                self.risk_per_trade * 3,   # cap at 3× normal risk
            )
        else:
            risk_frac = self.risk_per_trade

        risk_amount = self.balance * risk_frac
        position_value = risk_amount / stop_dist if stop_dist > 0 else 0.0
        units = position_value / entry_price if entry_price > 0 else 0.0
        leverage = position_value / self.balance if self.balance > 0 else 1.0

        direction = 1 if entry_price > stop_price else -1
        tp_1r = entry_price * (1 + direction * stop_dist)
        tp_2r = entry_price * (1 + direction * stop_dist * 2)
        tp_3r = entry_price * (1 + direction * stop_dist * 3)

        return {
            "risk_amount": round(risk_amount, 2),
            "stop_distance_pct": round(stop_dist * 100, 3),
            "position_value": round(position_value, 2),
            "units": round(units, 6),
            "leverage": round(leverage, 2),
            "tp_1r": round(tp_1r, 4),
            "tp_2r": round(tp_2r, 4),
            "tp_3r": round(tp_3r, 4),
        }

    # ── Trade recording ───────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord) -> None:
        self.balance += trade.pnl
        self._trades.append(trade)
        self._equity_curve.append(self.balance)

    # ── Kelly criterion ───────────────────────────────────────────────

    def kelly_criterion(self) -> dict:
        """
        Compute full Kelly fraction from trade history.
        Kelly = W/A - (1-W)/B
          W = win rate, A = avg loss %, B = avg win %
        """
        if len(self._trades) < 10:
            return {"full_kelly": 0.0, "half_kelly": 0.0, "note": "Need ≥10 trades"}

        wins = [t for t in self._trades if t.won]
        losses = [t for t in self._trades if not t.won]
        w = len(wins) / len(self._trades)
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(t.pnl_pct for t in losses) / len(losses)) if losses else 1.0

        full_kelly = max(0.0, w / avg_loss - (1 - w) / max(avg_win, 1e-9)) if avg_win > 0 else 0.0
        full_kelly = min(full_kelly, 0.5)  # hard cap
        return {
            "full_kelly": round(full_kelly * 100, 2),
            "half_kelly": round(full_kelly * 50, 2),
            "quarter_kelly": round(full_kelly * 25, 2),
            "win_rate": round(w * 100, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "rr_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0,
        }

    # ── Portfolio stats ───────────────────────────────────────────────

    def portfolio_stats(self, initial_balance: Optional[float] = None) -> PortfolioStats:
        init = initial_balance or self._equity_curve[0]
        trades = self._trades
        if not trades:
            return PortfolioStats(0,0,0,0,0,0,0,0,0,0,0,0,0)

        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        wr = len(wins) / len(trades) * 100
        avg_win = sum(t.pnl_pct for t in wins) / max(len(wins), 1)
        avg_loss = abs(sum(t.pnl_pct for t in losses) / max(len(losses), 1))
        rr = avg_win / avg_loss if avg_loss > 0 else 0.0
        total_ret = (self.balance - init) / init * 100

        rets = [t.pnl_pct for t in trades]
        avg_ret = sum(rets) / len(rets)
        std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in rets) / len(rets))
        sharpe = (avg_ret / std_ret * math.sqrt(252 / 4)) if std_ret > 0 else 0.0

        neg_rets = [r for r in rets if r < 0]
        std_neg = math.sqrt(sum(r**2 for r in neg_rets) / max(len(neg_rets), 1))
        sortino = (avg_ret / std_neg * math.sqrt(252 / 4)) if std_neg > 0 else 0.0

        peak, mdd = init, 0.0
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        kc = self.kelly_criterion()
        expectancy = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss

        return PortfolioStats(
            total_return_pct=round(total_ret, 2),
            win_rate=round(wr, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            rr_ratio=round(rr, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(mdd, 1),
            profit_factor=round(pf, 2),
            num_trades=len(trades),
            full_kelly=kc.get("full_kelly", 0.0),
            half_kelly=kc.get("half_kelly", 0.0),
            expectancy_pct=round(expectancy, 3),
        )

    # ── Drawdown utilities ────────────────────────────────────────────

    @property
    def current_drawdown_pct(self) -> float:
        if not self._equity_curve:
            return 0.0
        peak = max(self._equity_curve)
        return (peak - self._equity_curve[-1]) / peak * 100 if peak > 0 else 0.0

    @property
    def equity_curve(self) -> List[float]:
        return list(self._equity_curve)
