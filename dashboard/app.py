"""
dashboard/app.py
FastAPI backend that serves scanner signals and backtest results.
Run: uvicorn dashboard.app:app --reload --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import random
import math

from core.indicators import enrich_dataframe, ema, rsi as rsi_calc, macd as macd_calc, bollinger_bands
from core.risk_manager import RiskManager, TradeRecord
from core.scanner import Scanner, Signal

app = FastAPI(
    title="QuantCore API",
    description="Quant trading system — scanner, backtest, risk manager",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────

class StrategyCfg(BaseModel):
    rsi_oversold:   int   = 35
    rsi_overbought: int   = 65
    macd_fast:      int   = 12
    macd_slow:      int   = 26
    macd_signal:    int   = 9
    bb_period:      int   = 20
    bb_std:         float = 2.0

class BacktestRequest(BaseModel):
    pair:     str = "BTC/USDT"
    candles:  int = 200
    strategy: StrategyCfg = StrategyCfg()

class RiskRequest(BaseModel):
    balance:        float = 10_000.0
    risk_pct:       float = 2.0
    entry_price:    float = 67_000.0
    stop_pct:       float = 5.0
    win_rate:       float = 55.0
    avg_win_pct:    float = 4.5
    avg_loss_pct:   float = 2.8


# ── Helpers ───────────────────────────────────────────────────────────

def _gen_candles(n: int, start: float, vol: float = 0.013) -> List[dict]:
    candles, p = [], start
    for _ in range(n):
        ch = (random.random() - 0.48) * vol
        p *= 1 + ch
        o = p / (1 + ch)
        h = max(o, p) * (1 + random.random() * 0.004)
        l = min(o, p) * (1 - random.random() * 0.004)
        candles.append({"open": o, "high": h, "low": l, "close": p, "volume": random.uniform(200, 1200)})
    return candles

_START_PRICES = {"BTC/USDT": 67_400, "ETH/USDT": 3_440, "SOL/USDT": 154, "BNB/USDT": 414, "XRP/USDT": 0.52}


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/scanner", response_model=List[dict])
def scanner(
    pairs: str = Query("BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT"),
    timeframe: str = "4h",
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
):
    cfg = {"rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
           "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "bb_period": 20, "bb_std": 2.0}
    sc = Scanner(cfg)
    results = []
    for pair in pairs.split(","):
        pair = pair.strip()
        start = _START_PRICES.get(pair, 100)
        candles = _gen_candles(220, start)
        sig = sc.scan_candles(pair, timeframe, candles)
        results.append({
            "pair": sig.pair, "signal": sig.signal, "strength": sig.strength,
            "price": round(sig.price, 4), "rsi": sig.rsi,
            "macd_dir": sig.macd_dir, "ema_trend": sig.ema_trend,
            "bb_position": sig.bb_position, "reason": sig.reason,
        })
    return results


@app.post("/backtest")
def backtest(req: BacktestRequest):
    start = _START_PRICES.get(req.pair, 100)
    candles = _gen_candles(req.candles, start)
    closes = [c["close"] for c in candles]
    cfg = req.strategy

    rs = rsi_calc(closes)
    mc = macd_calc(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    balance, pos, trades, equity = 10_000.0, None, [], [10_000.0]
    for i in range(1, len(closes)):
        if not rs[i] or not mc["macd_line"][i] or not mc["signal_line"][i]:
            continue
        cross_up = mc["macd_line"][i-1] < mc["signal_line"][i-1] and mc["macd_line"][i] > mc["signal_line"][i]
        cross_dn = mc["macd_line"][i-1] > mc["signal_line"][i-1] and mc["macd_line"][i] < mc["signal_line"][i]

        if cross_up and rs[i] < cfg.rsi_oversold and pos is None:
            pos = {"entry": closes[i], "size": balance * 0.95 / closes[i]}
            balance -= pos["size"] * closes[i]

        if cross_dn and rs[i] > cfg.rsi_overbought and pos:
            pnl = (closes[i] - pos["entry"]) * pos["size"]
            balance += pos["size"] * closes[i]
            trades.append({"pnl": round(pnl, 2), "pct": round((closes[i]-pos["entry"])/pos["entry"]*100, 3), "won": pnl > 0})
            equity.append(round(balance, 2))
            pos = None

    wins = [t for t in trades if t["won"]]
    wr = len(wins)/len(trades)*100 if trades else 0
    rets = [t["pct"] for t in trades]
    avg = sum(rets)/(len(rets) or 1)
    std = math.sqrt(sum((r-avg)**2 for r in rets)/(len(rets) or 1))
    sharpe = avg/std*math.sqrt(252/4) if std > 0 else 0
    peak, mdd = 10_000.0, 0.0
    for e in equity:
        if e > peak: peak = e
        d = (peak - e)/peak*100
        if d > mdd: mdd = d

    return {
        "pair": req.pair,
        "total_return_pct": round((balance-10_000)/10_000*100, 2),
        "win_rate": round(wr, 1),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 1),
        "num_trades": len(trades),
        "final_balance": round(balance, 2),
        "equity_curve": equity,
        "recent_trades": trades[-10:],
    }


@app.post("/risk")
def risk(req: RiskRequest):
    rm = RiskManager(balance=req.balance, risk_per_trade=req.risk_pct/100)
    stop_price = req.entry_price * (1 - req.stop_pct/100)
    sizing = rm.position_size(req.entry_price, stop_price)

    w = req.win_rate / 100
    b = req.avg_win_pct / max(req.avg_loss_pct, 1e-9)
    full_kelly = max(0.0, w - (1 - w) / b)
    kelly_size = min(full_kelly * 0.5, 0.3) * req.balance

    return {
        **sizing,
        "full_kelly_pct": round(full_kelly * 100, 2),
        "half_kelly_size": round(kelly_size, 2),
        "expectancy_pct": round(w * req.avg_win_pct - (1-w) * req.avg_loss_pct, 3),
        "rr_ratio": round(req.avg_win_pct / max(req.avg_loss_pct, 1e-9), 2),
    }
