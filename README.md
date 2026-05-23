# QuantCore — Quantitative Trading System

A production-grade quant trading system built around a **RSI + MACD confluence strategy** with Bollinger Bands filtering, ATR-based dynamic stops, and full Freqtrade integration for KuCoin.

---

## Architecture

```
quantative-anaysis/
├── core/
│   ├── indicators.py       # Pure-Python RSI, MACD, BB, EMA, ATR, Stochastic, VWAP
│   ├── risk_manager.py     # Kelly criterion, position sizing, portfolio stats
│   ├── backtest_engine.py  # Event-driven backtester with SL/TP/trailing stop
│   └── scanner.py          # Multi-pair signal scanner (live ccxt or simulated)
│
├── strategies/
│   └── QuantCoreStrategy.py  # Freqtrade IStrategy — hyperopt-ready
│
├── dashboard/
│   └── app.py              # FastAPI REST API (scanner, backtest, risk endpoints)
│
├── config/
│   └── kucoin_config.json  # Freqtrade config for KuCoin (dry-run safe)
│
├── tests/
└── notebooks/
```

---

## Strategy Logic

| Condition | Entry (BUY) | Exit (SELL) |
|---|---|---|
| RSI(14) | < 35 (oversold) | > 65 (overbought) |
| MACD(12,26,9) | Bullish crossover | Bearish crossover |
| BB filter | Near lower band (optional) | Near upper band (optional) |
| EMA50 trend | Above EMA50 (toggleable) | — |

All thresholds are `IntParameter` / `DecimalParameter` — fully Hyperopt-compatible.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download historical data (KuCoin)

```bash
freqtrade download-data \
  -p BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --timeframe 4h \
  --exchange kucoin \
  --timerange 20230101-
```

### 3. Backtest

```bash
freqtrade backtesting \
  --strategy QuantCoreStrategy \
  --config config/kucoin_config.json \
  --timerange 20240101- \
  --export trades
```

### 4. Hyperopt (tune parameters)

```bash
freqtrade hyperopt \
  --strategy QuantCoreStrategy \
  --config config/kucoin_config.json \
  --hyperopt-loss SharpeHyperOptLoss \
  -e 500 \
  --spaces buy sell stoploss \
  --timeframe 4h
```

### 5. Dry-run (paper trade)

```bash
# Edit config/kucoin_config.json → "dry_run": true
freqtrade trade \
  --strategy QuantCoreStrategy \
  --config config/kucoin_config.json
```

### 6. Run API dashboard

```bash
uvicorn dashboard.app:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/scanner?pairs=BTC/USDT,ETH/USDT` | Multi-pair signal scan |
| POST | `/backtest` | Run backtest on simulated data |
| POST | `/risk` | Position sizing + Kelly criterion |

---

## Risk Management

The `RiskManager` class handles:

- **Fixed fractional** sizing (default 2% risk per trade)
- **Kelly criterion** sizing derived from live trade history
- **ATR-based dynamic stop-loss** via `custom_stoploss()` in the strategy
- **Trailing stop** activates at +3%, trails 1.5% below peak

```python
from core.risk_manager import RiskManager

rm = RiskManager(balance=10_000, risk_per_trade=0.02)
sizing = rm.position_size(entry_price=67_000, stop_price=63_500)
print(sizing)
# {'risk_amount': 200.0, 'stop_distance_pct': 5.224, 'position_value': 3830.0,
#  'units': 0.057164, 'leverage': 0.38, 'tp_1r': ..., 'tp_2r': ..., 'tp_3r': ...}
```

---

## Live Trading Warning

> **Use dry-run mode first.** Set `"dry_run": true` in `kucoin_config.json` until you have validated at least 3 months of paper trading results. Never risk more than 2% per trade. Past simulated performance does not guarantee future results.

Before going live:
1. Add your KuCoin API key/secret to `config/kucoin_config.json`
2. Set `"dry_run": false`
3. Change `api_server.jwt_secret_key` and `api_server.password`
4. Never commit your live config — it is in `.gitignore`

---

## Author

**GypsianMonk** — [github.com/GypsianMonk](https://github.com/GypsianMonk)
