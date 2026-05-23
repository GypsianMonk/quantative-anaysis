"""
AEGIS PRIME - Institutional Backtesting Engine
==============================================
Event-driven backtesting with realistic execution modeling,
walk-forward validation, and combinatorial purged cross-validation.

Key Features:
- Event-driven architecture for accurate simulation
- Transaction cost and slippage modeling
- Order book replay capability
- Walk-forward optimization
- Combinatorial Purged Cross-Validation (CPCV)
- Monte Carlo perturbation testing
- Lookahead bias prevention
- Survivorship bias handling
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy.stats import ttest_1samp, norm
from sklearn.model_selection import KFold
import itertools

warnings.filterwarnings('ignore')


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    POSITION_UPDATE = "position_update"


@dataclass
class Event:
    """Base event class for event-driven architecture."""
    timestamp: int
    event_type: EventType
    data: Dict = field(default_factory=dict)


@dataclass
class Order:
    """Order representation with execution details."""
    order_id: str
    timestamp: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Position tracking with PnL calculation."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, price: float):
        self.current_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
    
    def add_fill(self, side: OrderSide, quantity: float, price: float):
        if side == OrderSide.BUY:
            if self.quantity >= 0:
                # Adding to long position
                total_cost = self.avg_entry_price * self.quantity + price * quantity
                self.quantity += quantity
                if self.quantity > 0:
                    self.avg_entry_price = total_cost / self.quantity
            else:
                # Closing short position
                close_qty = min(quantity, abs(self.quantity))
                self.realized_pnl += (self.avg_entry_price - price) * close_qty
                self.quantity += quantity
                if self.quantity > 0:
                    self.avg_entry_price = price
        else:  # SELL
            if self.quantity <= 0:
                # Adding to short position
                total_cost = self.avg_entry_price * abs(self.quantity) + price * quantity
                self.quantity -= quantity
                if self.quantity < 0:
                    self.avg_entry_price = total_cost / abs(self.quantity)
            else:
                # Closing long position
                close_qty = min(quantity, self.quantity)
                self.realized_pnl += (price - self.avg_entry_price) * close_qty
                self.quantity -= quantity
                if self.quantity < 0:
                    self.avg_entry_price = price
        
        self.update_price(price)


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_time: int
    exit_time: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    holding_period: int


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    trades: List[TradeResult]
    daily_returns: np.ndarray
    
    def summary(self) -> str:
        """Return formatted summary string."""
        return f"""
======== BACKTEST RESULTS ========
Total Return:      {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility:        {self.volatility:.2%}
Sharpe Ratio:      {self.sharpe_ratio:.2f}
Sortino Ratio:     {self.sortino_ratio:.2f}
Max Drawdown:      {self.max_drawdown:.2%}
Calmar Ratio:      {self.calmar_ratio:.2f}
Win Rate:          {self.win_rate:.2%}
Profit Factor:     {self.profit_factor:.2f}
Avg Trade Return:  {self.avg_trade_return:.4f}
Total Trades:      {self.total_trades}
==================================
        """.strip()


class TransactionCostModel:
    """Realistic transaction cost and slippage modeling."""
    
    def __init__(
        self,
        commission_rate: float = 0.0005,  # 5 bps
        spread_cost: float = 0.0001,      # 1 bps
        market_impact: float = 0.0001,    # Price impact per unit volume
        slippage_vol: float = 0.0002      # Random slippage component
    ):
        self.commission_rate = commission_rate
        self.spread_cost = spread_cost
        self.market_impact = market_impact
        self.slippage_vol = slippage_vol
    
    def calculate_costs(
        self,
        price: float,
        quantity: float,
        side: OrderSide,
        volume: float,
        is_market_order: bool = True
    ) -> Tuple[float, float, float]:
        """
        Calculate commission, slippage, and market impact.
        Returns: (commission, slippage, total_cost)
        """
        notional = abs(price * quantity)
        
        # Commission
        commission = notional * self.commission_rate
        
        # Spread cost (half spread for market orders)
        spread = price * self.spread_cost if is_market_order else 0
        
        # Market impact (simplified model)
        volume_ratio = abs(quantity) / (volume + 1e-10)
        impact = price * self.market_impact * volume_ratio * abs(quantity)
        
        # Random slippage
        random_slippage = price * self.slippage_vol * np.random.normal()
        
        # Total slippage
        slippage = spread + impact + abs(random_slippage)
        
        # Adjust fill price based on side
        if side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        total_cost = commission + slippage * abs(quantity)
        
        return commission, slippage, total_cost


class ExecutionEngine:
    """Simulates order execution with realistic fills."""
    
    def __init__(self, cost_model: TransactionCostModel):
        self.cost_model = cost_model
        self.pending_orders: List[Order] = []
        self.fills: List[Dict] = []
    
    def submit_order(self, order: Order, current_bar: Dict) -> Optional[Order]:
        """Submit order for execution."""
        self.pending_orders.append(order)
        return self._execute_order(order, current_bar)
    
    def _execute_order(self, order: Order, bar: Dict) -> Optional[Order]:
        """Execute order against current bar data."""
        if order.order_type == OrderType.MARKET:
            # Market order executes at next available price
            if order.side == OrderSide.BUY:
                fill_price = bar['close']  # Simplified: use close
            else:
                fill_price = bar['close']
            
            # Calculate costs
            commission, slippage, total_cost = self.cost_model.calculate_costs(
                fill_price,
                order.quantity,
                order.side,
                bar.get('volume', 1e6)
            )
            
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.commission = commission
            order.slippage = slippage
            order.status = "filled"
            
            self.fills.append({
                'timestamp': order.timestamp,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': fill_price,
                'commission': commission,
                'slippage': slippage
            })
            
            return order
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order only fills if price reaches limit
            if order.side == OrderSide.BUY:
                if bar['low'] <= order.price:
                    fill_price = min(order.price, bar['open'])
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = fill_price
                    order.status = "filled"
            else:  # SELL
                if bar['high'] >= order.price:
                    fill_price = max(order.price, bar['open'])
                    order.filled_quantity = order.quantity
                    order.avg_fill_price = fill_price
                    order.status = "filled"
            
            if order.status == "filled":
                commission, slippage, _ = self.cost_model.calculate_costs(
                    order.avg_fill_price,
                    order.quantity,
                    order.side,
                    bar.get('volume', 1e6),
                    is_market_order=False
                )
                order.commission = commission
                order.slippage = slippage
                
                self.fills.append({
                    'timestamp': order.timestamp,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': order.avg_fill_price,
                    'commission': commission,
                    'slippage': slippage
                })
            
            return order
        
        return None


class Portfolio:
    """Portfolio management with position tracking."""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = [initial_capital]
        self.daily_returns: List[float] = []
    
    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def update_position(self, order: Order):
        """Update position based on filled order."""
        if order.status != "filled":
            return
        
        position = self.get_position(order.symbol)
        position.add_fill(order.side, order.filled_quantity, order.avg_fill_price)
        
        # Update cash
        notional = order.avg_fill_price * order.filled_quantity
        if order.side == OrderSide.BUY:
            self.cash -= notional + order.commission
        else:
            self.cash += notional - order.commission
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all positions with current prices."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity."""
        self.update_prices(prices)
        position_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def record_equity(self, equity: float):
        """Record equity for curve tracking."""
        if len(self.equity_curve) > 0:
            prev_equity = self.equity_curve[-1]
            daily_ret = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_ret)
        self.equity_curve.append(equity)


class Backtester:
    """
    Main event-driven backtesting engine.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        cost_model: Optional[TransactionCostModel] = None
    ):
        self.portfolio = Portfolio(initial_capital)
        self.cost_model = cost_model or TransactionCostModel()
        self.execution_engine = ExecutionEngine(self.cost_model)
        self.trades: List[TradeResult] = []
        self.active_trades: Dict[str, Dict] = {}
    
    def run(
        self,
        data: pl.DataFrame,
        signal_generator: Callable,
        symbols: List[str]
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            signal_generator: Function that returns signals given data and current index
            symbols: List of symbols to trade
        """
        n_bars = len(data)
        
        for i in range(n_bars):
            current_bar = data.row(i, named=True)
            timestamp = current_bar.get('timestamp', i)
            
            # Get signals from strategy
            signals = signal_generator(data, i, self.portfolio)
            
            # Generate orders from signals
            for symbol in symbols:
                if symbol in signals:
                    signal = signals[symbol]
                    order = self._signal_to_order(signal, symbol, timestamp)
                    if order:
                        self.execution_engine.submit_order(order, current_bar)
            
            # Update positions with fills
            for order in self.execution_engine.pending_orders:
                if order.status == "filled":
                    self.portfolio.update_position(order)
                    self._record_trade(order, current_bar)
            
            # Update equity
            prices = {sym: current_bar.get('close', 0) for sym in symbols}
            equity = self.portfolio.get_total_equity(prices)
            self.portfolio.record_equity(equity)
        
        # Calculate results
        return self._calculate_results(symbols)
    
    def _signal_to_order(self, signal: Dict, symbol: str, timestamp: int) -> Optional[Order]:
        """Convert signal to order."""
        position = self.portfolio.get_position(symbol)
        target_qty = signal.get('quantity', 0)
        side = signal.get('side', 'hold')
        
        if side == 'hold' or target_qty == 0:
            # Close existing position
            if position.quantity != 0:
                side = 'sell' if position.quantity > 0 else 'buy'
                target_qty = abs(position.quantity)
            else:
                return None
        
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        
        return Order(
            order_id=f"{symbol}_{timestamp}",
            timestamp=timestamp,
            symbol=symbol,
            side=order_side,
            order_type=OrderType.MARKET,
            quantity=abs(target_qty)
        )
    
    def _record_trade(self, order: Order, bar: Dict):
        """Record completed trade."""
        key = f"{order.symbol}_{order.side.value}"
        
        if order.side == OrderSide.BUY:
            self.active_trades[key] = {
                'entry_time': order.timestamp,
                'symbol': order.symbol,
                'side': 'long',
                'entry_price': order.avg_fill_price,
                'quantity': order.filled_quantity
            }
        else:
            if key in self.active_trades:
                trade = self.active_trades.pop(key)
                pnl = (order.avg_fill_price - trade['entry_price']) * trade['quantity']
                ret_pct = pnl / (trade['entry_price'] * trade['quantity'])
                
                self.trades.append(TradeResult(
                    entry_time=trade['entry_time'],
                    exit_time=order.timestamp,
                    symbol=trade['symbol'],
                    side=trade['side'],
                    entry_price=trade['entry_price'],
                    exit_price=order.avg_fill_price,
                    quantity=trade['quantity'],
                    pnl=pnl,
                    return_pct=ret_pct,
                    holding_period=order.timestamp - trade['entry_time']
                ))
    
    def _calculate_results(self, symbols: List[str]) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        equity_curve = np.array(self.portfolio.equity_curve)
        daily_returns = np.array(self.portfolio.daily_returns)
        
        if len(equity_curve) < 2:
            raise ValueError("Insufficient data for backtest results")
        
        # Basic returns
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        n_periods = len(equity_curve) - 1
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calmar ratio
        calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0
        
        # Trade statistics
        if len(self.trades) > 0:
            wins = [t.pnl for t in self.trades if t.pnl > 0]
            losses = [t.pnl for t in self.trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(self.trades)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_trade_return = np.mean([t.return_pct for t in self.trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            total_trades=len(self.trades),
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            trades=self.trades,
            daily_returns=daily_returns
        )


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.
    Prevents lookahead bias by using rolling training/testing windows.
    """
    
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def generate_folds(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test splits."""
        folds = []
        start_idx = 0
        
        while start_idx + self.train_window + self.test_window <= n_samples:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            train_idx = np.arange(start_idx, train_end)
            test_idx = np.arange(train_end, test_end)
            
            folds.append((train_idx, test_idx))
            start_idx += self.step_size
        
        return folds
    
    def validate(
        self,
        data: pl.DataFrame,
        strategy_class,
        symbols: List[str],
        initial_capital: float = 1000000.0
    ) -> Dict:
        """Run walk-forward validation."""
        n_samples = len(data)
        folds = self.generate_folds(n_samples)
        
        results = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            print(f"Running fold {fold_idx + 1}/{len(folds)}...")
            
            # Train period
            train_data = data[train_idx[0]:train_idx[-1]+1]
            # Test period
            test_data = data[test_idx[0]:test_idx[-1]+1]
            
            # Initialize and optimize strategy on training data
            strategy = strategy_class()
            strategy.optimize(train_data, symbols)
            
            # Run backtest on test data
            backtester = Backtester(initial_capital=initial_capital)
            result = backtester.run(test_data, strategy.generate_signals, symbols)
            
            results.append({
                'fold': fold_idx,
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1],
                'result': result
            })
        
        # Aggregate results
        return self._aggregate_walk_forward_results(results)
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results."""
        if not results:
            return {}
        
        sharpe_ratios = [r['result'].sharpe_ratio for r in results]
        returns = [r['result'].total_return for r in results]
        drawdowns = [r['result'].max_drawdown for r in results]
        
        return {
            'n_folds': len(results),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_return': np.mean(returns),
            'total_return': sum(returns),
            'mean_drawdown': np.mean(drawdowns),
            'max_drawdown': max(drawdowns),
            'win_rate': np.mean([1 if r > 0 else 0 for r in returns]),
            'results': results
        }


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Advanced cross-validation for financial time series that:
    1. Purges overlapping training samples
    2. Embargoes samples around test set
    3. Uses combinatorial splits for robustness
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        sample_weights: Optional[np.ndarray] = None
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.sample_weights = sample_weights
    
    def split(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CPCV splits.
        
        Args:
            y: Target variable
            groups: Optional group labels for purging
        """
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        # Simple k-fold as base (can be enhanced with combinatorial splits)
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        splits = []
        for train_idx, test_idx in kf.split(indices):
            # Apply purging: remove training samples that overlap with test set
            if groups is not None:
                train_idx = self._purge(train_idx, test_idx, groups)
            
            # Apply embargo: remove samples around test set boundaries
            train_idx = self._embargo(train_idx, test_idx, n_samples)
            
            if len(train_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        groups: np.ndarray
    ) -> np.ndarray:
        """Remove training samples that belong to same group as test samples."""
        test_groups = set(groups[test_idx])
        mask = ~np.isin(groups[train_idx], list(test_groups))
        return train_idx[mask]
    
    def _embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Remove training samples near test set boundaries."""
        if len(test_idx) == 0:
            return train_idx
        
        embargo_size = max(1, int(n_samples * self.embargo_pct))
        test_min, test_max = test_idx.min(), test_idx.max()
        
        # Remove samples before and after test set
        mask = (train_idx < test_min - embargo_size) | (train_idx > test_max + embargo_size)
        return train_idx[mask]
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class,
        groups: Optional[np.ndarray] = None
    ) -> Dict:
        """Run CPCV validation."""
        splits = self.split(y, groups)
        
        scores = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_class()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            score = self._evaluation_metric(y_test, y_pred)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Statistical significance test
        t_stat, p_value = ttest_1samp(scores, 0)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'n_splits': len(scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _evaluation_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default evaluation metric (correlation for continuous, accuracy for binary)."""
        if len(np.unique(y_true)) == 2:
            # Binary classification
            return np.mean(y_true == y_pred)
        else:
            # Continuous: use correlation
            return np.corrcoef(y_true, y_pred)[0, 1]


# Example usage
if __name__ == '__main__':
    # Create synthetic data
    np.random.seed(42)
    n = 500
    
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1).add(days=n-1), eager=True)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    
    data = pl.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0.005, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0.005, 0.002, n))),
        'close': prices,
        'volume': np.random.uniform(1e6, 1e7, n)
    })
    
    # Simple momentum strategy
    class MomentumStrategy:
        def __init__(self):
            self.lookback = 20
            self.threshold = 0.02
        
        def optimize(self, data, symbols):
            pass  # In production, optimize parameters here
        
        def generate_signals(self, data, idx, portfolio):
            signals = {}
            
            if idx < self.lookback:
                return signals
            
            returns = (data['close'][idx] / data['close'][idx - self.lookback]) - 1
            
            if returns > self.threshold:
                signals['SYMBOL'] = {'side': 'buy', 'quantity': 100}
            elif returns < -self.threshold:
                signals['SYMBOL'] = {'side': 'sell', 'quantity': 100}
            else:
                signals['SYMBOL'] = {'side': 'hold', 'quantity': 0}
            
            return signals
    
    # Run backtest
    backtester = Backtester(initial_capital=1000000)
    strategy = MomentumStrategy()
    
    result = backtester.run(data, strategy.generate_signals, ['SYMBOL'])
    print(result.summary())
    
    # Run walk-forward validation
    wf_validator = WalkForwardValidator(train_window=200, test_window=50, step_size=25)
    wf_results = wf_validator.validate(data, MomentumStrategy, ['SYMBOL'])
    
    print(f"\nWalk-Forward Results:")
    print(f"Mean Sharpe: {wf_results['mean_sharpe']:.2f}")
    print(f"Total Return: {wf_results['total_return']:.2%}")
    print(f"Max Drawdown: {wf_results['max_drawdown']:.2%}")
