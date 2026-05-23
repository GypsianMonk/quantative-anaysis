"""
Logging configuration for AEGIS trading system.

Provides structured logging with JSON formatting, correlation IDs for request tracing,
and integration with monitoring systems.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Optional
from pythonjsonlogger import jsonlogger
from .settings import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields for trading context."""
    
    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add correlation ID if present (for distributed tracing)
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add trading-specific fields
        if hasattr(record, 'symbol'):
            log_record['symbol'] = record.symbol
        if hasattr(record, 'strategy'):
            log_record['strategy'] = record.strategy
        if hasattr(record, 'order_id'):
            log_record['order_id'] = record.order_id
        if hasattr(record, 'pnl'):
            log_record['pnl'] = record.pnl


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = True
) -> logging.Logger:
    """
    Configure logging for the AEGIS system.
    
    Args:
        log_level: Override default log level from settings
        log_file: Optional file path for log output
        enable_json: Enable JSON formatting for structured logs
    
    Returns:
        Configured logger instance
    """
    level = getattr(logging, (log_level or settings.log_level).upper())
    
    # Create root logger
    logger = logging.getLogger('aegis')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_json:
        # JSON formatter for production (structured logging)
        formatter = CustomJsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        # Human-readable formatter for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.INFO)
    
    return logger


class TradingLogger:
    """
    Context-aware logger for trading operations.
    
    Provides methods for logging trading-specific events with relevant context.
    """
    
    def __init__(self, name: str = 'aegis'):
        self.logger = logging.getLogger(name)
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        order_id: Optional[str] = None,
        pnl: Optional[float] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """Log a message with trading context."""
        extra = {
            'symbol': symbol,
            'strategy': strategy,
            'order_id': order_id,
            'pnl': pnl,
            'correlation_id': correlation_id,
            **kwargs
        }
        
        # Filter out None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    # Trading-specific convenience methods
    def order_submitted(self, symbol: str, side: str, quantity: float, price: float, order_id: str):
        self.info(
            f"Order submitted: {side} {quantity} {symbol} @ {price}",
            symbol=symbol,
            order_id=order_id,
            side=side,
            quantity=quantity,
            price=price
        )
    
    def order_filled(self, symbol: str, side: str, quantity: float, fill_price: float, order_id: str, pnl: float = 0.0):
        self.info(
            f"Order filled: {side} {quantity} {symbol} @ {fill_price}, PnL: {pnl:.2f}",
            symbol=symbol,
            order_id=order_id,
            pnl=pnl
        )
    
    def signal_generated(self, symbol: str, strategy: str, signal: float, confidence: float):
        self.info(
            f"Signal generated: {symbol} by {strategy}, signal={signal:.4f}, confidence={confidence:.2%}",
            symbol=symbol,
            strategy=strategy,
            signal=signal,
            confidence=confidence
        )
    
    def risk_breach(self, symbol: str, metric: str, value: float, limit: float):
        self.warning(
            f"Risk breach: {metric}={value:.4f} exceeds limit={limit:.4f} for {symbol}",
            symbol=symbol,
            metric=metric,
            value=value,
            limit=limit
        )
    
    def circuit_breaker_triggered(self, reason: str, details: dict):
        self.critical(
            f"Circuit breaker triggered: {reason}",
            reason=reason,
            **details
        )


# Global logger instance
trading_logger = TradingLogger()
