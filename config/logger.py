import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Logging level enumeration"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass
class TrainingMetrics:
    """Data class for storing training metrics"""

    returns: List[float] = field(default_factory=list)

    def clear(self) -> None:
        """Clear all stored metrics"""
        self.returns.clear()

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Safely calculate mean of a list"""
        return sum(values) / len(values) if values else 0.0


class TradingLogger:
    """
    Enhanced logging system for trading applications with improved structure and type safety.
    """

    TRADING_DAYS_PER_YEAR = 252
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, log_file_path: str = "trading_simulation.log"):
        self.logger = logging.getLogger("trading_simulation")
        self.metrics = TrainingMetrics()
        self._setup_logger(log_file_path)

    def _setup_logger(self, log_file_path: str) -> None:
        """Configure logging handlers and formatters"""
        self.logger.setLevel(LogLevel.DEBUG.value)
        self.logger.handlers.clear()

        # Setup file handler
        file_handler = logging.FileHandler(
            log_file_path, mode="w", encoding="utf-8"
        )
        file_handler.setLevel(LogLevel.DEBUG.value)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt=self.DATE_FORMAT,
            )
        )

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LogLevel.INFO.value)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:

        self._log_training_summary(epoch, loss, additional_metrics)
        self.metrics.clear()

    def _log_training_summary(
        self,
        epoch: int,
        loss: float,
        additional_metrics: Optional[Dict[str, Any]],
    ) -> None:
        """Log training summary with current metrics"""
        self.logger.info(f"\n=== Training Metrics (Epoch {epoch}) ===")
        self.logger.info(f"Loss: {loss:.6f}")
        self.logger.info(
            f"Mean Return: {sum(self.metrics.returns) / len(self.metrics.returns):.6f}"
        )

        if additional_metrics:
            self.logger.info("Additional Metrics:")
            for key, value in additional_metrics.items():
                self.logger.info(f"{key}: {value}")

    def log_portfolio_performance(
        self,
        company: str,
        initial_value: float,
        final_value: float,
        trading_days: Optional[int] = None,
        benchmark_return: Optional[float] = None,
    ) -> None:
        """Log portfolio performance metrics with improved calculations"""
        total_return = self._calculate_total_return(initial_value, final_value)
        self.metrics.returns.append(total_return)

        self._log_portfolio_summary(
            company,
            initial_value,
            final_value,
            total_return,
            trading_days,
            benchmark_return,
        )

    def _calculate_total_return(
        self, initial_value: float, final_value: float
    ) -> float:
        """Calculate total return percentage"""
        return ((final_value - initial_value) / initial_value) * 100

    def _calculate_annualized_return(
        self, initial_value: float, final_value: float, trading_days: int
    ) -> float:
        """Calculate annualized return percentage"""
        years = trading_days / self.TRADING_DAYS_PER_YEAR
        return (((final_value / initial_value) ** (1 / years)) - 1) * 100

    def _log_portfolio_summary(
        self,
        company: str,
        initial_value: float,
        final_value: float,
        total_return: float,
        trading_days: Optional[int],
        benchmark_return: Optional[float],
    ) -> None:
        """Log comprehensive portfolio summary"""
        self.logger.info(f"\n=== Portfolio Performance Summary ({company}) ===")
        self.logger.info(f"Initial Portfolio Value: {initial_value:,.2f}")
        self.logger.info(f"Final Portfolio Value: {final_value:,.2f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")

        if trading_days:
            annualized_return = self._calculate_annualized_return(
                initial_value, final_value, trading_days
            )
            self.logger.info(f"Annualized Return: {annualized_return:.2f}%")

            if benchmark_return is not None:
                excess_return = total_return - benchmark_return
                self.logger.info(
                    f"Excess Return vs Benchmark: {excess_return:.2f}%"
                )

    def log_trade(
        self,
        timestamp: str,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        total_value: float,
    ) -> None:
        """Log trade execution details"""
        self.logger.info(
            f"Trade: {timestamp} | {symbol} | {action} | "
            f"Qty: {quantity} | Price: {price:.2f} | "
            f"Total: {total_value:,.2f}"
        )

    def log_message(self, level: LogLevel, message: str) -> None:
        """Generic logging method with type-safe level selection"""
        getattr(self.logger, level.name.lower())(message)
