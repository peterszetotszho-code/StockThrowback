"""Trading strategy definitions built on ``backtesting.Strategy``.

Each strategy overrides ``init()`` to precompute its indicators once (via
``self.I``) and ``next()`` to express entry/exit rules. Class attributes act
as tunable parameters for later optimization.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


class MAStrategy(Strategy):
    """Dual moving-average crossover: buy on golden cross, sell on death cross."""

    fast = 5
    slow = 20

    def init(self) -> None:
        """Precompute the fast and slow moving averages over the whole series."""
        self.ma_fast = self.I(
            lambda x: pd.Series(x).rolling(self.fast).mean(),
            self.data.Close,
            name=f"MA{self.fast}",
        )
        self.ma_slow = self.I(
            lambda x: pd.Series(x).rolling(self.slow).mean(),
            self.data.Close,
            name=f"MA{self.slow}",
        )

    def next(self) -> None:
        """Enter on a bullish crossover, exit on a bearish crossover."""
        if crossover(self.ma_fast, self.ma_slow):
            self.buy()
        elif crossover(self.ma_slow, self.ma_fast):
            self.position.close()


class MACDStrategy(Strategy):
    """MACD line crossing its signal line drives entries and exits."""

    fast = 12
    slow = 26
    signal = 9

    def init(self) -> None:
        """Precompute the MACD line and its signal line."""
        close = pd.Series(self.data.Close)
        macd_line = (
            close.ewm(span=self.fast, adjust=False).mean()
            - close.ewm(span=self.slow, adjust=False).mean()
        )
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        self.macd = self.I(lambda: macd_line, name="MACD")
        self.signal = self.I(lambda: signal_line, name="MACD_Signal")

    def next(self) -> None:
        """Buy when MACD crosses above signal, sell when it crosses below."""
        if crossover(self.macd, self.signal):
            self.buy()
        elif crossover(self.signal, self.macd):
            self.position.close()


class CompositeStrategy(Strategy):
    """MA golden cross filtered by RSI (only buy when not overbought)."""

    fast = 5
    slow = 20
    rsi_period = 14
    rsi_ceiling = 70

    def init(self) -> None:
        """Precompute the two moving averages and the RSI."""
        self.ma_fast = self.I(
            lambda x: pd.Series(x).rolling(self.fast).mean(),
            self.data.Close,
            name=f"MA{self.fast}",
        )
        self.ma_slow = self.I(
            lambda x: pd.Series(x).rolling(self.slow).mean(),
            self.data.Close,
            name=f"MA{self.slow}",
        )
        self.rsi = self.I(self._rsi, self.data.Close, name="RSI")

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI over the provided series."""
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def next(self) -> None:
        """Buy only on a golden cross while RSI is below the ceiling; exit on death cross."""
        golden = crossover(self.ma_fast, self.ma_slow)
        death = crossover(self.ma_slow, self.ma_fast)
        rsi = self.rsi[-1]
        # RSI may be NaN early on; treat an undefined RSI as a failed filter.
        rsi_ok = rsi is not None and not pd.isna(rsi) and rsi < self.rsi_ceiling
        if golden and rsi_ok:
            self.buy()
        elif death:
            self.position.close()
