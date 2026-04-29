"""
Portfolio engine.

Tracks cash, positions, NAV, P&L, and trade history for N assets.
All trades pass through the FeeSchedule so cost simulation is automatic.
Risk guardrails (drawdown halt, per-position stop-loss) are enforced here.

Multi-asset:  call step() once per bar per ticker; the engine handles
              all tickers in its universe identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    ACTIVE_FEE_SCHEDULE, FeeSchedule,
    INITIAL_CAPITAL, KELLY_FRACTION,
    MAX_PORTFOLIO_DRAWDOWN, MAX_POSITION_PCT,
    SIZING_METHOD, STOP_LOSS_PCT, VOL_TARGET,
)
from strategies.strategy import Signal

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Position:
    ticker:      str
    shares:      float = 0.0
    avg_cost:    float = 0.0    # average entry price (USD per share)
    entry_date:  Optional[datetime] = None

    @property
    def is_flat(self) -> bool:
        return abs(self.shares) < 1e-9

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealised_pnl(self, price: float) -> float:
        return self.shares * (price - self.avg_cost)

    def unrealised_pnl_pct(self, price: float) -> float:
        if self.avg_cost < 1e-9:
            return 0.0
        return (price - self.avg_cost) / self.avg_cost


@dataclass
class Trade:
    timestamp:   datetime
    ticker:      str
    side:        str            # "BUY" | "SELL"
    shares:      float
    price:       float
    fee:         float
    signal:      str

    @property
    def gross_value(self) -> float:
        return self.shares * self.price

    @property
    def net_value(self) -> float:
        return self.gross_value + (self.fee if self.side == "BUY" else -self.fee)


# ─────────────────────────────────────────────
# Portfolio engine
# ─────────────────────────────────────────────

class Portfolio:
    """
    Paper-trading portfolio for an arbitrary universe of tickers.

    Usage pattern (backtesting):
        pf = Portfolio(["AAPL", "MSFT"])
        for bar in bars:
            prices = {"AAPL": 175.0, "MSFT": 420.0}
            signal = {"AAPL": Signal.BUY, "MSFT": Signal.HOLD}
            volatilities = {"AAPL": 0.20, "MSFT": 0.18}   # annualised
            pf.step(bar.date, prices, signals, volatilities)

        report = pf.report()
    """

    def __init__(
        self,
        universe:     List[str],
        initial_cash: float       = INITIAL_CAPITAL,
        fee_schedule: FeeSchedule = ACTIVE_FEE_SCHEDULE,
        sizing:       str         = SIZING_METHOD,
    ):
        self.universe     = universe
        self.cash         = initial_cash
        self.initial_cash = initial_cash
        self.fees         = fee_schedule
        self.sizing       = sizing

        self.positions: Dict[str, Position] = {t: Position(t) for t in universe}
        self.trades:    List[Trade]         = []
        self.nav_history: List[Tuple[datetime, float]] = []

        self._peak_nav   = initial_cash
        self._halted     = False           # drawdown halt flag

    # ── Core step ────────────────────────────

    def step(
        self,
        timestamp:    datetime,
        prices:       Dict[str, float],
        signals:      Dict[str, Signal],
        volatilities: Optional[Dict[str, float]] = None,
    ):
        """
        Process one bar for all tickers:
          1. Enforce stop-losses
          2. Check portfolio-level drawdown halt
          3. Execute signals → orders
          4. Record NAV
        """
        if self._halted:
            self._record_nav(timestamp, prices)
            return

        # 1. Stop-losses
        for ticker, pos in self.positions.items():
            if ticker not in prices or pos.is_flat:
                continue
            price = prices[ticker]
            if pos.unrealised_pnl_pct(price) < -STOP_LOSS_PCT:
                log.warning(f"[{ticker}] Stop-loss triggered at {price:.2f}")
                self._execute(timestamp, ticker, Signal.SELL, price, "stop_loss")

        # 2. Drawdown halt
        nav = self._compute_nav(prices)
        self._peak_nav = max(self._peak_nav, nav)
        drawdown = (self._peak_nav - nav) / self._peak_nav
        if drawdown > MAX_PORTFOLIO_DRAWDOWN:
            log.warning(f"Portfolio drawdown {drawdown:.1%} exceeds limit — halting all trading.")
            self._liquidate_all(timestamp, prices)
            self._halted = True
            self._record_nav(timestamp, prices)
            return

        # 3. Execute signals
        for ticker, signal in signals.items():
            if ticker not in prices:
                continue
            price = prices[ticker]
            vol   = (volatilities or {}).get(ticker, 0.20)
            self._execute(timestamp, ticker, signal, price, "signal", nav=nav, vol=vol)

        # 4. Record NAV
        self._record_nav(timestamp, prices)

    # ── Order execution ───────────────────────

    def _execute(
        self,
        timestamp: datetime,
        ticker:    str,
        signal:    Signal,
        price:     float,
        reason:    str,
        nav:       float = 0.0,
        vol:       float = 0.20,
    ):
        pos = self.positions[ticker]
        nav = nav or self.cash + sum(
            p.market_value(price) for p in self.positions.values()
        )

        if signal == Signal.BUY and pos.is_flat:
            target_value = self._target_position_value(ticker, nav, vol)
            if target_value < price:             # can't afford one share
                return
            shares = target_value / price
            cost   = shares * price
            fee    = self.fees.total_cost(cost)
            total  = cost + fee
            if total > self.cash:
                shares = (self.cash - fee) / price
                if shares < 1e-6:
                    return
                cost  = shares * price
                fee   = self.fees.total_cost(cost)
                total = cost + fee

            self.cash      -= total
            pos.shares      = shares
            pos.avg_cost    = price
            pos.entry_date  = timestamp
            self._record_trade(timestamp, ticker, "BUY", shares, price, fee, reason)
            log.info(f"[{ticker}] BUY  {shares:.4f} @ {price:.2f}  fee={fee:.2f}  cash={self.cash:.2f}")

        elif signal == Signal.SELL and not pos.is_flat:
            shares   = pos.shares
            proceeds = shares * price
            fee      = self.fees.total_cost(proceeds)
            net      = proceeds - fee

            self.cash    += net
            pos.shares    = 0.0
            pos.avg_cost  = 0.0
            self._record_trade(timestamp, ticker, "SELL", shares, price, fee, reason)
            log.info(f"[{ticker}] SELL {shares:.4f} @ {price:.2f}  fee={fee:.2f}  cash={self.cash:.2f}")

    # ── Position sizing ───────────────────────

    def _target_position_value(self, ticker: str, nav: float, vol: float) -> float:
        max_value = nav * MAX_POSITION_PCT

        if self.sizing == "equal":
            n_active = max(1, len([p for p in self.positions.values() if not p.is_flat]) + 1)
            return min(max_value, nav / n_active)

        elif self.sizing == "vol_target":
            # Scale position inversely to volatility so each position
            # contributes equal risk.
            if vol < 1e-4:
                return max_value
            raw = (VOL_TARGET / vol) * nav / len(self.universe)
            return min(max_value, raw)

        elif self.sizing == "kelly":
            # Simplified fractional Kelly: use predicted_return as win-prob proxy.
            # Full Kelly requires win-rate and avg win/loss; use fraction as a
            # conservative approximation.
            return min(max_value, KELLY_FRACTION * nav / len(self.universe))

        return min(max_value, nav / len(self.universe))

    # ── Liquidation ───────────────────────────

    def _liquidate_all(self, timestamp: datetime, prices: Dict[str, float]):
        for ticker, pos in self.positions.items():
            if not pos.is_flat and ticker in prices:
                self._execute(timestamp, ticker, Signal.SELL, prices[ticker], "liquidation")

    # ── NAV computation ───────────────────────

    def _compute_nav(self, prices: Dict[str, float]) -> float:
        equity = sum(
            pos.market_value(prices[ticker])
            for ticker, pos in self.positions.items()
            if ticker in prices
        )
        return self.cash + equity

    def _record_nav(self, timestamp: datetime, prices: Dict[str, float]):
        self.nav_history.append((timestamp, self._compute_nav(prices)))

    def _record_trade(self, *args):
        self.trades.append(Trade(*args))

    # ── Reporting ─────────────────────────────

    def report(self, prices: Optional[Dict[str, float]] = None) -> dict:
        """
        On-demand report: all key portfolio metrics.
        Pass current prices to get live unrealised P&L.
        """
        nav_series = pd.Series(
            [v for _, v in self.nav_history],
            index=[t for t, _ in self.nav_history],
        )
        nav_now = nav_series.iloc[-1] if len(nav_series) else self.initial_cash

        total_return    = (nav_now - self.initial_cash) / self.initial_cash
        returns         = nav_series.pct_change().dropna()
        sharpe          = _sharpe(returns)
        max_dd          = _max_drawdown(nav_series)
        total_fees      = sum(t.fee for t in self.trades)
        n_trades        = len(self.trades)

        positions_out = {}
        for ticker, pos in self.positions.items():
            price = (prices or {}).get(ticker, pos.avg_cost)
            positions_out[ticker] = {
                "shares":         round(pos.shares, 4),
                "avg_cost":       round(pos.avg_cost, 4),
                "market_value":   round(pos.market_value(price), 2),
                "unrealised_pnl": round(pos.unrealised_pnl(price), 2),
                "unrealised_pct": round(pos.unrealised_pnl_pct(price) * 100, 2),
            }

        return {
            "nav":            round(nav_now, 2),
            "cash":           round(self.cash, 2),
            "total_return_pct": round(total_return * 100, 2),
            "gain_loss":      round(nav_now - self.initial_cash, 2),
            "sharpe_ratio":   round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_trades":       n_trades,
            "total_fees":     round(total_fees, 2),
            "halted":         self._halted,
            "positions":      positions_out,
            "nav_history":    [(str(t.date()), round(v, 2)) for t, v in self.nav_history],
        }


# ─────────────────────────────────────────────
# Risk metrics
# ─────────────────────────────────────────────

def _sharpe(returns: pd.Series, rf: float = 0.05, periods: int = 252) -> float:
    if len(returns) < 2 or returns.std() < 1e-9:
        return 0.0
    excess = returns.mean() * periods - rf
    vol    = returns.std() * np.sqrt(periods)
    return excess / vol


def _max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    roll_max = nav.cummax()
    dd       = (nav - roll_max) / roll_max
    return float(dd.min()) * -1   # return as positive number
