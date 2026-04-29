"""
Trading strategies.

Each strategy consumes a SignalContext and returns a Signal in {BUY, SELL, HOLD}.
Adding a new strategy = subclassing BaseStrategy and registering it in STRATEGY_REGISTRY.

Signal blending:
    final_signal = α * model_signal + (1-α) * sentiment_signal
where α = 1 - SENTIMENT_WEIGHT from settings.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import (
    BUY_THRESHOLD, MR_LOOKBACK, MR_ZSCORE_ENTRY, MR_ZSCORE_EXIT,
    MOMENTUM_FAST, MOMENTUM_SLOW, SELL_THRESHOLD, SENTIMENT_WEIGHT,
)

log = logging.getLogger(__name__)


class Signal(Enum):
    BUY  = 1
    HOLD = 0
    SELL = -1


@dataclass
class SignalContext:
    """Everything a strategy needs to produce a signal for one ticker."""
    ticker:           str
    df:               pd.DataFrame     # feature-engineered OHLCV + indicators
    predicted_return: float            # LSTM output (scaled, ~[-1,1])
    sentiment_score:  float            # [-1, +1] from Claude / 0 if disabled
    current_position: float            # current position size (shares or USD)
    current_price:    float


# ─────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────

class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, ctx: SignalContext) -> Signal:
        ...

    def _blend_signal(self, model_raw: float, sentiment: float) -> float:
        """Blend LSTM prediction with sentiment score."""
        w = SENTIMENT_WEIGHT
        return (1 - w) * model_raw + w * sentiment


# ─────────────────────────────────────────────
# Momentum strategy
# ─────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Combines:
      1. LSTM predicted return direction
      2. Fast/slow MA crossover confirmation
      3. Sentiment tilt

    Buys when all three agree on upside; sells when all agree on downside.
    """

    def generate(self, ctx: SignalContext) -> Signal:
        df    = ctx.df
        close = df["Close"]

        if len(close) < MOMENTUM_SLOW:
            return Signal.HOLD

        fast_ma = close.rolling(MOMENTUM_FAST).mean().iloc[-1]
        slow_ma = close.rolling(MOMENTUM_SLOW).mean().iloc[-1]
        ma_signal = 1.0 if fast_ma > slow_ma else -1.0

        blended = self._blend_signal(ctx.predicted_return, ctx.sentiment_score)

        # Both LSTM and MA must agree
        if blended > BUY_THRESHOLD and ma_signal > 0:
            return Signal.BUY
        if blended < SELL_THRESHOLD and ma_signal < 0:
            return Signal.SELL
        return Signal.HOLD


# ─────────────────────────────────────────────
# Mean-reversion strategy
# ─────────────────────────────────────────────

class MeanReversionStrategy(BaseStrategy):
    """
    Z-score based mean reversion on rolling close prices.
    Enters long when price is significantly below its mean (oversold),
    exits when it reverts. Sentiment used to widen/narrow thresholds.
    """

    def generate(self, ctx: SignalContext) -> Signal:
        close = ctx.df["Close"]
        if len(close) < MR_LOOKBACK:
            return Signal.HOLD

        roll  = close.rolling(MR_LOOKBACK)
        mu    = roll.mean().iloc[-1]
        sigma = roll.std().iloc[-1]
        if sigma < 1e-9:
            return Signal.HOLD

        z = (ctx.current_price - mu) / sigma

        # Sentiment widens/narrows entry threshold
        entry_thresh = MR_ZSCORE_ENTRY - 0.3 * ctx.sentiment_score

        in_position = ctx.current_position > 0

        if not in_position and z < entry_thresh:
            return Signal.BUY
        if in_position and z > MR_ZSCORE_EXIT:
            return Signal.SELL
        return Signal.HOLD


# ─────────────────────────────────────────────
# Ensemble strategy
# ─────────────────────────────────────────────

class EnsembleStrategy(BaseStrategy):
    """
    Votes from Momentum + MeanReversion + raw LSTM + sentiment.
    Majority vote → signal.  Requires 3-of-4 to avoid over-trading.
    """

    def __init__(self):
        self._momentum = MomentumStrategy()
        self._mr       = MeanReversionStrategy()

    def generate(self, ctx: SignalContext) -> Signal:
        votes = []
        votes.append(self._momentum.generate(ctx).value)
        votes.append(self._mr.generate(ctx).value)

        # Raw LSTM vote
        lstm_vote = (1 if ctx.predicted_return > BUY_THRESHOLD
                     else -1 if ctx.predicted_return < SELL_THRESHOLD
                     else 0)
        votes.append(lstm_vote)

        # Sentiment vote
        sent_vote = (1 if ctx.sentiment_score > 0.3
                     else -1 if ctx.sentiment_score < -0.3
                     else 0)
        votes.append(sent_vote)

        total = sum(votes)
        if total >= 3:
            return Signal.BUY
        if total <= -3:
            return Signal.SELL
        return Signal.HOLD


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, type] = {
    "momentum":       MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ensemble":       EnsembleStrategy,
}


def get_strategy(name: str) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Choose from {list(STRATEGY_REGISTRY)}")
    return cls()
