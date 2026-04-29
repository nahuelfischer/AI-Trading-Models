"""
Central configuration for the AI Portfolio Manager.
All tuneable parameters live here — swap values to change behaviour
without touching any logic files.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────
# Single-asset testing: keep one ticker.
# Multi-asset: add more — the engine handles N assets identically.
DEFAULT_UNIVERSE: List[str] = ["AAPL"]

MULTI_ASSET_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC",
    "XOM", "CVX",
    "SPY", "QQQ",
]


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
DATA_INTERVAL   = "1d"          # yfinance interval: 1m,5m,15m,1h,1d,1wk
HISTORY_YEARS   = 5             # years of history to download for training
FEATURE_WINDOW  = 60            # lookback window (bars) fed into LSTM
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# test_ratio = 1 - train - val


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
USE_GPU = True    # set False to force CPU

LSTM_HIDDEN_SIZE  = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.2
LSTM_EPOCHS       = 50
LSTM_BATCH_SIZE   = 32
LSTM_LR           = 1e-3
LSTM_PATIENCE     = 10          # early-stopping patience (epochs)
MODEL_DIR         = "saved_models"


# ──────────────────────────────────────────────
# Sentiment (Claude API)
# ──────────────────────────────────────────────
SENTIMENT_MODEL        = "claude-sonnet-4-20250514"
SENTIMENT_MAX_TOKENS   = 256
NEWS_MAX_HEADLINES     = 10     # headlines fed per sentiment request
SENTIMENT_WEIGHT       = 0.20   # blend weight vs price signal (0–1)


# ──────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────
STRATEGY = "momentum"           # "momentum" | "mean_reversion" | "ensemble"

# Signal thresholds (predicted return vs current price)
BUY_THRESHOLD  =  0.005         #  +0.5 % predicted upside → buy
SELL_THRESHOLD = -0.005         # -0.5 % predicted downside → sell

# Momentum params
MOMENTUM_FAST  = 10             # fast MA window
MOMENTUM_SLOW  = 30             # slow MA window

# Mean-reversion params
MR_ZSCORE_ENTRY  = -1.5        # z-score below which we buy
MR_ZSCORE_EXIT   =  0.5        # z-score above which we exit long
MR_LOOKBACK      = 20


# ──────────────────────────────────────────────
# Portfolio & risk
# ──────────────────────────────────────────────
INITIAL_CAPITAL    = 100_000.0  # USD

# Position sizing
SIZING_METHOD      = "equal"    # "equal" | "kelly" | "vol_target"
MAX_POSITION_PCT   = 0.20       # max 20 % of NAV in any single asset
VOL_TARGET         = 0.10       # annual vol target (vol_target sizing)
KELLY_FRACTION     = 0.25       # fractional Kelly (dampener)

# Risk guardrails
MAX_PORTFOLIO_DRAWDOWN = 0.15   # halt trading if drawdown exceeds 15 %
STOP_LOSS_PCT          = 0.05   # per-position stop-loss at 5 %

# Rebalancing
REBALANCE_FREQUENCY = "daily"   # "daily" | "weekly" | "monthly"


# ──────────────────────────────────────────────
# Trading costs (paper trading)
# ──────────────────────────────────────────────
@dataclass
class FeeSchedule:
    """
    Pluggable fee schedule.  Set all to 0 for zero-cost paper trading.
    Swap in real values to simulate a broker (e.g. Alpaca, IBKR).
    """
    commission_per_trade: float = 0.0      # flat USD per order
    commission_pct: float       = 0.0      # % of trade value (e.g. 0.001 = 0.1 %)
    spread_pct: float           = 0.0005   # bid-ask half-spread (e.g. 0.05 %)
    slippage_pct: float         = 0.0005   # market-impact slippage
    stamp_duty_pct: float       = 0.0      # UK stamp duty etc.

    def total_cost(self, trade_value: float) -> float:
        """Return total round-trip cost in USD for a given trade value."""
        pct_cost = (
            self.commission_pct
            + self.spread_pct
            + self.slippage_pct
            + self.stamp_duty_pct
        ) * trade_value
        return self.commission_per_trade + pct_cost


# Preset fee schedules
FEES_ZERO        = FeeSchedule(spread_pct=0.0, slippage_pct=0.0)         # pure paper
FEES_ALPACA      = FeeSchedule(commission_pct=0.0, spread_pct=0.0005)   # Alpaca (free + spread)
FEES_IBKR_TIERED = FeeSchedule(commission_pct=0.0005, spread_pct=0.0003, slippage_pct=0.0003)
FEES_RETAIL      = FeeSchedule(commission_per_trade=5.0, spread_pct=0.001, slippage_pct=0.001)

ACTIVE_FEE_SCHEDULE = FEES_ZERO   # ← change this to simulate costs


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────
REPORT_CURRENCY = "USD"
LOG_LEVEL       = "INFO"
