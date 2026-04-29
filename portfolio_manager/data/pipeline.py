"""
Data pipeline: download, clean and feature-engineer OHLCV data.

Works for any list of tickers — single-asset or full universe.
All features are computed per-ticker and returned in a dict keyed by ticker.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    DATA_INTERVAL, FEATURE_WINDOW, HISTORY_YEARS,
    TRAIN_RATIO, VAL_RATIO,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────

def fetch_ohlcv(
    tickers: List[str],
    interval: str = DATA_INTERVAL,
    years: int = HISTORY_YEARS,
) -> Dict[str, pd.DataFrame]:
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    result: Dict[str, pd.DataFrame] = {}
    cols = ["Open", "High", "Low", "Close", "Volume"]

    for ticker in tickers:
        try:
            # Handle MultiIndex columns (newer yfinance always returns these)
            if isinstance(raw.columns, pd.MultiIndex):
                # Try (field, ticker) layout first
                if ticker in raw.columns.get_level_values(1):
                    df = raw.xs(ticker, axis=1, level=1)[cols].copy()
                # Then (ticker, field) layout
                elif ticker in raw.columns.get_level_values(0):
                    df = raw.xs(ticker, axis=1, level=0)[cols].copy()
                else:
                    # Single ticker: just drop whichever level has one unique value
                    flat = raw.copy()
                    flat.columns = flat.columns.droplevel(
                        0 if len(raw.columns.get_level_values(0).unique()) == 1 else 1
                    )
                    df = flat[cols].copy()
            else:
                df = raw[cols].copy()

            df = df.dropna()
            if len(df) < FEATURE_WINDOW * 2:
                log.warning(f"{ticker}: insufficient data ({len(df)} rows), skipping.")
                continue
            result[ticker] = df
            log.info(f"{ticker}: {len(df)} rows downloaded.")
        except Exception as e:
            log.warning(f"{ticker}: failed to extract — {e}")

    return result


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a single-ticker OHLCV DataFrame.
    All NaN rows are dropped at the end so the LSTM never sees gaps.
    """
    d = df.copy()

    # Returns
    d["return_1d"]  = d["Close"].pct_change(1)
    d["return_5d"]  = d["Close"].pct_change(5)
    d["return_20d"] = d["Close"].pct_change(20)

    # Moving averages
    d["ma_10"]  = d["Close"].rolling(10).mean()
    d["ma_30"]  = d["Close"].rolling(30).mean()
    d["ma_50"]  = d["Close"].rolling(50).mean()
    d["ma_200"] = d["Close"].rolling(200).mean()

    # MA ratios (cross signals)
    d["ma10_30_ratio"] = d["ma_10"] / d["ma_30"]
    d["price_ma50_ratio"] = d["Close"] / d["ma_50"]

    # Volatility
    d["volatility_20"] = d["return_1d"].rolling(20).std()
    d["volatility_60"] = d["return_1d"].rolling(60).std()

    # RSI (14)
    d["rsi_14"] = _rsi(d["Close"], 14)

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # Bollinger Bands (20, 2σ)
    bb_mid             = d["Close"].rolling(20).mean()
    bb_std             = d["Close"].rolling(20).std()
    d["bb_upper"]      = bb_mid + 2 * bb_std
    d["bb_lower"]      = bb_mid - 2 * bb_std
    d["bb_position"]   = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    # ATR (14) — average true range
    d["atr_14"] = _atr(d, 14)

    # Volume features
    d["volume_ma_20"]   = d["Volume"].rolling(20).mean()
    d["volume_ratio"]   = d["Volume"] / (d["volume_ma_20"] + 1e-9)
    d["log_volume"]     = np.log1p(d["Volume"])

    # Log price (stationary-ish)
    d["log_close"] = np.log(d["Close"])

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────
# Scaling & windowing
# ─────────────────────────────────────────────

# Features actually used as LSTM inputs
FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "ma10_30_ratio", "price_ma50_ratio",
    "volatility_20", "volatility_60",
    "rsi_14",
    "macd_hist",
    "bb_position",
    "atr_14",
    "volume_ratio", "log_volume",
    "log_close",
]

TARGET_COL = "return_1d"   # what the LSTM predicts (next-day return)


def build_sequences(
    df: pd.DataFrame,
    window: int = FEATURE_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Create sliding windows of shape (N, window, n_features).
    Target is the next-day return (shifted by 1).
    Returns X, y, and the fitted scaler (needed for inference).
    """
    data = df[FEATURE_COLS].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(data.values)

    # Target: next-bar return (index of TARGET_COL in FEATURE_COLS)
    target_idx = FEATURE_COLS.index(TARGET_COL)

    X, y = [], []
    for i in range(window, len(scaled) - 1):
        X.append(scaled[i - window: i])
        y.append(scaled[i + 1, target_idx])   # predict *next* bar's return

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> Tuple:
    n      = len(X)
    n_tr   = int(n * train_ratio)
    n_val  = int(n * val_ratio)

    X_tr, y_tr   = X[:n_tr],         y[:n_tr]
    X_val, y_val = X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val]
    X_te, y_te   = X[n_tr+n_val:],   y[n_tr+n_val:]

    log.info(f"Split → train:{len(X_tr)}  val:{len(X_val)}  test:{len(X_te)}")
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ─────────────────────────────────────────────
# Convenience: full pipeline for one ticker
# ─────────────────────────────────────────────

def prepare_ticker(
    ticker: str,
    raw_df: pd.DataFrame,
    window: int = FEATURE_WINDOW,
) -> dict:
    """
    Run feature engineering + sequencing for a single ticker.
    Returns a dict ready to hand to the model trainer.
    """
    df_feat = add_features(raw_df)
    X, y, scaler = build_sequences(df_feat, window)
    splits = train_val_test_split(X, y)
    return {
        "ticker":  ticker,
        "df":      df_feat,
        "X":       X,
        "y":       y,
        "scaler":  scaler,
        "splits":  splits,
        "n_features": X.shape[2],
    }
