"""
Synthetic data generation.

Geometric Brownian Motion (GBM) and Heston stochastic-vol model.
Use to augment training data or stress-test the model with rare scenarios.

Usage:
    from utils.synthetic import generate_gbm, generate_heston
    df = generate_gbm(S0=150, mu=0.08, sigma=0.20, T=2, n_paths=10)
    # df has columns: [Open, High, Low, Close, Volume] — same schema as yfinance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


# ─────────────────────────────────────────────
# GBM
# ─────────────────────────────────────────────

def generate_gbm(
    S0:      float = 100.0,   # starting price
    mu:      float = 0.08,    # annual drift
    sigma:   float = 0.20,    # annual volatility
    T:       float = 1.0,     # years
    n_paths: int   = 1,       # number of independent paths
    dt:      float = 1/252,   # time step (1 trading day)
    seed:    Optional[int] = 42,
) -> pd.DataFrame:
    """
    Simulate GBM paths and return a DataFrame with OHLCV columns.
    If n_paths > 1 only the first path is returned as OHLCV; the raw
    matrix is available via the 'raw_paths' attribute of the returned df.
    """
    rng   = np.random.default_rng(seed)
    n     = int(T / dt)
    steps = rng.standard_normal((n_paths, n))

    # S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * steps
    paths       = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths       = np.hstack([np.full((n_paths, 1), S0), paths])   # prepend S0

    df = _paths_to_ohlcv(paths[0], dt=dt)
    df.attrs["raw_paths"] = paths   # all paths accessible via df.attrs
    return df


# ─────────────────────────────────────────────
# Heston stochastic volatility
# ─────────────────────────────────────────────

def generate_heston(
    S0:      float = 100.0,
    v0:      float = 0.04,    # initial variance (vol^2)
    mu:      float = 0.08,    # drift
    kappa:   float = 2.0,     # mean-reversion speed
    theta:   float = 0.04,    # long-run variance
    xi:      float = 0.3,     # vol-of-vol
    rho:     float = -0.7,    # price-vol correlation
    T:       float = 1.0,
    dt:      float = 1/252,
    seed:    Optional[int] = 42,
) -> pd.DataFrame:
    """
    Euler-Maruyama discretisation of the Heston model.
    Returns OHLCV DataFrame with realistic volatility clustering.
    """
    rng  = np.random.default_rng(seed)
    n    = int(T / dt)
    S    = np.zeros(n + 1)
    v    = np.zeros(n + 1)
    S[0] = S0
    v[0] = v0

    corr_mat = np.array([[1, rho], [rho, 1]])
    L        = np.linalg.cholesky(corr_mat)

    for i in range(n):
        z     = rng.standard_normal(2)
        z_cor = L @ z
        z_s, z_v = z_cor

        v_pos   = max(v[i], 0)
        dv      = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z_v
        v[i+1]  = v[i] + dv

        dS      = mu * S[i] * dt + np.sqrt(v_pos * dt) * S[i] * z_s
        S[i+1]  = S[i] + dS

    return _paths_to_ohlcv(S, dt=dt)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _paths_to_ohlcv(
    prices: np.ndarray,
    dt: float = 1/252,
    base_volume: float = 1_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Convert a 1-D price path to an OHLCV DataFrame.
    High/Low are simulated with intra-bar noise; Volume is log-normal.
    """
    rng    = np.random.default_rng(seed)
    n      = len(prices) - 1
    opens  = prices[:-1]
    closes = prices[1:]

    noise  = np.abs(rng.standard_normal(n)) * np.abs(closes - opens) * 0.5
    highs  = np.maximum(opens, closes) + noise
    lows   = np.minimum(opens, closes) - noise

    volumes = (base_volume * np.exp(rng.normal(0, 0.5, n))).astype(int)

    start_date = datetime(2020, 1, 2)
    dates = pd.bdate_range(start=start_date, periods=n)   # business days

    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=dates)


# ─────────────────────────────────────────────
# Augmentation helper
# ─────────────────────────────────────────────

def augment_with_synthetic(
    real_df:   pd.DataFrame,
    n_paths:   int   = 5,
    sigma_mul: float = 1.0,   # scale synthetic vol relative to real
) -> pd.DataFrame:
    """
    Fit a GBM to real OHLCV data, generate n_paths synthetic paths,
    and append them (with artificial date offsets) to the real DataFrame.
    Use to increase training set size for rare volatility regimes.
    """
    returns = real_df["Close"].pct_change().dropna()
    mu_hat  = returns.mean() * 252
    sig_hat = returns.std()  * np.sqrt(252) * sigma_mul
    S0      = float(real_df["Close"].iloc[-1])
    T       = len(real_df) / 252

    dfs = [real_df]
    for i in range(n_paths):
        syn = generate_gbm(S0=S0, mu=mu_hat, sigma=sig_hat, T=T, seed=i)
        # Offset dates so they don't clash with real dates
        offset = pd.tseries.offsets.BDay(int(len(real_df) * (i + 1)))
        syn.index = syn.index + offset
        syn.attrs = {}  # clear attrs to avoid concat conflict
        dfs.append(syn)

    real_df = real_df.copy()
    real_df.attrs = {}
    return pd.concat(dfs).sort_index()
