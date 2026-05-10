"""
Unit tests — run with: python -m pytest tests/test_portfolio.py -v (Requires pytest to be installed in your environment: pip install pytest)
"""

import sys
sys.path.insert(0, "..")

from datetime import datetime

import numpy as np
import pytest

from config.settings import FEES_RETAIL, FEES_ZERO, FeeSchedule
from portfolio.engine import Portfolio, _max_drawdown, _sharpe
from strategies.strategy import Signal
from utils.synthetic import generate_gbm, generate_heston, augment_with_synthetic


# ─────────────────────────────────────────────
# FeeSchedule
# ─────────────────────────────────────────────

def test_zero_fees():
    assert FEES_ZERO.total_cost(10_000) == 0.0


def test_retail_fees():
    cost = FEES_RETAIL.total_cost(10_000)
    assert cost > 5.0     # flat commission + pct


def test_fee_round_trip():
    fs = FeeSchedule(commission_pct=0.001, spread_pct=0.0005, slippage_pct=0.0005)
    cost = fs.total_cost(100_000)
    assert abs(cost - 200.0) < 0.01


# ─────────────────────────────────────────────
# Portfolio engine
# ─────────────────────────────────────────────

def _make_portfolio(tickers=None, cash=100_000):
    return Portfolio(tickers or ["ZC=F"], initial_cash=cash, fee_schedule=FEES_ZERO)


def test_initial_nav():
    pf = _make_portfolio()
    assert pf._compute_nav({"ZC=F": 150.0}) == 100_000.0


def test_buy_increases_position():
    pf  = _make_portfolio()
    ts  = datetime(2024, 1, 2)
    pf.step(ts, {"ZC=F": 150.0}, {"ZC=F": Signal.BUY})
    assert pf.positions["ZC=F"].shares > 0


def test_sell_closes_position():
    pf = _make_portfolio()
    ts = datetime(2024, 1, 2)
    pf.step(ts, {"ZC=F": 150.0}, {"ZC=F": Signal.BUY})
    pf.step(datetime(2024, 1, 3), {"ZC=F": 160.0}, {"ZC=F": Signal.SELL})
    assert pf.positions["ZC=F"].shares == 0.0


def test_fees_reduce_cash():
    pf_free = _make_portfolio(cash=100_000)
    pf_paid = Portfolio(["ZC=F"], initial_cash=100_000, fee_schedule=FEES_RETAIL)
    ts = datetime(2024, 1, 2)
    pf_free.step(ts, {"ZC=F": 150.0}, {"ZC=F": Signal.BUY})
    pf_paid.step(ts, {"ZC=F": 150.0}, {"ZC=F": Signal.BUY})
    # Portfolio with fees should have less cash after buying
    assert pf_paid.cash < pf_free.cash


def test_stop_loss_closes_position():
    pf = _make_portfolio()
    ts = datetime(2024, 1, 2)
    pf.step(ts, {"ZC=F": 100.0}, {"ZC=F": Signal.BUY})
    # Price drops 10% — exceeds 5% stop-loss
    pf.step(datetime(2024, 1, 3), {"ZC=F": 90.0}, {"ZC=F": Signal.HOLD})
    assert pf.positions["ZC=F"].shares == 0.0


def test_multi_asset():
    pf = _make_portfolio(["ZC=F", "ZS=F"])
    ts = datetime(2024, 1, 2)
    pf.step(ts, {"ZC=F": 150.0, "ZS=F": 300.0},
            {"ZC=F": Signal.BUY, "ZS=F": Signal.BUY})
    assert pf.positions["ZC=F"].shares > 0
    assert pf.positions["ZS=F"].shares > 0


def test_report_keys():
    pf = _make_portfolio()
    pf.step(datetime(2024, 1, 2), {"ZC=F": 150.0}, {"ZC=F": Signal.BUY})
    r = pf.report({"ZC=F": 155.0})
    for key in ["nav", "cash", "total_return_pct", "sharpe_ratio",
                "max_drawdown_pct", "n_trades", "total_fees", "positions"]:
        assert key in r


# ─────────────────────────────────────────────
# Risk metrics
# ─────────────────────────────────────────────

def test_sharpe_flat():
    import pandas as pd
    returns = pd.Series([0.0] * 252)
    assert _sharpe(returns) == 0.0


def test_max_drawdown():
    import pandas as pd
    nav = pd.Series([100, 110, 90, 95, 80, 100])
    dd  = _max_drawdown(nav)
    assert abs(dd - (110 - 80) / 110) < 0.01


# ─────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────

def test_gbm_shape():
    df = generate_gbm(S0=100, T=1.0, n_paths=3)
    assert set(df.columns) >= {"Open", "High", "Low", "Close", "Volume"}
    assert len(df) == 252


def test_gbm_prices_positive():
    df = generate_gbm(S0=100, mu=0.1, sigma=0.3, T=1.0)
    assert (df["Close"] > 0).all()


def test_heston_shape():
    df = generate_heston(S0=100, T=1.0)
    assert len(df) == 252
    assert "Close" in df.columns


def test_augment():
    df_real = generate_gbm(S0=100, T=2.0, seed=0)
    df_aug  = augment_with_synthetic(df_real, n_paths=2)
    assert len(df_aug) > len(df_real)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
