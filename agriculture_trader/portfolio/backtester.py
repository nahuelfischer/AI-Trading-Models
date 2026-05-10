"""
Backtester.

Drives the full pipeline over historical data:
  data → features → LSTM signals → strategy → portfolio → report

Designed for single-asset testing (easily extended: pass a full universe).
Everything is time-series safe: the model is trained on the training split
and evaluated only on the held-out test split.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    ACTIVE_FEE_SCHEDULE, DEFAULT_UNIVERSE, FEATURE_WINDOW,
    INITIAL_CAPITAL, SENTIMENT_WEIGHT, SIZING_METHOD, STRATEGY,
    PREDICTION_STAGE, PREDICTION_YEAR
)
from data.pipeline import (
    FEATURE_COLS, add_features, build_sequences,
    fetch_ohlcv, prepare_ticker, train_val_test_split,
)
from models.lstm_model import LSTMPriceModel, load_model, predict, train
from models.sentiment import fetch_headlines, score_headlines
from models.early_season_model import EarlySeasonYieldModel
from portfolio.engine import Portfolio
from strategies.strategy import Signal, SignalContext, get_strategy

log = logging.getLogger(__name__)


class Backtester:
    """
    End-to-end backtesting engine.

    Steps:
      1. Download OHLCV data
      2. Feature engineering
      3. Train LSTM on train split (or load saved weights)
      4. Walk-forward over test split, generating signals bar by bar
      5. Return portfolio report
    """

    def __init__(
        self,
        universe:      List[str]      = None,
        initial_cash:  float          = INITIAL_CAPITAL,
        strategy_name: str            = STRATEGY,
        use_sentiment: bool           = False,   # set True to call Claude API per bar
        retrain:       bool           = False,   # force re-training even if weights exist
        sizing:        str            = SIZING_METHOD,
        augment:       bool           = False,
        augment_paths: int            = 5,
    ):
        self.universe      = universe or DEFAULT_UNIVERSE
        self.strategy      = get_strategy(strategy_name)
        self.use_sentiment = use_sentiment
        self.retrain       = retrain
        self.portfolio     = Portfolio(self.universe, initial_cash, ACTIVE_FEE_SCHEDULE, sizing)
        self.augment       = augment
        self.augment_paths = augment_paths

        # --------------------------------
        # EARLY-SEASON YIELD MODEL
        # --------------------------------
        self.yield_model = EarlySeasonYieldModel()

        print("Loading yield/weather data...")
        self.yield_model.load_data()

        print("Training yield models...")
        self.yield_model.train_all()

        # --------------------------------
        # CURRENT MACRO PREDICTION
        # --------------------------------
        self.yield_signal = self.yield_model.predict_live(
            stage=PREDICTION_STAGE,
            year=PREDICTION_YEAR
        )

        # Per-ticker state populated during setup
        self._models:   Dict[str, LSTMPriceModel] = {}
        self._scalers:  Dict[str, object]          = {}
        self._dfs:      Dict[str, pd.DataFrame]    = {}
        self._test_idx: Dict[str, int]             = {}   # index in df where test starts

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def run(self) -> dict:
        log.info(f"Backtester starting — universe: {self.universe}")

        # 1. Fetch raw data
        raw = fetch_ohlcv(self.universe)

        # 2. Prepare each ticker (features + sequences + splits)
        prepared = {}
        for ticker in self.universe:
            if ticker not in raw:
                log.warning(f"{ticker} skipped (no data).")
                continue

            ticker_df = raw[ticker]

            # Augment with synthetic data if requested
            if self.augment:
                from utils.synthetic import augment_with_synthetic
                original_len = len(ticker_df)
                ticker_df = augment_with_synthetic(ticker_df, n_paths=self.augment_paths)
                log.info(
                    f"{ticker}: augmented {original_len} → {len(ticker_df)} rows "
                    f"({self.augment_paths} GBM paths added)"
                )

            prepared[ticker] = prepare_ticker(ticker, ticker_df)

        # 3. Train / load models
        for ticker, data in prepared.items():
            X_tr, y_tr, X_val, y_val, X_te, y_te = data["splits"]
            model = None
            if not self.retrain:
                model = load_model(ticker, data["n_features"])
            if model is None:
                model = train(ticker, X_tr, y_tr, X_val, y_val, data["n_features"])
            self._models[ticker]  = model
            self._scalers[ticker] = data["scaler"]
            self._dfs[ticker]     = data["df"]

            # Record where the test period starts in the df
            n_total = len(data["X"])
            n_tr    = len(X_tr)
            n_val   = len(X_val)
            self._test_idx[ticker] = FEATURE_WINDOW + n_tr + n_val   # absolute row in df

        # 4. Walk-forward over test split
        self._walk_forward(prepared)

        # 5. Report
        latest_prices = {
            ticker: float(self._dfs[ticker]["Close"].iloc[-1])
            for ticker in self._models
        }
        report = self.portfolio.report(prices=latest_prices)
        log.info(
            f"Backtest complete | NAV: {report['nav']:,.2f} | "
            f"Return: {report['total_return_pct']:.2f}% | "
            f"Sharpe: {report['sharpe_ratio']:.3f} | "
            f"Max DD: {report['max_drawdown_pct']:.2f}%"
        )
        return report

    # ──────────────────────────────────────────
    # Walk-forward loop
    # ──────────────────────────────────────────

    def _walk_forward(self, prepared: dict):
        """
        Iterate bar-by-bar over the test split.
        For each bar:
          - build the feature window
          - run LSTM → predicted_return
          - optionally fetch + score headlines
          - run strategy → signal
          - step portfolio
        """
        # Find the overlapping date range for the test periods
        # (so all tickers are stepped in sync)
        test_dates = self._common_test_dates(prepared)
        log.info(f"Walk-forward over {len(test_dates)} bars.")

        for date in test_dates:
            prices:      Dict[str, float]  = {}
            signals:     Dict[str, Signal] = {}
            volatilities: Dict[str, float] = {}

            for ticker, model in self._models.items():
                df = self._dfs[ticker]
                if date not in df.index:
                    continue

                row_idx = df.index.get_loc(date)
                if row_idx < FEATURE_WINDOW:
                    continue

                # Build window
                scaler  = self._scalers[ticker]
                window_df = df.iloc[row_idx - FEATURE_WINDOW: row_idx][FEATURE_COLS]
                window_scaled = scaler.transform(window_df.values)

                # LSTM prediction
                pred_scaled   = predict(model, window_scaled)
                # Inverse-scale the return prediction
                dummy         = np.zeros((1, len(FEATURE_COLS)))
                ret_idx       = FEATURE_COLS.index("return_1d")
                dummy[0, ret_idx] = pred_scaled
                pred_return   = float(scaler.inverse_transform(dummy)[0, ret_idx])

                # Sentiment
                sentiment = 0.0
                if self.use_sentiment:
                    headlines = fetch_headlines(ticker)
                    sentiment = score_headlines(ticker, headlines)

                # Price + vol
                price = float(df["Close"].iloc[row_idx])
                vol   = float(df["volatility_20"].iloc[row_idx]) * np.sqrt(252)

                # Strategy signal
                ctx = SignalContext(
                    ticker=ticker,
                    df=df.iloc[: row_idx + 1],
                    predicted_return=pred_return,
                    sentiment_score=sentiment,
                    current_position=self.portfolio.positions[ticker].shares,
                    current_price=price,
                    yield_prediction=self.yield_signal["mid"],
                    yield_revision=self.yield_signal["yield_revision"],
                    drought_index=self.yield_signal["drought_index"],
                )
                signal = self.strategy.generate(ctx)

                prices[ticker]       = price
                signals[ticker]      = signal
                volatilities[ticker] = vol

            if prices:
                self.portfolio.step(
                    timestamp=date if isinstance(date, datetime) else datetime.combine(date, datetime.min.time()),
                    prices=prices,
                    signals=signals,
                    volatilities=volatilities,
                )

    def _common_test_dates(self, prepared: dict) -> pd.DatetimeIndex:
        """
        The intersection of test-period dates across all tickers.
        Ensures the portfolio is stepped on dates where all assets have data.
        """
        sets = []
        for ticker, data in prepared.items():
            df       = self._dfs[ticker]
            t_start  = self._test_idx[ticker]
            dates    = df.index[t_start:]
            sets.append(set(dates))

        if not sets:
            return pd.DatetimeIndex([])

        common = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return pd.DatetimeIndex(sorted(common))
