#!/usr/bin/env python3
"""
Live paper trader.

Fetches real-time prices on a schedule and runs the full
signal → portfolio pipeline. No orders are sent anywhere —
all positions are tracked in memory and reported on demand.

Usage:
    python live_trader.py                                 # uses last backtest models
    python live_trader.py --universe AAPL MSFT GOOGL
    python live_trader.py --interval 60 --fees alpaca
    python live_trader.py --interval 60 --report          # print report each cycle
"""

import argparse
import logging
import time
from datetime import datetime

from matplotlib import ticker
from matplotlib import ticker
import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    ACTIVE_FEE_SCHEDULE, DEFAULT_UNIVERSE, FEATURE_WINDOW,
    FEES_ALPACA, FEES_IBKR_TIERED, FEES_RETAIL, FEES_ZERO,
    INITIAL_CAPITAL, MULTI_ASSET_UNIVERSE, SIZING_METHOD, STRATEGY,
)
from data.pipeline import FEATURE_COLS, add_features, fetch_ohlcv, prepare_ticker
from models.lstm_model import load_model, predict
from models.sentiment import fetch_headlines, score_headlines
from portfolio.engine import Portfolio
from strategies.strategy import Signal, SignalContext, get_strategy
from utils.reporter import report as print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live")

FEE_PRESETS = {
    "zero": FEES_ZERO, "alpaca": FEES_ALPACA,
    "ibkr": FEES_IBKR_TIERED, "retail": FEES_RETAIL,
}


class LiveTrader:
    def __init__(self, universe, initial_cash, strategy_name,
                 fee_schedule, sizing, use_sentiment):
        self.universe      = universe
        self.strategy      = get_strategy(strategy_name)
        self.use_sentiment = use_sentiment
        self.portfolio     = Portfolio(universe, initial_cash, fee_schedule, sizing)

        self._models  = {}
        self._scalers = {}
        self._dfs     = {}   # rolling history per ticker

    def setup(self):
        """Download history and load saved model weights."""
        log.info("Downloading historical data to build feature windows...")
        raw = fetch_ohlcv(self.universe)

        for ticker in self.universe:
            if ticker not in raw:
                log.warning(f"{ticker}: no data, skipping.")
                continue

            data  = prepare_ticker(ticker, raw[ticker])
            model = load_model(ticker, data["n_features"])

            if model is None:
                log.warning(
                    f"{ticker}: no saved model found. "
                    f"Run 'python main.py backtest --universe {ticker}' first to train."
                )
                continue

            self._models[ticker]  = model
            self._scalers[ticker] = data["scaler"]
            self._dfs[ticker]     = data["df"]   # historical context for features

        if not self._models:
            raise RuntimeError(
                "No models loaded. Train first with: python main.py backtest"
            )
        log.info(f"Ready — {len(self._models)} model(s) loaded: {list(self._models)}")

    def tick(self, print_report_flag: bool = False):
        """One live cycle: fetch prices → signals → portfolio step."""
        now    = datetime.now()
        prices = self._fetch_live_prices()
        if not prices:
            log.warning("No live prices retrieved, skipping tick.")
            return

        signals      = {}
        volatilities = {}

        for ticker, model in self._models.items():
            if ticker not in prices:
                continue

            price = prices[ticker]
            df    = self._dfs[ticker]

            # Append today's live bar so features stay current
            df = self._append_live_bar(df, ticker, price)
            self._dfs[ticker] = df

            if len(df) < FEATURE_WINDOW + 1:
                continue

            # Build feature window
            scaler        = self._scalers[ticker]
            window_df     = df.iloc[-FEATURE_WINDOW:][FEATURE_COLS]
            window_scaled = scaler.transform(window_df.values)

            # LSTM prediction
            pred_scaled = predict(model, window_scaled)
            dummy       = np.zeros((1, len(FEATURE_COLS)))
            ret_idx     = FEATURE_COLS.index("return_1d")
            dummy[0, ret_idx] = pred_scaled
            pred_return = float(scaler.inverse_transform(dummy)[0, ret_idx])

            # Sentiment (optional)
            sentiment = 0.0
            if self.use_sentiment:
                sentiment = score_headlines(ticker, fetch_headlines(ticker))

            vol = float(df["volatility_20"].iloc[-1]) * np.sqrt(252)

            ctx = SignalContext(
                ticker=ticker,
                df=df,
                predicted_return=pred_return,
                sentiment_score=sentiment,
                current_position=self.portfolio.positions[ticker].shares,
                current_price=price,
            )
            signal = self.strategy.generate(ctx)
            signals[ticker]      = signal
            volatilities[ticker] = vol

            log.info(
                f"{ticker:8s}  price={price:>9.2f}  "
                f"pred_ret={pred_return:>+.4f}  "
                f"sentiment={sentiment:>+.2f}  "
                f"signal={signal.name}"
            )

        if signals:
            self.portfolio.step(now, prices, signals, volatilities)

        if print_report_flag:
            print_report(self.portfolio, prices=prices)

    def _fetch_live_prices(self):
        """Get the latest price for each ticker via yfinance."""
        prices = {}
        for ticker in self._models:
            try:
                data = yf.Ticker(ticker).fast_info
                price = data.last_price or data.previous_close
                if price:
                    prices[ticker] = float(price)
            except Exception as e:
                log.warning(f"{ticker}: price fetch failed — {e}")
        return prices

    def _append_live_bar(self, df, ticker, price):
        if df.empty:
            log.warning(f"{ticker}: history DataFrame is empty, skipping append.")
            return df

        last = df.iloc[-1]
        new_row = {
            "Open":   last["Close"],
            "High":   max(last["Close"], price),
            "Low":    min(last["Close"], price),
            "Close":  price,
            "Volume": last["Volume"],
        }
        now = datetime.now().replace(second=0, microsecond=0)

        # Work on the raw OHLCV columns only before re-running add_features
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        raw = df[ohlcv_cols].copy()

        if now not in raw.index:
            raw.loc[now] = new_row

        # Re-run feature engineering on the clean OHLCV base
        enriched = add_features(raw)

        if enriched.empty or len(enriched) < FEATURE_WINDOW:
            log.warning(f"{ticker}: DataFrame too short after feature engineering ({len(enriched)} rows), keeping previous.")
            return df

        return enriched


def main():
    parser = argparse.ArgumentParser(description="Live paper trader")
    parser.add_argument("--universe", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--capital",  type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--strategy", default=STRATEGY,
                        choices=["momentum", "mean_reversion", "ensemble"])
    parser.add_argument("--sizing",   default=SIZING_METHOD,
                        choices=["equal", "kelly", "vol_target"])
    parser.add_argument("--fees",     default="zero", choices=list(FEE_PRESETS))
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between ticks (default: 300 = 5 min)")
    parser.add_argument("--sentiment", action="store_true")
    parser.add_argument("--report",    action="store_true",
                        help="Print full report after every tick")
    parser.add_argument("--multi",     action="store_true",
                        help="Use full multi-asset universe")
    args = parser.parse_args()

    universe = MULTI_ASSET_UNIVERSE if args.multi else args.universe

    trader = LiveTrader(
        universe=universe,
        initial_cash=args.capital,
        strategy_name=args.strategy,
        fee_schedule=FEE_PRESETS[args.fees],
        sizing=args.sizing,
        use_sentiment=args.sentiment,
    )

    trader.setup()

    log.info(f"Live trading started — tick every {args.interval}s. Ctrl+C to stop.")
    log.info("Tip: run 'python main.py report' in another terminal at any time.")

    try:
        while True:
            trader.tick(print_report_flag=args.report)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
        print_report(trader.portfolio, prices=trader._fetch_live_prices())


if __name__ == "__main__":
    main()