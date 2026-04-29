# AI Portfolio Manager

An end-to-end paper-trading system with LSTM price prediction, Claude-powered
sentiment analysis, and pluggable trading strategies.

---

## Project structure

```
portfolio_manager/
│
├── config/
│   └── settings.py          ← ALL tuneable parameters live here
│
├── data/
│   └── pipeline.py          ← OHLCV download, feature engineering, windowing
│
├── models/
│   ├── lstm_model.py        ← LSTM definition, training, inference
│   └── sentiment.py         ← Claude API sentiment scoring
│
├── strategies/
│   └── strategy.py          ← Momentum / MeanReversion / Ensemble
│
├── portfolio/
│   ├── engine.py            ← Position tracking, NAV, fees, risk guardrails
│   └── backtester.py        ← Walk-forward backtest driver
│
├── utils/
│   ├── reporter.py          ← On-demand portfolio report
│   └── synthetic.py         ← GBM / Heston synthetic data generation
│
├── tests/
│   └── test_portfolio.py    ← Unit tests (pytest)
│
├── live_trader.py           ← Live paper trading loop
├── main.py                  ← CLI entry point (backtest + report)
└── requirements.txt
```

---

## Recommended workflow

```
1. Backtest  →  trains & saves model weights
2. Live trade →  loads those weights, fetches real prices, manages portfolio
3. Report    →  check portfolio state at any time
```

You must run a backtest before live trading — the live trader loads the
model weights that the backtester saves to `saved_models/`.

---

## Step 1 — Install

```bash
cd portfolio_manager
pip install -r requirements.txt
```

For GPU training (NVIDIA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check if your GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Step 2 — Train (backtest)

Always do this first. It downloads historical data, trains the LSTM, and
saves weights to `saved_models/<TICKER>.pt`.

```bash
# Single asset (fast, good for testing)
python main.py backtest

# Specific tickers
python main.py backtest --universe AAPL MSFT GOOGL NVDA

# Full 12-ticker universe (recommended before live trading)
python main.py multi

# With synthetic GBM augmentation (more training data, better generalisation)
python main.py backtest --augment --augment-paths 5 --retrain

# Force retrain even if saved weights already exist
python main.py backtest --retrain
```

After training you will see a report printed and results saved to `last_backtest.json`.

---

## Step 3 — Live paper trading

The live trader fetches real prices on a schedule, runs the LSTM and strategy,
and manages a paper portfolio. No real orders are sent anywhere.

### Basic usage

```bash
# Single asset (AAPL), tick every 5 minutes
python live_trader.py

# Print a full portfolio report after every tick
python live_trader.py --report

# Full multi-asset universe, tick every minute
python live_trader.py --multi --interval 60 --report
```

### All live trader options

| Flag | Default | Description |
|---|---|---|
| `--universe AAPL MSFT ...` | `AAPL` | Tickers to trade |
| `--multi` | off | Use full 12-ticker universe |
| `--interval 60` | `300` | Seconds between ticks |
| `--capital 100000` | `100000` | Starting cash (USD) |
| `--strategy` | `momentum` | `momentum` / `mean_reversion` / `ensemble` |
| `--sizing` | `equal` | `equal` / `kelly` / `vol_target` |
| `--fees` | `zero` | `zero` / `alpaca` / `ibkr` / `retail` |
| `--sentiment` | off | Enable Claude news sentiment (needs API key) |
| `--report` | off | Print full report each tick |

### Recommended live trading commands

```bash
# Standard: multi-asset, 1 minute ticks, ensemble strategy, Alpaca fees
python live_trader.py --multi --interval 60 --strategy ensemble --fees alpaca --report

# Conservative: momentum only, equal sizing, no fees
python live_trader.py --multi --interval 300 --strategy momentum

# Aggressive: ensemble, Kelly sizing, 1 minute ticks
python live_trader.py --multi --interval 60 --strategy ensemble --sizing kelly --report

# With Claude sentiment analysis
set ANTHROPIC_API_KEY=sk-ant-...
python live_trader.py --multi --interval 60 --sentiment --report
```

### Check the portfolio without stopping the trader

Open a second terminal in the same folder and run:

```bash
python main.py report
```

### What you see each tick (with --report)

```
  Portfolio value   :  $  102,341.50
  Cash              :  $   18,204.10
  Total return      :  +2.34%
  Gain / Loss       :  $   +2,341.50

  RISK METRICS
  Sharpe ratio      :     1.234
  Max drawdown      :    3.21%
  Trading halted    :  no

  TRADING ACTIVITY
  Total trades      :        14
  Total fees paid   :  $       0.00

  OPEN POSITIONS
  AAPL      166.2300 sh  @   150.20  MV $   24,990.05  P&L +3.41%
  MSFT       58.1200 sh  @   415.80  MV $   24,801.34  P&L +1.22%
  NVDA       34.5600 sh  @   875.40  MV $   25,103.22  P&L +2.87%
```

### Outside market hours

The live trader still runs outside 9:30am–4pm ET — it uses the last
known close price. Signals will be flat or slow-moving until the market
opens. This is expected behaviour and does no harm.

### Stopping the live trader

Press `Ctrl+C`. It will print a final portfolio report before exiting.

---

## Key parameters (`config/settings.py`)

All behaviour is controlled from one file. The most important ones:

| Parameter | Default | Description |
|---|---|---|
| `USE_GPU` | `True` | Use GPU for training if available |
| `DEFAULT_UNIVERSE` | `["AAPL"]` | Tickers for single-asset mode |
| `MULTI_ASSET_UNIVERSE` | 12 tickers | Full universe used by `--multi` |
| `INITIAL_CAPITAL` | `100,000` | Starting cash (USD) |
| `STRATEGY` | `"momentum"` | Default strategy |
| `SIZING_METHOD` | `"equal"` | Default position sizing |
| `BUY_THRESHOLD` | `0.005` | Min predicted return to trigger BUY |
| `SELL_THRESHOLD` | `-0.005` | Max predicted return to trigger SELL |
| `MAX_POSITION_PCT` | `0.20` | Max 20% of NAV in any single position |
| `MAX_PORTFOLIO_DRAWDOWN` | `0.15` | Halt all trading at 15% drawdown |
| `STOP_LOSS_PCT` | `0.05` | Per-position stop-loss at 5% |
| `SENTIMENT_WEIGHT` | `0.20` | How much sentiment blends into signal |
| `ACTIVE_FEE_SCHEDULE` | `FEES_ZERO` | Default fee preset |

> **If the live trader takes no positions:** lower `BUY_THRESHOLD` to `0.001`
> and `SELL_THRESHOLD` to `-0.001` in `settings.py`. The raw LSTM predictions
> are small numbers and the default thresholds may be too tight for your model.

---

## Fee schedules

| Preset | Commission | Spread | Slippage | When to use |
|---|---|---|---|---|
| `FEES_ZERO` | 0 | 0 | 0 | Pure paper trading, no cost simulation |
| `FEES_ALPACA` | 0 | 0.05% | 0 | Simulating Alpaca free tier |
| `FEES_IBKR_TIERED` | 0.05% | 0.03% | 0.03% | Simulating IBKR tiered pricing |
| `FEES_RETAIL` | $5 flat | 0.1% | 0.1% | Simulating traditional broker |

Pass via `--fees alpaca` etc. or change `ACTIVE_FEE_SCHEDULE` in `settings.py`.

---

## Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `momentum` | LSTM prediction + fast/slow MA crossover must agree | Trending markets |
| `mean_reversion` | Buys when z-score drops below threshold, exits on reversion | Range-bound markets |
| `ensemble` | Majority vote across momentum + mean_reversion + LSTM + sentiment | General use |

Ensemble is the recommended default for live trading — it requires 3-of-4
signals to agree before acting, which reduces over-trading.

---

## Position sizing

| Method | How it works | Best for |
|---|---|---|
| `equal` | Splits NAV equally across all positions | Simple, stable |
| `vol_target` | Sizes inversely to volatility so each position contributes equal risk | Multi-asset |
| `kelly` | Fractional Kelly (25%) based on predicted return | Higher risk/reward |

`vol_target` is recommended for multi-asset live trading.

---

## Synthetic data augmentation

Train on more data by generating GBM paths fitted to each ticker's real
historical returns and volatility:

```bash
python main.py backtest --augment --augment-paths 10 --retrain
```

More paths = more training data = better generalisation to unseen market
conditions. 5–10 paths is a good starting point. Use `--retrain` to ensure
the model is retrained from scratch on the augmented dataset.

---

## Adding a new strategy

```python
# strategies/strategy.py

class MyStrategy(BaseStrategy):
    def generate(self, ctx: SignalContext) -> Signal:
        # ctx.predicted_return  — LSTM output
        # ctx.sentiment_score   — Claude sentiment [-1, +1]
        # ctx.df                — full feature DataFrame
        # ctx.current_position  — shares currently held
        # ctx.current_price     — latest price
        if ctx.predicted_return > 0.005 and ctx.sentiment_score > 0.3:
            return Signal.BUY
        return Signal.HOLD

STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

Then use it with `--strategy my_strategy` in both `main.py` and `live_trader.py`.

---

## Run tests

```bash
# From inside the portfolio_manager folder
set PYTHONPATH=.
python -m pytest tests/ -v
```

All 16 tests should pass.

---

## Connecting a real broker (future)

The portfolio engine is designed so that switching from paper to live is
one function change. In `portfolio/engine.py`, find `_execute()` and replace
the cash/shares accounting with a real order call:

```python
# Alpaca example
import alpaca_trade_api as tradeapi
api = tradeapi.REST(KEY, SECRET, BASE_URL)
api.submit_order(symbol=ticker, qty=shares, side='buy', type='market', time_in_force='day')

# IBKR example (ib_insync)
from ib_insync import *
ib = IB(); ib.connect('127.0.0.1', 7497, clientId=1)
ib.placeOrder(contract, MarketOrder('BUY', shares))
```

All fee tracking, position sizing, stop-losses, and drawdown guardrails
remain unchanged — they sit above the execution layer.
