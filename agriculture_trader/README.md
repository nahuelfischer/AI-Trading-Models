# Agriculture Trading Bot

An end-to-end agricultural paper-trading system combining:

* LSTM price prediction
* Weather-driven corn yield forecasting
* XGBoost early-season anomaly models
* Optional Claude-powered sentiment analysis
* Macro + technical trading strategies
* Live paper trading infrastructure

The system is currently focused on **corn futures (`ZC=F`)**, but the architecture
supports extension to soybeans, wheat, and broader agricultural commodities.

---

# Project structure

```text
agriculture_trader/
│
├── config/
│   └── settings.py              ← ALL tuneable parameters live here
│
├── data/
│   ├── pipeline.py              ← OHLCV download + feature engineering
│   ├── weather_pipeline.py      ← Historical weather/yield dataset builder
│   └── weather_collector.py     ← Open-Meteo live weather collector
│
├── models/
│   ├── lstm_model.py            ← LSTM price prediction model
│   ├── sentiment.py             ← Claude API sentiment scoring
│   ├── yield_model.py           ← Full-year corn yield prediction model
│   └── early_season_model.py    ← Early-season anomaly/revision model
│
├── strategies/
│   └── strategy.py              ← Momentum / MeanReversion / Fundamental / Ensemble
│
├── portfolio/
│   ├── engine.py                ← Portfolio management + execution
│   └── backtester.py            ← Walk-forward backtest engine
│
├── journal/
│   ├── crop_data/               ← Historical USDA yield data
│   ├── weather_data/            ← Historical station weather CSVs
│   ├── live_weather/            ← Continuously updated live weather data
│   └── yield_predictions/       ← Stored historical yield predictions
│
├── utils/
│   ├── reporter.py              ← Portfolio reports
│   └── synthetic.py             ← GBM synthetic data generation
│
├── tests/
│   └── test_portfolio.py
│
├── live_trader.py               ← Live paper trading loop
├── main.py                      ← CLI entry point
└── requirements.txt
```

---

# System architecture

The trading system combines:

```text
LIVE WEATHER
      ↓
Yield Forecast Models
      ↓
Macro Agricultural Signals
      ↓
Trading Strategies
      ↓
Portfolio Engine
```

The architecture blends:

| Layer            | Purpose                             |
| ---------------- | ----------------------------------- |
| LSTM model       | Short-term market prediction        |
| Yield models     | Long-term agricultural fundamentals |
| Sentiment model  | News/event interpretation           |
| Strategies       | Trade decision logic                |
| Portfolio engine | Risk management + execution         |

This creates a hybrid:

```text
macro + technical commodity trading system
```

similar to real agricultural trading desks.

---

# Recommended workflow

```text
1. Build weather/yield dataset
2. Train yield models
3. Train LSTM trading model
4. Backtest strategies
5. Run live paper trading
```

---

# Step 1 — Install

```bash
cd agriculture_trader
pip install -r requirements.txt
```

For GPU training (NVIDIA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check GPU visibility:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# Step 2 — Historical weather data

The yield models require historical weather station data.

Place CSV files inside:

```text
journal/weather_data/
```

Format:

```text
Year;Jan;Feb;Mar;Apr;May;Jun;Jul;Aug;Sep;Oct;Nov;Dec;Annual
1937;M;24.8;36.6;M;63.5;69.4;78.9;M;65.9;51.8;37.2;M;53.5
```

Supported missing values:

| Value | Meaning             |
| ----- | ------------------- |
| `M`   | Missing             |
| `T`   | Trace precipitation |

Naming convention:

```text
des_moines_temp.csv
des_moines_prec.csv

st_louis_temp.csv
st_louis_prec.csv
```
(Currently contains some .csv files of selected weather stations. See weatherdata.md for more info.)
---

# Step 3 — Build training dataset

Run:

```bash
python data/weather_pipeline.py
```

This:

* loads all weather stations
* cleans missing values
* computes regional averages/std
* merges USDA corn yield data
* creates final ML dataset

Output:

```text
journal/crop_data/final_dataset.csv
```

---

# Step 4 — Collect live weather data

The system uses [Open-Meteo](https://open-meteo.com/?utm_source=chatgpt.com) for free weather collection.

Run:

```bash
python data/weather_collector.py
```

This:

* downloads current station weather
* updates `journal/live_weather/`
* stores daily temperature + precipitation
* continuously extends current-year weather history

Only a few API calls/day are needed.

---

# Step 5 — Train yield models

## Full-year yield model

```bash
python models/yield_model.py
```

Outputs:

* expected annual corn yield
* prediction range
* feature importance graph

---

## Early-season anomaly model

```bash
python models/early_season_model.py
```

Outputs:

* April/May/June yield forecasts
* yield revisions
* drought index
* prediction confidence range

Example:

```text
===== EARLY-SEASON FORECAST =====

Stage: april
Trend Yield: 184.1 bu/ac

Predicted Yield Range:
Low : 176.2
Mid : 179.8
High: 183.4

Yield Revision: -2.7
Drought Index: 9.1
```

---

# Why yield revisions matter

The strongest agricultural signal is usually NOT:

```text
absolute yield level
```

but:

```text
change in expected yield
```

Example:

| Situation                    | Market Impact |
| ---------------------------- | ------------- |
| Yield estimate falls sharply | Bullish corn  |
| Yield estimate rises sharply | Bearish corn  |
| Severe drought emerges       | Bullish corn  |
| Excellent weather persists   | Bearish corn  |

The system stores historical predictions inside:

```text
journal/yield_predictions/
```

allowing revision tracking over time.

---

# Step 6 — Train trading model (backtest)

```bash
python main.py backtest
```

This:

* downloads historical futures data
* trains LSTM models
* trains yield models
* runs walk-forward backtest
* evaluates portfolio performance

---

# Recommended backtest commands

```bash
# Basic corn backtest
python main.py backtest

# Force retrain
python main.py backtest --retrain

# Ensemble strategy
python main.py backtest --strategy ensemble

# Fundamental strategy
python main.py backtest --strategy fundamental

# Synthetic augmentation
python main.py backtest --augment --augment-paths 5 --retrain
```

---

# Step 7 — Live paper trading

```bash
python live_trader.py
```

The live trader:

* fetches live futures prices
* updates technical indicators
* loads live macro yield forecast
* generates trading signals
* manages a paper portfolio

No real orders are sent.

---

# Recommended live trading commands

```bash
# Corn futures
python live_trader.py

# Fundamental strategy
python live_trader.py --strategy fundamental --report

# Ensemble strategy
python live_trader.py --strategy ensemble --interval 60 --report

# With sentiment analysis
set ANTHROPIC_API_KEY=sk-ant-...
python live_trader.py --strategy ensemble --sentiment --report
```

---

# Available strategies

| Strategy         | How it works                         | Best for                   |
| ---------------- | ------------------------------------ | -------------------------- |
| `momentum`       | LSTM + MA crossover                  | Trending markets           |
| `mean_reversion` | Z-score reversal strategy            | Range-bound markets        |
| `fundamental`    | Yield revisions + drought conditions | Agricultural macro trading |
| `ensemble`       | Combines all models/signals          | General use                |

---

# Fundamental strategy logic

The agricultural strategy uses:

| Signal          | Bullish Corn     | Bearish Corn     |
| --------------- | ---------------- | ---------------- |
| Yield revision  | Falling          | Rising           |
| Drought index   | High drought     | Low drought      |
| Sentiment       | Bullish ag news  | Bearish ag news  |
| LSTM prediction | Positive returns | Negative returns |

This creates a:

```text
weather-driven macro trading system
```

instead of pure technical analysis.

---

# Key parameters (`config/settings.py`)

| Parameter                | Description                   |
| ------------------------ | ----------------------------- |
| `PREDICTION_YEAR`        | Current crop year             |
| `PREDICTION_STAGE`       | april / may / june etc        |
| `BUY_THRESHOLD`          | Min LSTM prediction for BUY   |
| `SELL_THRESHOLD`         | Max LSTM prediction for SELL  |
| `SENTIMENT_WEIGHT`       | Weight of sentiment in signal |
| `MAX_POSITION_PCT`       | Max portfolio allocation      |
| `STOP_LOSS_PCT`          | Per-position stop loss        |
| `MAX_PORTFOLIO_DRAWDOWN` | Global risk halt              |

---

# Important modeling notes

## 1. Trend dominance

Corn yields trend upward over decades due to:

* genetics
* fertilizer
* farming efficiency

The models therefore:

* detrend yield
* predict anomalies relative to trend

instead of learning only:

```text
year → higher yield
```

---

## 2. Weather station aggregation

The system averages multiple Corn Belt stations:

* Des Moines
* St. Louis
* Rochester
* Indianapolis
* Kearney
* Aberdeen

This creates a regional agricultural weather signal.

---

## 3. Monthly vs daily data

Historical training:

* monthly weather data

Live forecasting:

* daily weather updates aggregated into monthly features

This balances:

* historical depth
* operational simplicity
* API efficiency

---

# Prediction history

Predictions are automatically saved to:

```text
journal/yield_predictions/prediction_history.csv
```

allowing:

* revision tracking
* model monitoring
* future retraining

Duplicate predictions for the same:

* year
* stage

are automatically overwritten.

---

# Synthetic data augmentation

Generate synthetic futures paths:

```bash
python main.py backtest --augment --augment-paths 10 --retrain
```

Useful for:

* improving generalisation
* stress testing
* low-data commodity environments

