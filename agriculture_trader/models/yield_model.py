from logging import critical
import os
import sys
import numpy as np
import pandas as pd
from sympy import series
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# -----------------------------
# IMPORT SETTINGS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

sys.path.append(CONFIG_DIR)

from settings import PREDICTION_YEAR

# -----------------------------
# IMPORT PIPELINES
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.append(DATA_DIR)

from weather_pipeline import build_full_dataset
from weather_collector import WeatherCollector, aggregate_stations_monthly


# -----------------------------
# MODEL CLASS
# -----------------------------
class CornYieldModel:
    def __init__(self):
        self.models = []
        self.features = None
        self.df = None
        self.trend_model = None

    # -------------------------
    # LOAD TRAINING DATA
    # -------------------------
    def load_data(self):
        df = build_full_dataset()

        keep_cols = [
            "Year",
            "temp_growing_avg",
            "temp_critical_avg",
            "temp_growing_std",
            "prec_growing_sum",
            "prec_critical_sum",
            "prec_growing_std",
            "drought_index",
            "yield"
        ]

        df = df[keep_cols].copy()
        df = df.sort_values("Year")

        # -------------------------
        # BUILD TREND MODEL
        # -------------------------
        X_trend = df[["Year"]]
        y_trend = df["yield"]

        self.trend_model = LinearRegression()
        self.trend_model.fit(X_trend, y_trend)

        # Expected trend yield
        df["trend_yield"] = self.trend_model.predict(X_trend)

        # Yield anomaly (THIS is target now)
        df["yield_anomaly"] = df["yield"] - df["trend_yield"]

        self.df = df

        # Features NO LONGER include trend
        self.features = [
            "temp_growing_avg",
            "temp_critical_avg",
            "temp_growing_std",
            "prec_growing_sum",
            "prec_critical_sum",
            "prec_growing_std",
            "drought_index"
        ]

        return df

    # -------------------------
    # TRAIN ENSEMBLE
    # -------------------------
    def train(self, n_models=5):
        df = self.df

        train = df[df["Year"] <= 2015]

        X = train[self.features]
        y = train["yield_anomaly"]

        self.models = []

        for i in range(n_models):
            model = XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=i
            )

            model.fit(X, y)
            self.models.append(model)

        print(f"Trained {n_models} ensemble models")

    # -------------------------
    # PREDICTION WITH RANGE
    # -------------------------
    def predict_range(self, X, year):
        preds = []

        # Trend expectation
        trend_yield = self.trend_model.predict([[year]])[0]

        for model in self.models:
            anomaly_pred = model.predict(X)[0]

            # Add trend back
            final_pred = trend_yield + anomaly_pred

            preds.append(final_pred)

        preds = np.array(preds)

        return {
            "low": np.percentile(preds, 20),
            "mid": np.mean(preds),
            "high": np.percentile(preds, 80),
            "trend": trend_yield
        }

    # -------------------------
    # BUILD FEATURES FROM LIVE DATA
    # -------------------------
    def build_live_features(self, monthly_df, year):
        def safe_mean(series):
            return series.mean() if len(series) > 0 else np.nan

        def safe_sum(series):
            return series.sum() if len(series) > 0 else 0
        
        growing = [4,5,6,7,8,9]
        critical = [7,8]

        df = monthly_df[monthly_df["Year"] == year]

        temp_growing_avg = safe_mean(df[df["Month"].isin(growing)]["temp_avg"])
        temp_critical_avg = safe_mean(df[df["Month"].isin(critical)]["temp_avg"])

        prec_growing_sum = safe_sum(df[df["Month"].isin(growing)]["prec_avg"])
        prec_critical_sum = safe_sum(df[df["Month"].isin(critical)]["prec_avg"])

        return pd.DataFrame([{
        "temp_growing_avg": temp_growing_avg,
        "temp_critical_avg": temp_critical_avg,
        "temp_growing_std": df["temp_std"].mean(),
        "prec_growing_sum": prec_growing_sum,
        "prec_critical_sum": prec_critical_sum,
        "prec_growing_std": df["prec_std"].mean(),
        "drought_index": temp_critical_avg / (prec_critical_sum + 1)
        }])

    # -------------------------
    # LIVE PREDICTION
    # -------------------------
    def predict_live(self, year):
        wc = WeatherCollector()

        # Step 1: update data
        wc.update_all()

        # Step 2: load stored data
        station_data = wc.load_all()

        # Step 3: aggregate
        monthly = aggregate_stations_monthly(station_data)

        # Step 4: build features
        X = self.build_live_features(monthly, year)
        X = X.fillna(0)
        
        # Step 5: predict
        pred = self.predict_range(X, year)
        
        print(f"\nPrediction for {year}:")
        print(f"Low:  {pred['low']:.2f}")
        print(f"Mid:  {pred['mid']:.2f}")
        print(f"High: {pred['high']:.2f}")

        return pred

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    def plot_feature_importance(self):
        import matplotlib.pyplot as plt

        if not self.models:
            raise ValueError("Model not trained yet.")

        # Collect importances from all models
        all_importances = []

        for model in self.models:
            all_importances.append(model.feature_importances_)

        # Average across ensemble
        mean_importance = np.mean(all_importances, axis=0)

        # Create series
        feat_imp = pd.Series(mean_importance, index=self.features)
        feat_imp = feat_imp.sort_values()

        # Plot
        plt.figure()
        feat_imp.plot(kind="barh")
        plt.title("Feature Importance (Ensemble Average)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    model = CornYieldModel()

    print("Loading data...")
    model.load_data()

    print("Training...")
    model.train()

    print("Live prediction...")
    model.predict_live(year=PREDICTION_YEAR)

    print("\nFeature importance...")
    model.plot_feature_importance()