"""
Early-season yield prediction model.

This model focuses on making early predictions based on weather data from the growing season.
- Uses historical yield data to model the long-term trend and detrend yields to focus on anomalies.
- Trains separate XGBoost regression models for different stages of the growing season (April to September).
- Each model uses features from the months leading up to and including its stage (e.g., April model uses April data, May model uses April+May data, etc.).
- Provides a live prediction function that:
    - Collects recent weather data from the WeatherCollector.
    - Builds features based on the available monthly data for the current stage.
    - Predicts yield using the appropriate stage model and adds back the long-term trend.
    - Tracks prediction history and revisions over time, allowing for updates as new weather data comes in.
    - Includes a function to plot feature importance for each stage model, helping to identify which weather factors are most influential at different points in the season.
"""

from datetime import datetime
import os
import sys
from unittest import result
from matplotlib.style import available
from matplotlib.style import available
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# -----------------------------
# IMPORT SETTINGS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

sys.path.append(CONFIG_DIR)

from settings import PREDICTION_YEAR, PREDICTION_STAGE    

# -------------------------------------------------
# IMPORT WEATHER PIPELINE
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

sys.path.append(DATA_DIR)

from weather_pipeline import build_full_dataset
from weather_collector import (
    WeatherCollector,
    aggregate_stations_monthly
)

# -------------------------------------------------
# EARLY-SEASON MODEL
# -------------------------------------------------
class EarlySeasonYieldModel:
    def __init__(self):
        self.models = {}
        self.trend_model = None
        self.df = None

        # growing season progression
        self.month_sets = {
            "april": [1,2,3,4],
            "may": [1,2,3,4,5],
            "june": [1,2,3,4,5,6],
            "july": [1,2,3,4,5,6,7],
            "august": [1,2,3,4,5,6,7,8],
            "september": [1,2,3,4,5,6,7,8,9]
        }

    def build_live_features(self, monthly_df, year, stage):
        df = monthly_df[monthly_df["Year"] == year].copy()

        month_names = {
            1:"jan",2:"feb",3:"mar",4:"apr",
            5:"may",6:"jun",7:"jul",8:"aug",9:"sep"
        }

        result = {}

        # -------------------------
        # MONTHLY FEATURES
        # -------------------------
        for month in self.month_sets[stage]:

            name = month_names[month]

            month_df = df[df["Month"] == month]

            if len(month_df) == 0:
                result[f"{name}_temp_avg"] = np.nan
                result[f"{name}_prec_avg"] = np.nan
                continue

            result[f"{name}_temp_avg"] = month_df["temp_avg"].mean()
            result[f"{name}_prec_avg"] = month_df["prec_avg"].mean()

        # -------------------------
        # SEASONAL FEATURES
        # -------------------------
        available = df[df["Month"].isin(self.month_sets[stage])]

        result["temp_growing_avg"] = available["temp_avg"].mean()
        result["temp_critical_avg"] = available[available["Month"].isin([7,8])]["temp_avg"].mean()

        result["prec_growing_sum"] = available["prec_avg"].sum()

        result["prec_critical_sum"] = available[available["Month"].isin([7,8])]["prec_avg"].sum()

        result["drought_index"] = (result["temp_growing_avg"] / (result["prec_growing_sum"] + 1))

        return pd.DataFrame([result])
    
    def save_prediction_history(
        self,
        prediction,
        stage,
        year,
        drought_index=None
    ):
        import os
        from datetime import datetime

        history_dir = os.path.join(
            BASE_DIR,
            "journal",
            "yield_predictions"
        )

        os.makedirs(history_dir, exist_ok=True)

        history_path = os.path.join(
            history_dir,
            "prediction_history.csv"
        )

        new_row = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "year": year,
            "stage": stage,
            "prediction": prediction,
            "drought_index": drought_index
        }

        # -----------------------------
        # LOAD EXISTING
        # -----------------------------
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)

            # -----------------------------
            # REMOVE DUPLICATE ENTRY
            # (same year + stage)
            # -----------------------------
            df = df[
                ~((df["year"] == year) &
                (df["stage"] == stage))
            ]

            df = pd.concat(
                [df, pd.DataFrame([new_row])],
                ignore_index=True
            )

        else:
            df = pd.DataFrame([new_row])

        df.to_csv(history_path, index=False)
    
    def get_previous_prediction(self, year):
        """
        Load previous prediction for revision tracking.
        """

        import os

        history_path = os.path.join(
            BASE_DIR,
            "journal",
            "yield_predictions",
            "prediction_history.csv"
        )

        if not os.path.exists(history_path):
            return None

        df = pd.read_csv(history_path)

        df = df[df["year"] == year]

        if len(df) == 0:
            return None

        return df.iloc[-1]["prediction"]

    def predict_live(self, stage, year):
        # -------------------------
        # LOAD LIVE WEATHER
        # -------------------------
        wc = WeatherCollector()

        station_data = wc.load_all()

        # -------------------------
        # AGGREGATE STATIONS
        # -------------------------
        monthly = aggregate_stations_monthly(station_data)

        # -------------------------
        # BUILD FEATURES
        # -------------------------
        live_features = self.build_live_features(
            monthly_df=monthly,
            year=year,
            stage=stage
        )

        # -------------------------
        # PREDICT
        # -------------------------
        pred = self.predict(
            stage=stage,
            live_features=live_features,
            year=year
        )
        # --------------------------------
        # PREVIOUS PREDICTION
        # --------------------------------
        previous_prediction = self.get_previous_prediction(year)

        if previous_prediction is None:
            previous_prediction = pred["mid"]

        # --------------------------------
        # REVISION
        # --------------------------------
        revision = (
            pred["mid"]
            - previous_prediction
        )

        # --------------------------------
        # SAVE CURRENT PREDICTION
        # --------------------------------
        self.save_prediction_history(
        prediction=pred["mid"],
        stage=stage,
        year=year,
        drought_index=live_features["drought_index"].iloc[0]
        )

        # --------------------------------
        # ADD TO OUTPUT
        # --------------------------------
        pred["previous_prediction"] = previous_prediction
        pred["yield_revision"] = revision
        pred["drought_index"] = (
            live_features["drought_index"].iloc[0]
        )

        print("\n===== EARLY-SEASON YIELD PREDICTION =====")
        print(f"Stage: {pred['stage']}")
        print(f"Trend Yield: {pred['trend']:.2f}")
        print(f"Low Estimate: {pred['low']:.2f}")
        print(f"Expected Yield: {pred['mid']:.2f}")
        print(f"High Estimate: {pred['high']:.2f}")

        return pred

    # -------------------------------------------------
    # LOAD DATA + DETREND
    # -------------------------------------------------
    def load_data(self):
        df = build_full_dataset()
        df = df.sort_values("Year")

        # -----------------------------
        # TREND MODEL
        # -----------------------------
        X_trend = df[["Year"]]
        y_trend = df["yield"]

        self.trend_model = LinearRegression()
        self.trend_model.fit(X_trend, y_trend)

        df["trend_yield"] = self.trend_model.predict(X_trend)
        df["yield_anomaly"] = df["yield"] - df["trend_yield"]

        self.df = df

        return df


    # -------------------------------------------------
    # BUILD FEATURE LIST FOR GIVEN MONTHS
    # -------------------------------------------------
    def get_feature_columns(self, allowed_months):
        cols = []

        month_names = {
            1:"jan",2:"feb",3:"mar",4:"apr",
            5:"may",6:"jun",7:"jul",8:"aug",9:"sep"
        }

        for m in allowed_months:
            name = month_names[m]

            temp_col = f"{name}_temp_avg"
            prec_col = f"{name}_prec_avg"

            if temp_col in self.df.columns:
                cols.append(temp_col)

            if prec_col in self.df.columns:
                cols.append(prec_col)

        # seasonal aggregate features
        additional = [
            "temp_growing_avg",
            "temp_critical_avg",
            "prec_growing_sum",
            "prec_critical_sum",
            "drought_index"
        ]

        for col in additional:
            if col in self.df.columns:
                cols.append(col)

        return cols


    # -------------------------------------------------
    # TRAIN ALL MONTHLY MODELS
    # -------------------------------------------------
    def train_all(self):
        if self.df is None:
            self.load_data()

        train = self.df[self.df["Year"] <= 2015]

        for stage, months in self.month_sets.items():
            print(f"Training {stage} model...")

            feature_cols = self.get_feature_columns(months)

            X = train[feature_cols].copy()
            y = train["yield_anomaly"]

            # fill missing values safely
            X = X.fillna(0)

            ensemble = []

            for i in range(5):
                model = XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    random_state=i
                )

                model.fit(X, y)
                ensemble.append(model)

            self.models[stage] = {
                "models": ensemble,
                "features": feature_cols
            }

        print("All early-season models trained.")


    # -------------------------------------------------
    # PREDICT WITH RANGE
    # -------------------------------------------------
    def predict(self, stage, live_features, year):
        if stage not in self.models:
            raise ValueError(f"Missing model for stage: {stage}")

        model_info = self.models[stage]

        ensemble = model_info["models"]
        feature_cols = model_info["features"]

        X = live_features[feature_cols].copy()
        X = X.fillna(0)

        trend_yield = self.trend_model.predict([[year]])[0]

        preds = []

        for model in ensemble:
            anomaly = model.predict(X)[0]
            final_yield = trend_yield + anomaly
            preds.append(final_yield)

        preds = np.array(preds)

        return {
            "stage": stage,
            "trend": trend_yield,
            "low": np.percentile(preds, 20),
            "mid": np.mean(preds),
            "high": np.percentile(preds, 80)
        }


    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------
    def plot_feature_importance(self, stage):
        import matplotlib.pyplot as plt

        if stage not in self.models:
            raise ValueError("Stage model not found")

        model_info = self.models[stage]

        all_importances = []

        for model in model_info["models"]:
            all_importances.append(model.feature_importances_)

        mean_importance = np.mean(all_importances, axis=0)

        feat_imp = pd.Series(
            mean_importance,
            index=model_info["features"]
        )

        feat_imp = feat_imp.sort_values()

        plt.figure(figsize=(10,8))
        feat_imp.plot(kind="barh")
        plt.title(f"Feature Importance - {stage.title()} Model")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()


# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
if __name__ == "__main__":

    model = EarlySeasonYieldModel()

    print("Loading historical data...")
    model.load_data()

    print("Training models...")
    model.train_all()

    print("Running live prediction...")
    model.predict_live(
        stage=PREDICTION_STAGE,
        year=PREDICTION_YEAR
    )

    print("Showing feature importance...")
    model.plot_feature_importance(PREDICTION_STAGE)
