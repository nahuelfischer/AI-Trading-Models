import requests
import pandas as pd
import os
from datetime import datetime

from yfinance import data

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(BASE_DIR, "journal", "live_weather")

os.makedirs(LIVE_DIR, exist_ok=True)


# -----------------------------
# WEATHER COLLECTOR
# -----------------------------
class WeatherCollector:
    def __init__(self):
        # Corn Belt stations (lat, lon)
        self.stations = {
            "des_moines": (41.59, -93.62),
            "indianapolis": (39.77, -86.16),
            "kearney": (40.70, -99.08),
            "aberdeen": (45.46, -98.49),
            "st_louis": (38.63, -90.20),
            "rochester": (44.01, -92.46),
        }

    # -------------------------
    # FETCH DAILY WEATHER
    # -------------------------
    def fetch_daily(self, lat, lon):
        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
            "past_days": 7  # fetch recent past too (important!)
        }

        r = requests.get(url, params=params)

        if r.status_code != 200:
            raise Exception(f"API request failed: {r.text}")

        data = r.json()["daily"]

        temps_c = data["temperature_2m_mean"]
        prec_mm = data["precipitation_sum"]

        # Convert Celsius → Fahrenheit
        temps_f = [(t * 9/5) + 32 if t is not None else None for t in temps_c]

        # Convert mm → inches
        prec_in = [p / 25.4 if p is not None else None for p in prec_mm]

        df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp": temps_f,
        "prec": prec_in
        })

        return df

    # -------------------------
    # UPDATE ALL STATIONS
    # -------------------------
    def update_all(self):
        for name, (lat, lon) in self.stations.items():
            print(f"Updating {name}...")

            df_new = self.fetch_daily(lat, lon)
            path = os.path.join(LIVE_DIR, f"{name}.csv")

            if os.path.exists(path):
                df_old = pd.read_csv(path, parse_dates=["date"])

                # Merge: keep old (manual) data first
                df = pd.concat([df_old, df_new])

                # Remove duplicates (IMPORTANT)
                df = df.drop_duplicates(subset=["date"], keep="first")

            else:
                df = df_new

            # Sort chronologically (VERY IMPORTANT)
            df = df.sort_values("date")

            # Optional: remove impossible values
            df = df[(df["temp"].notna()) & (df["prec"].notna())]

            df.to_csv(path, index=False)

        print("Weather data updated successfully.")

    # -------------------------
    # LOAD ALL STATIONS
    # -------------------------
    def load_all(self):
        data = {}

        for name in self.stations.keys():
            path = os.path.join(LIVE_DIR, f"{name}.csv")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing data for {name}")

            df = pd.read_csv(path, parse_dates=["date"])

            # ---- CLEAN DATA ----
            df["prec"] = df["prec"].replace("T", 0)
            df["prec"] = pd.to_numeric(df["prec"], errors="coerce").fillna(0)

            df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
            df["temp"] = df["temp"].ffill()

            data[name] = df

        return data


# -----------------------------
# HELPER: DAILY → MONTHLY
# -----------------------------
def daily_to_monthly(df):
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month

    monthly = df.groupby(["Year", "Month"]).agg({
        "temp": "mean",
        "prec": "sum"
    }).reset_index()

    return monthly


# -----------------------------
# HELPER: AGGREGATE STATIONS
# -----------------------------
def aggregate_stations_monthly(station_data):
    frames = []

    for name, df in station_data.items():
        m = daily_to_monthly(df)
        m["station"] = name
        frames.append(m)

    df_all = pd.concat(frames)

    grouped = df_all.groupby(["Year", "Month"]).agg({
        "temp": ["mean", "std"],
        "prec": ["mean", "std"]
    })

    grouped.columns = [
        "temp_avg", "temp_std",
        "prec_avg", "prec_std"
    ]

    return grouped.reset_index()


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    wc = WeatherCollector()
    wc.update_all()