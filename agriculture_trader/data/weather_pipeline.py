"""
Weather Pipeline for Corn Belt Data.

Processes weather data CSVs and aggregates them into monthly features for analysis.
- Loads temperature and precipitation data from CSV files
- Handles missing values and special markers
- Aggregates data across stations
- Generates features for growing and critical periods
"""
import pandas as pd
import numpy as np
import glob
import os

# -----------------------------
# PATH SETUP (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEATHER_DIR = os.path.join(BASE_DIR, "journal", "weather_data")
YIELD_PATH = os.path.join(BASE_DIR, "journal", "crop_data", "yield.csv")

# -----------------------------
# 1. LOAD WEATHER CSV
# -----------------------------
def load_weather_csv(path, data_type):
    # Try common separators
    try:
        df = pd.read_csv(path, sep=";")
    except:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")

    # If still only one column → fallback
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")

    # Clean column names
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # DEBUG (optional)
    # print(path, df.columns)

    # Drop Annual
    if "Annual" in df.columns:
        df = df.drop(columns=["Annual"])

    # Replace missing markers
    df = df.replace("M", np.nan)

    if data_type == "prec":
        df = df.replace("T", 0.01)
    else:
        df = df.replace("T", np.nan)

    # Convert numeric
    for col in df.columns:
        if col != "Year":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure Year exists
    if "Year" not in df.columns:
        raise ValueError(f"'Year' column not found in {path}. Found columns: {df.columns}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    return df.set_index("Year")

# -----------------------------
# 2. LOAD ALL STATIONS
# -----------------------------
def load_all_stations():
    temp_files = glob.glob(os.path.join(WEATHER_DIR, "*_temp.csv"))
    prec_files = glob.glob(os.path.join(WEATHER_DIR, "*_prec.csv"))

    temp_data = {}
    prec_data = {}

    for f in temp_files:
        name = os.path.basename(f).replace("_temp.csv", "")
        temp_data[name] = load_weather_csv(f, "temp")

    for f in prec_files:
        name = os.path.basename(f).replace("_prec.csv", "")
        prec_data[name] = load_weather_csv(f, "prec")

    return temp_data, prec_data


# -----------------------------
# 3. AGGREGATE STATIONS
# -----------------------------
def aggregate_stations(temp_data, prec_data):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    years = list(temp_data.values())[0].index
    result = pd.DataFrame(index=years)

    # TEMPERATURE
    for m in months:
        vals = np.array([df[m].values for df in temp_data.values()])
        result[f"{m}_temp_avg"] = np.nanmean(vals, axis=0)
        result[f"{m}_temp_std"] = np.nanstd(vals, axis=0)

    # PRECIPITATION
    for m in months:
        vals = np.array([df[m].values for df in prec_data.values()])
        result[f"{m}_prec_avg"] = np.nanmean(vals, axis=0)
        result[f"{m}_prec_std"] = np.nanstd(vals, axis=0)

    return result.reset_index().rename(columns={"index": "Year"})


# -----------------------------
# 4. FEATURE ENGINEERING
# -----------------------------
def build_features(df):
    growing = ["Apr","May","Jun","Jul","Aug","Sep"]
    critical = ["Jul","Aug"]

    # Temperature features
    df["temp_growing_avg"] = df[[f"{m}_temp_avg" for m in growing]].mean(axis=1)
    df["temp_critical_avg"] = df[[f"{m}_temp_avg" for m in critical]].mean(axis=1)
    df["temp_growing_std"] = df[[f"{m}_temp_std" for m in growing]].mean(axis=1)

    # Precipitation features
    df["prec_growing_sum"] = df[[f"{m}_prec_avg" for m in growing]].sum(axis=1)
    df["prec_critical_sum"] = df[[f"{m}_prec_avg" for m in critical]].sum(axis=1)
    df["prec_growing_std"] = df[[f"{m}_prec_std" for m in growing]].mean(axis=1)

    # Drought index
    df["drought_index"] = df["temp_critical_avg"] / (df["prec_critical_sum"] + 1)

    return df


# -----------------------------
# 5. LOAD YIELD DATA
# -----------------------------
def load_yield():   
    df = pd.read_csv(YIELD_PATH, sep=";")

    # Clean column names
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # Detect year column automatically
    year_col = None
    for col in df.columns:
        if "year" in col.lower():
            year_col = col
            break

    if year_col is None:
        raise ValueError(f"No Year column found. Columns: {df.columns}")

    df = df.rename(columns={year_col: "Year"})

    # Detect yield column automatically
    yield_col = None
    for col in df.columns:
        if "yield" in col.lower():
            yield_col = col
            break

    if yield_col is None:
        raise ValueError(f"No Yield column found. Columns: {df.columns}")

    df = df.rename(columns={yield_col: "yield"})

    # Convert types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    df = df.dropna(subset=["Year", "yield"])

    df["Year"] = df["Year"].astype(int)

    return df


# -----------------------------
# 6. MERGE DATA
# -----------------------------
def merge_all(weather_df, yield_df):
    df = pd.merge(weather_df, yield_df, on="Year")

    # Add trend
    df["trend"] = df["Year"] - df["Year"].min()

    return df


# -----------------------------
# 7. FULL PIPELINE
# -----------------------------
def build_full_dataset():
    temp_data, prec_data = load_all_stations()

    weather_df = aggregate_stations(temp_data, prec_data)
    weather_df = build_features(weather_df)

    yield_df = load_yield()

    final_df = merge_all(weather_df, yield_df)

    return final_df


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    df = build_full_dataset()
    print(df.head())
    print("\nShape:", df.shape)

