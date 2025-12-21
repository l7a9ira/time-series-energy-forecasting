from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# UCI dataset is distributed as a ZIP containing "household_power_consumption.txt"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
)


@dataclass(frozen=True)
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    full_series: pd.Series  # hourly target series
    feature_names: list[str]


def ensure_energy_csv(data_dir: Path) -> Path:
    """
    Ensure data/energy.csv exists.

    If missing:
    - Download the official UCI ZIP
    - Extract household_power_consumption.txt
    - Convert to a smaller CSV (still large) named energy.csv

    This makes the repo reproducible for reviewers.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "energy.csv"
    if csv_path.exists():
        return csv_path

    # Download + extract using stdlib only (no extra dependencies)
    import io
    import zipfile
    import urllib.request

    zip_bytes = urllib.request.urlopen(UCI_ZIP_URL).read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # file name inside zip
        txt_name = "household_power_consumption.txt"
        with zf.open(txt_name) as f:
            # UCI txt is semicolon-separated
            raw = pd.read_csv(
                f,
                sep=";",
                low_memory=False,
            )

    # Save as CSV for simpler future loading
    raw.to_csv(csv_path, index=False)
    return csv_path


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """
    Load the dataset and perform basic cleaning.

    Key points:
    - Missing values are represented as '?' in the original UCI file.
    - Date and Time are separate columns; we combine them into a DateTime index.
    """
    df = pd.read_csv(csv_path)

    # Standard expected column names in UCI dataset
    # Date, Time, Global_active_power, Global_reactive_power, Voltage, Global_intensity,
    # Sub_metering_1, Sub_metering_2, Sub_metering_3
    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Time' not found in energy.csv.")

    # Replace '?' with NaN then convert numeric columns
    df = df.replace("?", np.nan)

    # Combine Date + Time to timestamp
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df.insert(0, "timestamp", dt)
    df = df.drop(columns=["Date", "Time"]).dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Convert target to numeric
    # Target: Global_active_power (kilowatts)
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # Drop rows where target is missing
    df = df.dropna(subset=["Global_active_power"])

    return df


def to_hourly_series(df: pd.DataFrame) -> pd.Series:
    """
    Convert minute-level measurements to an hourly mean series.
    This reduces noise and makes classical ML feature engineering cleaner.
    """
    hourly = df["Global_active_power"].resample("H").mean()
    hourly = hourly.dropna()
    hourly.name = "target_kw"
    return hourly


def make_features(series: pd.Series) -> pd.DataFrame:
    """
    Build time-based features for forecasting.

    Features:
    - Lag features: previous 1..24 hours
    - Rolling means: 24h, 168h
    - Calendar: hour, dayofweek, month

    The prediction target is the value at time t (current hour), using past information only.
    """
    df = pd.DataFrame({"target_kw": series})

    # Calendar features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek  # 0=Mon .. 6=Sun
    df["month"] = df.index.month

    # Lag features (previous hours)
    for lag in range(1, 25):
        df[f"lag_{lag}"] = df["target_kw"].shift(lag)

    # Rolling averages (use shift(1) to avoid leakage from current hour)
    df["roll_mean_24"] = df["target_kw"].shift(1).rolling(window=24).mean()
    df["roll_mean_168"] = df["target_kw"].shift(1).rolling(window=168).mean()

    # Drop rows with NaNs created by lags/rolling
    df = df.dropna()

    return df


def time_split(
    feature_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data respecting time order to prevent leakage.

    Train: earliest portion
    Test: last portion
    """
    n = len(feature_df)
    split_idx = int(np.floor((1.0 - test_size) * n))
    train_df = feature_df.iloc[:split_idx]
    test_df = feature_df.iloc[split_idx:]

    X_train = train_df.drop(columns=["target_kw"])
    y_train = train_df["target_kw"]
    X_test = test_df.drop(columns=["target_kw"])
    y_test = test_df["target_kw"]

    return X_train, X_test, y_train, y_test


def prepare_data(project_root: Path, *, test_size: float = 0.2) -> PreparedData:
    """
    Full preprocessing pipeline:
    - ensure dataset exists
    - load/clean
    - resample hourly
    - feature engineering
    - chronological split
    """
    data_dir = project_root / "data"
    csv_path = ensure_energy_csv(data_dir)
    raw = load_and_clean(csv_path)
    hourly = to_hourly_series(raw)
    feats = make_features(hourly)

    X_train, X_test, y_train, y_test = time_split(feats, test_size=test_size)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        full_series=hourly,
        feature_names=list(X_train.columns),
    ) 
