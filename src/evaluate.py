from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class EvalResult:
    mae: float
    rmse: float


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> EvalResult:
    """
    Compute MAE and RMSE.

    - MAE: average absolute error (easy to interpret)
    - RMSE: penalizes large errors more strongly
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return EvalResult(mae=mae, rmse=rmse)


def plot_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    title: str = "Energy Consumption Forecast: Actual vs Predicted",
    last_n: int = 400,
) -> Path:
    """
    Plot actual vs predicted values.

    We plot the last 'last_n' points for readability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_plot = y_true.iloc[-last_n:]
    y_pred_plot = pd.Series(y_pred, index=y_true.index).iloc[-last_n:]

    plt.figure()
    plt.plot(y_true_plot.index, y_true_plot.values, label="Actual")
    plt.plot(y_pred_plot.index, y_pred_plot.values, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    return out_path


def save_metrics(result: EvalResult, out_path: Path) -> None:
    """
    Save metrics to outputs/results.txt (good academic practice).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "Time Series Forecasting for Energy Consumption\n"
        "Metrics on Test Set:\n"
        f"- MAE:  {result.mae:.4f} kW\n"
        f"- RMSE: {result.rmse:.4f} kW\n"
    )
    out_path.write_text(text, encoding="utf-8") 
