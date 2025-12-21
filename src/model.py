from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def build_model(model_name: str = "random_forest"):
    """
    Model factory.

    model_name:
    - "linear": Linear Regression baseline
    - "random_forest": stronger non-linear classical model

    We keep models simple (academic clarity > optimization).
    """
    name = model_name.strip().lower()

    if name == "linear":
        return LinearRegression()

    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name='{model_name}'. Use 'linear' or 'random_forest'.") 
