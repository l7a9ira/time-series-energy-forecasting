from __future__ import annotations

from pathlib import Path

from src.preprocess import prepare_data
from src.model import build_model
from src.evaluate import evaluate_regression, plot_forecast, save_metrics


def main() -> None:
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "outputs"

    # 1) Prepare data
    data = prepare_data(project_root, test_size=0.2)

    # 2) Train model
    # Choose: "linear" or "random_forest"
    model = build_model("random_forest")
    model.fit(data.X_train, data.y_train)

    # 3) Predict and evaluate
    y_pred = model.predict(data.X_test)
    metrics = evaluate_regression(data.y_test, y_pred)

    # 4) Save outputs
    save_metrics(metrics, outputs_dir / "results.txt")
    plot_forecast(data.y_test, y_pred, outputs_dir / "forecast_plot.png", last_n=400)

    print("=== Done ===")
    print(f"MAE:  {metrics.mae:.4f} kW")
    print(f"RMSE: {metrics.rmse:.4f} kW")
    print(f"Saved plot: {outputs_dir / 'forecast_plot.png'}")
    print(f"Saved metrics: {outputs_dir / 'results.txt'}")


if __name__ == "__main__":
    main() 
