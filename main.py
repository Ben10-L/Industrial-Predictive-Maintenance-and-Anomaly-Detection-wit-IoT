
from pathlib import Path

from src.data_loader import display_basic_info, load_data
from src.evaluation import (
    evaluate_anomaly_models,
    evaluate_supervised_models,
    save_model_comparison,
)
from src.feature_engineering import add_industrial_features
from src.models import train_anomaly_models, train_supervised_models
from src.preprocessing import prepare_train_test_data
from src.realtime_simulation import run_realtime_simulation
from src.visualization import save_eda_figures


DATA_PATH = Path("data/ai4i2020.csv")
RESULTS_PATH = Path("results/model_comparison.csv")
FIGURES_DIR = Path("results/figures")


def main() -> None:
    df = load_data(DATA_PATH)
    display_basic_info(df)

    df_engineered = add_industrial_features(df)
    save_eda_figures(df_engineered, FIGURES_DIR)

    prepared = prepare_train_test_data(df_engineered)

    supervised_models = train_supervised_models(prepared)
    anomaly_models = train_anomaly_models(prepared)

    results = []
    results.extend(evaluate_supervised_models(supervised_models, prepared))
    results.extend(evaluate_anomaly_models(anomaly_models, prepared))

    comparison = save_model_comparison(results, RESULTS_PATH)
    print("Model comparison:")
    print(comparison)
    print(f"\nSaved comparison table to {RESULTS_PATH}")
    print(f"Saved EDA figures to {FIGURES_DIR}")

    best_model = supervised_models["Random Forest"]
    run_realtime_simulation(
        model=best_model,
        data=df_engineered,
        feature_columns=prepared.feature_columns,
        delay_seconds=0.05,
        max_rows=20,
    )


if __name__ == "__main__":
    main()
