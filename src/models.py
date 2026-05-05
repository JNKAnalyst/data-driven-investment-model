"""Model factory and training entrypoint.

Run from the repository root:

    python -m src.models --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from .data import load_config, load_dataset, split_features_target
from .evaluate import cross_validate_model, format_results_table, holdout_metrics
from .features import engineer_features


def build_models(model_cfg: dict) -> dict:
    """Return a dict of {name: estimator} ready for fit/predict."""
    rf_cfg = model_cfg.get("random_forest", {})
    knn_cfg = model_cfg.get("knn", {})
    lr_cfg = model_cfg.get("linear_regression", {})

    return {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**lr_cfg)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(**knn_cfg)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            random_state=model_cfg.get("random_state", 42), **rf_cfg
        ),
    }


def run(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg.get("models", {})
    eval_cfg = cfg.get("evaluation", {})

    df = load_dataset(data_cfg["data_path"])
    df = engineer_features(df)
    X, y = split_features_target(df, data_cfg["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )

    estimators = build_models(model_cfg)
    cv_folds = eval_cfg.get("cv_folds", 5)
    random_state = data_cfg.get("random_state", 42)

    scores = []
    for name, est in estimators.items():
        scores.append(
            cross_validate_model(
                name=name,
                estimator=est,
                X=X_train,
                y=y_train,
                cv_folds=cv_folds,
                random_state=random_state,
            )
        )

    print("\nCross-validated training scores:\n")
    print(format_results_table(scores))

    print("\nHold-out test scores:\n")
    for name, est in estimators.items():
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        m = holdout_metrics(y_test, preds)
        print(f"  {name:<20} RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare investment models.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config)
