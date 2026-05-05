"""Cross-validation and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


@dataclass
class ModelScore:
    name: str
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    r2_std: float


def cross_validate_model(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> ModelScore:
    """Run k-fold CV and return RMSE and R² (mean ± std)."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    neg_mse = cross_val_score(estimator, X, y, scoring="neg_mean_squared_error", cv=kfold)
    rmse_scores = np.sqrt(-neg_mse)

    r2_scores = cross_val_score(estimator, X, y, scoring="r2", cv=kfold)

    return ModelScore(
        name=name,
        rmse_mean=float(rmse_scores.mean()),
        rmse_std=float(rmse_scores.std()),
        r2_mean=float(r2_scores.mean()),
        r2_std=float(r2_scores.std()),
    )


def holdout_metrics(y_true, y_pred) -> dict:
    """Compute RMSE and R² on a held-out split."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_results_table(scores: list[ModelScore]) -> str:
    """Render a list of ModelScore objects as a plain-text table."""
    header = f"{'Model':<20} {'RMSE (mean)':>12} {'RMSE (std)':>12} {'R² (mean)':>12} {'R² (std)':>12}"
    sep = "-" * len(header)
    rows = [header, sep]
    for s in scores:
        rows.append(
            f"{s.name:<20} {s.rmse_mean:>12.4f} {s.rmse_std:>12.4f} "
            f"{s.r2_mean:>12.4f} {s.r2_std:>12.4f}"
        )
    return "\n".join(rows)
