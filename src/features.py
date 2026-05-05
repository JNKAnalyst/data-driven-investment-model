"""Feature engineering for the investment model.

The transformations here are intentionally conservative — they cover common
preprocessing steps for tabular financial data without baking in assumptions
about a specific dataset:

- drop non-numeric columns that cannot be modeled directly (e.g., date strings)
- forward/back-fill small gaps and drop remaining NA rows
- add simple rolling statistics for any column flagged as a price/level series
- standardize numeric features (mean 0, std 1) for distance-based models like KNN

If you bring your own dataset, adjust `engineer_features` to suit it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def _coerce_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop date/time columns — they're useful for ordering, not as raw features."""
    out = df.copy()
    for col in list(out.columns):
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out = out.drop(columns=[col])
        elif out[col].dtype == object:
            try:
                pd.to_datetime(out[col])
                out = out.drop(columns=[col])
            except (ValueError, TypeError):
                pass
    return out


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Append rolling mean and rolling std for each (column, window) pair."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_roll{w}_mean"] = out[col].rolling(window=w, min_periods=1).mean()
            out[f"{col}_roll{w}_std"] = out[col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return out


def engineer_features(
    df: pd.DataFrame,
    rolling_columns: list[str] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Apply the default feature pipeline to a raw DataFrame."""
    out = _coerce_datetime_columns(df)
    out = out.select_dtypes(include="number")
    out = out.ffill().bfill().dropna()

    if rolling_columns and rolling_windows:
        out = add_rolling_features(out, rolling_columns, rolling_windows)

    return out


def standardize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and apply to both splits."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
