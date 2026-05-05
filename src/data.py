"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file into a dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Read the input CSV into a pandas DataFrame.

    The CSV is expected to follow the schema documented in docs/data_schema.md.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Update data.data_path in config.yaml "
            "or place a CSV at the expected location."
        )
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from the feature matrix."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. Available columns: "
            f"{list(df.columns)}"
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
