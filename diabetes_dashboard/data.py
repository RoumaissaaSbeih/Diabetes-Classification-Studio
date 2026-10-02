from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET_COLUMN, ZERO_AS_MISSING


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = set(FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {formatted}")
    return frame


def build_dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    positives = int(frame[TARGET_COLUMN].sum())
    negatives = int(len(frame) - positives)
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "positive_cases": positives,
        "negative_cases": negatives,
        "positive_rate": round((positives / len(frame)) * 100, 2),
    }


def compute_feature_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        series = frame[feature]
        valid = series.replace(0, np.nan) if feature in ZERO_AS_MISSING else series
        valid = valid.dropna()

        summary[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(valid.median()),
            "q1": float(valid.quantile(0.25)),
            "q3": float(valid.quantile(0.75)),
            "mean": float(valid.mean()),
        }

    return summary


def build_preview_records(frame: pd.DataFrame, rows: int = 10) -> list[dict[str, Any]]:
    return frame.head(rows).to_dict(orient="records")
