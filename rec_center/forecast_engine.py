"""Load forecast model and predict utilization by date, time, and location."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rec_center_utils import (
    MAX_FORECAST_DAYS,
    build_slot_dataframe,
    get_feature_frame,
    models_path,
    utilization_to_level,
)

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_forecast_model():
    import joblib

    path = models_path("catboost_forecast.joblib")
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast model not found at {path}. Run: python train_forecast_model.py"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_model_metadata() -> dict:
    meta_path = models_path("location_capacity.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {meta_path}. Run: python train_forecast_model.py"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def get_capacity_map() -> dict[str, float]:
    return load_model_metadata()["capacity"]


def get_locations() -> list[str]:
    from rec_center_utils import order_locations_for_ui

    return order_locations_for_ui(load_model_metadata()["locations"])


def _validate_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    dates = pd.date_range(start, end, freq="D")
    if len(dates) > MAX_FORECAST_DAYS:
        raise ValueError(f"Date range exceeds {MAX_FORECAST_DAYS} days. Narrow the range.")
    return dates


def predict_range(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    location: str,
) -> pd.DataFrame:
    """Predict utilization for each 30-min slot in [start_date, end_date] at one location."""
    dates = _validate_date_range(start_date, end_date)
    capacity_map = get_capacity_map()
    slots = build_slot_dataframe(dates, location, capacity_map)

    model = load_forecast_model()
    preds = np.clip(model.predict(get_feature_frame(slots)), 0.0, 1.0)

    out = pd.DataFrame(
        {
            "date": slots["timestamp"].dt.date,
            "slot_start": slots["timestamp"],
            "hour": slots["hour"],
            "minute": slots["timestamp"].dt.minute,
            "day_of_week": slots["day_of_week"],
            "location": slots["location"],
            "predicted_utilization": preds,
        }
    )
    out["usage_level"] = out["predicted_utilization"].map(utilization_to_level)
    return out


def predict_all_locations(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Predict utilization for all locations (for explorer heatmaps)."""
    dates = _validate_date_range(start_date, end_date)
    capacity_map = get_capacity_map()
    locations = get_locations()
    slots = build_slot_dataframe(dates, locations, capacity_map)

    model = load_forecast_model()
    preds = np.clip(model.predict(get_feature_frame(slots)), 0.0, 1.0)

    out = slots[["timestamp", "location", "hour", "day_of_week"]].copy()
    out["date"] = out["timestamp"].dt.date
    out["slot_start"] = out["timestamp"]
    out["minute"] = out["timestamp"].dt.minute
    out["predicted_utilization"] = preds
    out["usage_level"] = out["predicted_utilization"].map(utilization_to_level)
    return out


def model_is_ready() -> bool:
    return models_path("catboost_forecast.joblib").exists()
