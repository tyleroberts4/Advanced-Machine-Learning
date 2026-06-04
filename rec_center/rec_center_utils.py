"""Shared utilities for the Cal Poly Rec Center usage prediction project."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# Time-based split cutoffs (inclusive/exclusive boundaries documented in notebooks).
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")

CLASS_LOW = 0.30
CLASS_HIGH = 0.60

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_summer",
    "is_winter_break",
    "is_spring_break",
    "is_finals_week",
    "capacity",
]

CATEGORICAL_COLUMNS = ["location"]

# Whole-facility Occuspace series (for students who use multiple areas).
GENERAL_LOCATION = "Rec Center"
GENERAL_LOCATION_LABEL = "General Rec Center (any area)"

DEFAULT_LOCATIONS = [
    GENERAL_LOCATION,
    "1st Floor",
    "2nd Floor",
    "Lower Exercise Room",
    "Upper Exercise Room",
    "Track Exercise Room",
]

LOCATION_DISPLAY_NAMES = {
    GENERAL_LOCATION: GENERAL_LOCATION_LABEL,
}

REC_CENTER_OPEN = time(6, 0)
REC_CENTER_CLOSE = time(23, 30)
SLOT_MINUTES = 30
MAX_FORECAST_DAYS = 31


def project_root() -> Path:
    return Path(__file__).resolve().parent


def data_path(name: str = "rec_center_clean.parquet") -> Path:
    return project_root() / "data" / name


def figures_path(name: str) -> Path:
    path = project_root() / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_path() -> Path:
    return project_root().parent / "datasets" / "rec_center_usage.xlsx"


def _date_ranges() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Approximate Cal Poly academic calendar windows for 2023-2026."""
    return {
        "is_summer": [
            (pd.Timestamp("2023-06-15"), pd.Timestamp("2023-09-17")),
            (pd.Timestamp("2024-06-15"), pd.Timestamp("2024-09-15")),
            (pd.Timestamp("2025-06-14"), pd.Timestamp("2025-09-14")),
            (pd.Timestamp("2026-06-13"), pd.Timestamp("2026-09-13")),
        ],
        "is_winter_break": [
            (pd.Timestamp("2023-12-16"), pd.Timestamp("2024-01-07")),
            (pd.Timestamp("2024-12-07"), pd.Timestamp("2025-01-05")),
            (pd.Timestamp("2025-12-06"), pd.Timestamp("2026-01-04")),
        ],
        "is_spring_break": [
            (pd.Timestamp("2024-03-23"), pd.Timestamp("2024-03-31")),
            (pd.Timestamp("2025-03-22"), pd.Timestamp("2025-03-30")),
            (pd.Timestamp("2026-03-21"), pd.Timestamp("2026-03-29")),
        ],
        "is_finals_week": [
            (pd.Timestamp("2023-12-11"), pd.Timestamp("2023-12-15")),
            (pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-14")),
            (pd.Timestamp("2024-12-09"), pd.Timestamp("2024-12-13")),
            (pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-13")),
            (pd.Timestamp("2025-12-08"), pd.Timestamp("2025-12-12")),
            (pd.Timestamp("2026-06-08"), pd.Timestamp("2026-06-12")),
        ],
    }


def _in_any_window(ts: pd.Series, ranges: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=ts.index)
    for start, end in ranges:
        mask |= (ts >= start) & (ts <= end)
    return mask.astype(int)


def add_academic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    for flag, ranges in _date_ranges().items():
        out[flag] = _in_any_window(ts, ranges)
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    path = path or raw_data_path()
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "Location": "location",
            "Timestamp": "timestamp",
            "Date": "date",
            "Day of Week": "day_of_week",
            "Week of Year": "week_of_year",
            "Time": "time",
            "Hour of Day": "hour",
            "Average Occupancy": "average_occupancy",
            "Average Utilization": "average_utilization",
            "Peak Occupancy": "peak_occupancy",
            "Peak Utilization": "peak_utilization",
            "Capacity": "capacity",
            "Location Path": "location_path",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean duplicates, cap utilization, and engineer base features."""
    report = {
        "rows_raw": len(df),
        "duplicate_location_timestamp": int(df.duplicated(subset=["location", "timestamp"]).sum()),
        "utilization_above_one": int((df["average_utilization"] > 1.0).sum()),
    }

    cleaned = (
        df.sort_values(["location", "timestamp"])
        .drop_duplicates(subset=["location", "timestamp"], keep="first")
        .copy()
    )

    cleaned["average_utilization_raw"] = cleaned["average_utilization"]
    cleaned["average_utilization"] = cleaned["average_utilization"].clip(upper=1.0)

    cleaned["month"] = cleaned["timestamp"].dt.month
    cleaned["is_weekend"] = cleaned["day_of_week"].isin([5, 6]).astype(int)

    cleaned = add_cyclical_features(cleaned)
    cleaned = add_academic_calendar_features(cleaned)

    cleaned["usage_level"] = pd.cut(
        cleaned["average_utilization"],
        bins=[-np.inf, CLASS_LOW, CLASS_HIGH, np.inf],
        labels=["low", "medium", "high"],
    )

    cleaned["split"] = assign_time_split(cleaned["timestamp"])

    report["rows_clean"] = len(cleaned)
    report["split_counts"] = cleaned["split"].value_counts().to_dict()
    return cleaned, report


def assign_time_split(timestamp: pd.Series) -> pd.Series:
    split = pd.Series(index=timestamp.index, dtype="object")
    split[timestamp <= TRAIN_END] = "train"
    split[(timestamp > TRAIN_END) & (timestamp <= VAL_END)] = "validation"
    split[timestamp > VAL_END] = "test"
    return split


def get_feature_frame(df: pd.DataFrame, include_calendar: bool = True) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLUMNS if include_calendar or not c.startswith("is_")]
    return df[cols + ["location"]].copy()


def get_xy_regression(df: pd.DataFrame, include_calendar: bool = True):
    x = get_feature_frame(df, include_calendar=include_calendar)
    y = df["average_utilization"].astype(float)
    return x, y


def get_xy_classification(df: pd.DataFrame, include_calendar: bool = True):
    x = get_feature_frame(df, include_calendar=include_calendar)
    y = df["usage_level"].astype(str)
    return x, y


def save_clean_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_clean_data(path: Path | None = None) -> pd.DataFrame:
    path = path or data_path()
    return pd.read_parquet(path)


def models_path(name: str = "catboost_forecast.joblib") -> Path:
    path = project_root() / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def location_display_name(location: str) -> str:
    """User-facing label for a location (general option for multi-area gym-goers)."""
    return LOCATION_DISPLAY_NAMES.get(location, location)


def order_locations_for_ui(locations: list[str]) -> list[str]:
    """Put general whole-facility option first, then specific areas alphabetically."""
    specific = sorted(loc for loc in locations if loc != GENERAL_LOCATION)
    if GENERAL_LOCATION in locations:
        return [GENERAL_LOCATION] + specific
    return specific


def get_locations(df: pd.DataFrame | None = None) -> list[str]:
    """Return unique locations from data or defaults, general option first."""
    if df is not None and "location" in df.columns:
        raw = df["location"].dropna().unique().tolist()
        return order_locations_for_ui(raw)
    path = data_path()
    if path.exists():
        raw = pd.read_parquet(path, columns=["location"])["location"].unique().tolist()
        return order_locations_for_ui(raw)
    return order_locations_for_ui(list(DEFAULT_LOCATIONS))


def location_capacity_map(df: pd.DataFrame) -> dict[str, float]:
    """Median listed capacity per location for forecast features."""
    return df.groupby("location")["capacity"].median().astype(float).to_dict()


def utilization_to_level(util: float) -> str:
    if util < CLASS_LOW:
        return "low"
    if util <= CLASS_HIGH:
        return "medium"
    return "high"


def iter_half_hour_slots() -> list[tuple[int, int]]:
    """(hour, minute) pairs from 6:00 through 23:30 inclusive."""
    slots: list[tuple[int, int]] = []
    for hour in range(6, 24):
        for minute in (0, 30):
            slots.append((hour, minute))
    return slots


def _normalize_dates(dates: Union[pd.DatetimeIndex, list, date, pd.Timestamp]) -> pd.DatetimeIndex:
    if isinstance(dates, pd.DatetimeIndex):
        return dates.normalize()
    return pd.DatetimeIndex(pd.to_datetime(dates)).normalize()


def build_slot_dataframe(
    dates: Union[pd.DatetimeIndex, list, date, pd.Timestamp],
    locations: Union[str, list[str]],
    capacity_map: dict[str, float],
) -> pd.DataFrame:
    """Build feature-ready rows for each date, 30-min slot, and location."""
    date_index = _normalize_dates(dates)
    if isinstance(locations, str):
        locs = [locations]
    else:
        locs = list(locations)

    rows: list[dict] = []
    for day in date_index:
        for hour, minute in iter_half_hour_slots():
            ts = pd.Timestamp(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=hour,
                minute=minute,
            )
            for loc in locs:
                rows.append(
                    {
                        "timestamp": ts,
                        "location": loc,
                        "hour": hour,
                        "day_of_week": int(ts.dayofweek),
                        "month": int(ts.month),
                        "week_of_year": int(ts.isocalendar().week),
                        "capacity": float(capacity_map.get(loc, 100.0)),
                    }
                )

    df = pd.DataFrame(rows)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df = add_cyclical_features(df)
    df = add_academic_calendar_features(df)
    return df


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
