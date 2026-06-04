"""Recommend gym times from forecast predictions and student schedule constraints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Optional

import pandas as pd

from forecast_engine import predict_range
from rec_center_utils import (
    REC_CENTER_CLOSE,
    REC_CENTER_OPEN,
    SLOT_MINUTES,
    location_display_name,
    utilization_to_level,
)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class ClassBlock:
    day_of_week: int
    start_time: time
    end_time: time


@dataclass
class SlotRecommendation:
    date: date
    day_name: str
    start_time: time
    end_time: time
    predicted_utilization: float
    usage_level: str
    location: str


@dataclass
class DayPlan:
    date: date
    day_name: str
    primary: Optional[SlotRecommendation]
    alternates: list[SlotRecommendation]
    message: str


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_time(minutes: int) -> time:
    minutes = max(0, min(24 * 60 - 1, minutes))
    return time(minutes // 60, minutes % 60)


def _busy_intervals_for_day(
    day_of_week: int,
    weekly_classes: list[ClassBlock],
    buffer_minutes: int,
) -> list[tuple[int, int]]:
    """Return list of (start_min, end_min) busy blocks including pre-class buffer."""
    intervals: list[tuple[int, int]] = []
    for block in weekly_classes:
        if block.day_of_week != day_of_week:
            continue
        start = _time_to_minutes(block.start_time)
        end = _time_to_minutes(block.end_time)
        buffered_start = max(_time_to_minutes(REC_CENTER_OPEN), start - buffer_minutes)
        intervals.append((buffered_start, end))
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlaps_busy(start_min: int, end_min: int, busy: list[tuple[int, int]]) -> bool:
    for busy_start, busy_end in busy:
        if start_min < busy_end and end_min > busy_start:
            return True
    return False


def _slots_for_workout(
    forecast_day: pd.DataFrame,
    workout_minutes: int,
) -> pd.DataFrame:
    """Average utilization over consecutive 30-min slots covering workout_minutes."""
    if forecast_day.empty:
        return pd.DataFrame()

    n_slots = max(1, workout_minutes // SLOT_MINUTES)
    day = forecast_day.sort_values("slot_start").reset_index(drop=True)
    times = day["slot_start"].tolist()
    time_to_idx = {ts: i for i, ts in enumerate(times)}

    records: list[dict] = []
    for i, row in day.iterrows():
        start_ts = row["slot_start"]
        indices = [i]
        for step in range(1, n_slots):
            next_ts = start_ts + timedelta(minutes=SLOT_MINUTES * step)
            j = time_to_idx.get(next_ts)
            if j is None:
                indices = []
                break
            indices.append(j)
        if len(indices) != n_slots:
            continue
        window = day.iloc[indices]
        records.append(
            {
                "slot_start": start_ts,
                "start_time": start_ts.time(),
                "end_time": (start_ts + timedelta(minutes=workout_minutes)).time(),
                "predicted_utilization": float(window["predicted_utilization"].mean()),
            }
        )

    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records)
    result["usage_level"] = result["predicted_utilization"].map(utilization_to_level)
    return result


def recommend_daily_plans(
    start_date: date | str,
    end_date: date | str,
    preferred_location: str,
    wake_time: time,
    weekly_classes: list[ClassBlock],
    workout_minutes: int = 60,
    buffer_minutes: int = 15,
    latest_finish_time: time = REC_CENTER_CLOSE,
    max_alternates: int = 2,
) -> list[DayPlan]:
    """Return ranked gym recommendations for each day in the forecast range."""
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    forecast = predict_range(start, end, preferred_location)

    plans: list[DayPlan] = []
    open_min = _time_to_minutes(REC_CENTER_OPEN)
    close_min = _time_to_minutes(REC_CENTER_CLOSE)
    wake_min = max(open_min, _time_to_minutes(wake_time))
    finish_min = min(close_min, _time_to_minutes(latest_finish_time))
    if finish_min < wake_min:
        raise ValueError("Latest finish time must be after your wake-up time.")
    latest_start = finish_min - workout_minutes

    for current in pd.date_range(start, end, freq="D"):
        current_date = current.date()
        dow = int(current.dayofweek)
        day_name = DAY_NAMES[dow]
        busy = _busy_intervals_for_day(dow, weekly_classes, buffer_minutes)

        day_forecast = forecast[forecast["date"] == current_date]
        candidates = _slots_for_workout(day_forecast, workout_minutes)
        if candidates.empty:
            plans.append(
                DayPlan(
                    date=current_date,
                    day_name=day_name,
                    primary=None,
                    alternates=[],
                    message="No forecast slots available for this day.",
                )
            )
            continue

        feasible_rows: list[dict] = []
        for _, row in candidates.iterrows():
            start_min = _time_to_minutes(row["start_time"])
            end_min = start_min + workout_minutes
            if start_min < wake_min or start_min > latest_start:
                continue
            if end_min > finish_min:
                continue
            if _overlaps_busy(start_min, end_min, busy):
                continue
            feasible_rows.append(row.to_dict())

        if not feasible_rows:
            class_hint = ""
            day_classes = [c for c in weekly_classes if c.day_of_week == dow]
            if day_classes:
                first = min(day_classes, key=lambda c: c.start_time)
                class_hint = (
                    f" No open window before your {first.start_time.strftime('%I:%M %p').lstrip('0')} class "
                    f"(with {buffer_minutes}-min buffer). Try an earlier wake time, a later finish time, "
                    f"or a shorter workout."
                )
            else:
                class_hint = (
                    " Try an earlier wake time, a later latest finish time, or a shorter workout."
                )
            plans.append(
                DayPlan(
                    date=current_date,
                    day_name=day_name,
                    primary=None,
                    alternates=[],
                    message=f"No feasible gym window on this day.{class_hint}",
                )
            )
            continue

        feasible = pd.DataFrame(feasible_rows).sort_values("predicted_utilization")
        picks = feasible.head(1 + max_alternates)

        def to_rec(row: pd.Series) -> SlotRecommendation:
            return SlotRecommendation(
                date=current_date,
                day_name=day_name,
                start_time=row["start_time"],
                end_time=row["end_time"],
                predicted_utilization=float(row["predicted_utilization"]),
                usage_level=str(row["usage_level"]),
                location=preferred_location,
            )

        primary = to_rec(picks.iloc[0])
        alternates = [to_rec(picks.iloc[i]) for i in range(1, len(picks))]
        util_pct = int(round(primary.predicted_utilization * 100))
        plans.append(
            DayPlan(
                date=current_date,
                day_name=day_name,
                primary=primary,
                alternates=alternates,
                message=(
                    f"{day_name} {primary.start_time.strftime('%I:%M %p').lstrip('0')} — "
                    f"{primary.usage_level.title()} crowding (~{util_pct}% full) in "
                    f"{location_display_name(preferred_location)}."
                ),
            )
        )

    return plans


def format_alternates(alternates: list[SlotRecommendation]) -> str:
    if not alternates:
        return ""
    parts = []
    for alt in alternates:
        t = alt.start_time.strftime("%I:%M %p").lstrip("0")
        parts.append(f"{t} ({alt.usage_level})")
    return ", ".join(parts)
