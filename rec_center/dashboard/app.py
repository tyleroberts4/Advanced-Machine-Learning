"""Streamlit dashboard: Rec Center busyness forecasts and personalized gym recommendations."""

from __future__ import annotations

import sys
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast_engine import model_is_ready, predict_all_locations, predict_range  # noqa: E402
from recommendation import (  # noqa: E402
    ClassBlock,
    DAY_NAMES,
    format_alternates,
    recommend_daily_plans,
)
from rec_center_utils import (  # noqa: E402
    GENERAL_LOCATION,
    MAX_FORECAST_DAYS,
    get_locations,
    location_display_name,
    order_locations_for_ui,
)

st.set_page_config(
    page_title="Rec Center Gym Planner",
    page_icon="🏋️",
    layout="wide",
)

DAY_OPTIONS = list(range(7))


def _time_labels_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sort slots by time and build 12-hour labels in chronological order."""
    sorted_df = df.sort_values("slot_start").copy()
    labels = sorted_df["slot_start"].dt.strftime("%I:%M %p").str.lstrip("0").tolist()
    return sorted_df, labels


def _utilization_bar_chart(df: pd.DataFrame, labels: list[str], height: int = 350) -> None:
    """Bar chart with x-axis times ordered earliest to latest (not alphabetical)."""
    import altair as alt

    plot_df = df.assign(time_label=labels)
    chart = (
        alt.Chart(plot_df)
        .mark_bar(color="#4C72B0")
        .encode(
            x=alt.X(
                "time_label:N",
                sort=labels,
                title="Time",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("predicted_utilization:Q", title="Predicted utilization", scale=alt.Scale(domain=[0, 1])),
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _default_class_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": ["Monday", "Wednesday", "Friday"],
            "start": [time(10, 0), time(10, 0), time(10, 0)],
            "end": [time(11, 0), time(11, 0), time(11, 0)],
        }
    )


def _parse_classes(editor_df: pd.DataFrame) -> list[ClassBlock]:
    blocks: list[ClassBlock] = []
    if editor_df is None or editor_df.empty:
        return blocks
    for _, row in editor_df.iterrows():
        day_label = str(row.get("day", "")).strip()
        if day_label not in DAY_NAMES:
            continue
        start = row.get("start")
        end = row.get("end")
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            continue
        if isinstance(start, str):
            start = pd.to_datetime(start).time()
        if isinstance(end, str):
            end = pd.to_datetime(end).time()
        blocks.append(
            ClassBlock(
                day_of_week=DAY_NAMES.index(day_label),
                start_time=start,
                end_time=end,
            )
        )
    return blocks


def _date_range_controls() -> tuple[date, date]:
    mode = st.radio("Forecast window", ["Next 7 days", "Custom range"], horizontal=True)
    today = date.today()
    if mode == "Next 7 days":
        return today, today + timedelta(days=6)
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=today)
    with col2:
        end = st.date_input("End date", value=today + timedelta(days=6))
    if (end - start).days + 1 > MAX_FORECAST_DAYS:
        st.warning(f"Ranges longer than {MAX_FORECAST_DAYS} days are not supported. Showing first {MAX_FORECAST_DAYS} days.")
        end = start + timedelta(days=MAX_FORECAST_DAYS - 1)
    return start, end


def tab_gym_plan(locations: list[str]) -> None:
    st.subheader("My Gym Plan")
    st.caption("Enter your schedule and preferences — we recommend the least crowded time each day.")

    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox(
            "Preferred workout area",
            order_locations_for_ui(locations),
            format_func=location_display_name,
            help=(
                f"Choose **{location_display_name(GENERAL_LOCATION)}** if you move between floors and rooms; "
                "choose a specific area if you always use the same space."
            ),
        )
        wake_time = st.time_input("Earliest gym start (wake-up)", value=time(7, 0))
    with col2:
        latest_finish_time = st.time_input(
            "Latest finish time (done by)",
            value=time(22, 0),
            help="Your workout must end by this time each day.",
        )
        workout_minutes = st.slider("Workout length (minutes)", 30, 120, 60, step=30)
    with col3:
        buffer_minutes = st.slider("Buffer before class (minutes)", 0, 60, 15, step=5)

    st.markdown("**Class schedule** (weekly repeating)")
    class_df = st.data_editor(
        _default_class_df(),
        num_rows="dynamic",
        column_config={
            "day": st.column_config.SelectboxColumn("Day", options=DAY_NAMES),
            "start": st.column_config.TimeColumn("Start"),
            "end": st.column_config.TimeColumn("End"),
        },
        hide_index=True,
    )

    start_date, end_date = _date_range_controls()

    if st.button("Get recommendations", type="primary"):
        if wake_time >= latest_finish_time:
            st.error("Earliest gym start must be before your latest finish time.")
            return
        weekly_classes = _parse_classes(class_df)
        try:
            plans = recommend_daily_plans(
                start_date=start_date,
                end_date=end_date,
                preferred_location=location,
                wake_time=wake_time,
                weekly_classes=weekly_classes,
                workout_minutes=workout_minutes,
                buffer_minutes=buffer_minutes,
                latest_finish_time=latest_finish_time,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        rows = []
        for plan in plans:
            if plan.primary:
                rows.append(
                    {
                        "Date": plan.date.strftime("%a %m/%d"),
                        "Day": plan.day_name,
                        "Recommended start": plan.primary.start_time.strftime("%I:%M %p").lstrip("0"),
                        "End": plan.primary.end_time.strftime("%I:%M %p").lstrip("0"),
                        "Crowd level": plan.primary.usage_level.title(),
                        "Utilization": f"{plan.primary.predicted_utilization * 100:.0f}%",
                        "Alternates": format_alternates(plan.alternates),
                    }
                )
            else:
                rows.append(
                    {
                        "Date": plan.date.strftime("%a %m/%d"),
                        "Day": plan.day_name,
                        "Recommended start": "—",
                        "End": "—",
                        "Crowd level": "—",
                        "Utilization": "—",
                        "Alternates": plan.message,
                    }
                )

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown("#### Daily summary")
        for plan in plans:
            st.write(plan.message)


def tab_explorer(locations: list[str]) -> None:
    st.subheader("Busyness Explorer")
    st.caption("See predicted crowd levels by hour — no class schedule required.")

    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox(
            "Location",
            order_locations_for_ui(locations),
            format_func=location_display_name,
            key="explorer_location",
        )
    with col2:
        explore_date = st.date_input("Date", value=date.today(), key="explorer_date")

    try:
        day_pred = predict_range(explore_date, explore_date, location)
    except ValueError as exc:
        st.error(str(exc))
        return

    if day_pred.empty:
        st.warning("No predictions for this date.")
        return

    chart_df, time_labels = _time_labels_chronological(day_pred)
    _utilization_bar_chart(chart_df, time_labels)
    st.caption("Utilization scale: 0 = empty, 1 = at capacity. Times shown earliest to latest.")

    display = chart_df[["slot_start", "predicted_utilization", "usage_level"]].copy()
    display["predicted_utilization"] = (display["predicted_utilization"] * 100).round(0).astype(int).astype(str) + "%"
    display.columns = ["Time", "Predicted utilization", "Crowd level"]
    st.dataframe(display, hide_index=True, use_container_width=True)

    with st.expander("Compare all locations (selected day)"):
        try:
            all_locs = predict_all_locations(explore_date, explore_date).sort_values("slot_start")
            pivot = all_locs.pivot_table(
                index="slot_start",
                columns="location",
                values="predicted_utilization",
                aggfunc="mean",
            )
            pivot = pivot.sort_index()
            pivot.index = pivot.index.strftime("%I:%M %p").str.lstrip("0")
            st.dataframe((pivot * 100).round(0).astype(int).astype(str) + "%", use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))


def main() -> None:
    st.title("Cal Poly Rec Center — Gym Time Planner")
    st.markdown(
        "Uses historical Occuspace data and a **CatBoost forecast model** to suggest "
        "when your preferred workout area will be least crowded."
    )

    if not model_is_ready():
        st.error(
            "Forecast model not found. From the `rec_center` folder, run:\n\n"
            "```\npython train_forecast_model.py\n```\n\n"
            "Then restart this app."
        )
        st.stop()

    try:
        from forecast_engine import get_locations as model_locations

        locations = model_locations()
    except FileNotFoundError:
        locations = get_locations()

    tab1, tab2 = st.tabs(["My Gym Plan", "Busyness Explorer"])
    with tab1:
        tab_gym_plan(locations)
    with tab2:
        tab_explorer(locations)


if __name__ == "__main__":
    main()
