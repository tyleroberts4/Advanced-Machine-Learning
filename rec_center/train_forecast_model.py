"""Train and persist CatBoost forecast model for the student dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rec_center_utils import (  # noqa: E402
    RANDOM_STATE,
    get_feature_frame,
    get_locations,
    load_clean_data,
    location_capacity_map,
    models_path,
)

METRICS_PATH = ROOT / "data" / "metrics.json"


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_columns if c != "location"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
            ("num", "passthrough", numeric_cols),
        ]
    )


def load_catboost_params() -> dict:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    raw = metrics["regression"]["catboost"]["params"]
    return {k.replace("model__", ""): v for k, v in raw.items()}


def main() -> None:
    import joblib

    print("Loading data...")
    df = load_clean_data()
    train_df = df[df["split"].isin(["train", "validation"])].copy()
    capacity_map = location_capacity_map(df)

    feature_cols = get_feature_frame(train_df).columns.tolist()
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(feature_cols)),
            (
                "model",
                CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **load_catboost_params()),
            ),
        ]
    )

    print(f"Training on {len(train_df):,} rows...")
    X = get_feature_frame(train_df)
    y = train_df["average_utilization"].values
    pipe.fit(X, y)

    model_path = models_path("catboost_forecast.joblib")
    joblib.dump(pipe, model_path)
    print("Saved", model_path)

    meta = {
        "locations": get_locations(df),
        "capacity": capacity_map,
        "feature_columns": feature_cols,
        "catboost_params": load_catboost_params(),
    }
    meta_path = models_path("location_capacity.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved", meta_path)


if __name__ == "__main__":
    main()
