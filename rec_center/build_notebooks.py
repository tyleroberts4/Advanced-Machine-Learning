"""Generate Rec Center Jupyter notebooks with structured sections."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


def save(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", path)


def setup_cells() -> str:
    return """
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
if (ROOT / "rec_center_utils.py").exists():
    pass
elif (ROOT / "rec_center" / "rec_center_utils.py").exists():
    ROOT = ROOT / "rec_center"
elif (ROOT.parent / "rec_center_utils.py").exists():
    ROOT = ROOT.parent
else:
    raise FileNotFoundError("Could not locate rec_center_utils.py")
sys.path.insert(0, str(ROOT))

from rec_center_utils import (
    TRAIN_END,
    VAL_END,
    clean_data,
    data_path,
    figures_path,
    load_clean_data,
    load_raw_data,
    save_clean_data,
)

sns.set_theme(style="whitegrid", context="notebook")
""".strip()


def notebook_01():
    cells = [
        md(
            "# 01 — Data Cleaning and Feature Engineering\n\n"
            "This notebook loads the Occuspace Rec Center export, applies cleaning rules, "
            "engineers time and academic-calendar features, assigns a time-based train/validation/test split, "
            "and saves `data/rec_center_clean.parquet` for downstream modeling."
        ),
        code(setup_cells()),
        md(
            "## Load raw data\n\n"
            "Source workbook: `datasets/rec_center_usage.xlsx` (~223k rows, six locations, 30-minute intervals)."
        ),
        code(
            """
raw = load_raw_data()
raw.head()
"""
        ),
        md(
            "## Cleaning decisions\n\n"
            "- Drop duplicate `location` + `timestamp` pairs (keep first after sorting).\n"
            "- Cap `average_utilization` at 1.0 for modeling while retaining the raw value.\n"
            "- Exclude peak occupancy/utilization from predictors in modeling notebooks to avoid same-interval leakage.\n"
            "- Engineer cyclical time features and Cal Poly academic calendar flags."
        ),
        code(
            """
cleaned, report = clean_data(raw)
report
"""
        ),
        code(
            """
cleaned[["location", "timestamp", "average_utilization", "average_utilization_raw", "usage_level", "split"]].head()
"""
        ),
        md(
            "## Time-based split\n\n"
            "- **Train:** through 2024-06-30\n"
            "- **Validation:** through 2024-12-31\n"
            "- **Test:** remaining recent months (held out for final evaluation)"
        ),
        code(
            """
cleaned["split"].value_counts()
"""
        ),
        code(
            """
output_path = save_clean_data(cleaned)
output_path
"""
        ),
    ]
    save("01_data_cleaning.ipynb", cells)


def notebook_02():
    cells = [
        md(
            "# 02 — Exploratory Data Analysis\n\n"
            "Reproduces proposal figures and adds academic-period and correlation views that inform feature choices."
        ),
        code(setup_cells()),
        code(
            """
df = load_clean_data()
df.shape
"""
        ),
        md("## Target distribution"),
        code(
            """
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["average_utilization"], bins=50, kde=True, ax=ax)
ax.set_title("Distribution of Average Utilization")
fig.savefig(figures_path("average_utilization_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md("## Utilization by hour, weekday, and location"),
        code(
            """
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(data=df.groupby("hour")["average_utilization"].mean().reset_index(), x="hour", y="average_utilization", ax=axes[0])
axes[0].set_title("By Hour")
sns.barplot(data=df.groupby("day_of_week")["average_utilization"].mean().reset_index(), x="day_of_week", y="average_utilization", ax=axes[1])
axes[1].set_title("By Weekday")
loc = df.groupby("location")["average_utilization"].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=loc, y="location", x="average_utilization", ax=axes[2])
axes[2].set_title("By Location")
fig.tight_layout()
for name, ax in zip(["utilization_by_hour.png", "utilization_by_weekday.png", "utilization_by_location.png"], axes):
    ax.figure.savefig(figures_path(name), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md("## Monthly trend and weekday-hour heatmap"),
        code(
            """
monthly = df.groupby(df["timestamp"].dt.to_period("M"))["average_utilization"].mean()
fig, ax = plt.subplots(figsize=(9, 4))
monthly.plot(ax=ax)
ax.set_title("Monthly Average Utilization Trend")
fig.savefig(figures_path("monthly_utilization_trend.png"), dpi=150, bbox_inches="tight")
plt.show()

pivot = df.pivot_table(values="average_utilization", index="day_of_week", columns="hour", aggfunc="mean")
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
ax.set_title("Weekday-Hour Heatmap")
fig.savefig(figures_path("weekday_hour_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md("## Additional insights for modeling"),
        code(
            """
period = pd.Series("In Quarter", index=df.index)
period[df["is_summer"] == 1] = "Summer"
period[df["is_winter_break"] == 1] = "Winter Break"
period_df = df.assign(academic_period=period)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=period_df.groupby("academic_period")["average_utilization"].mean().reset_index(),
    x="academic_period",
    y="average_utilization",
    order=["In Quarter", "Summer", "Winter Break"],
    ax=ax,
)
ax.set_title("Utilization by Academic Period")
fig.savefig(figures_path("utilization_by_academic_period.png"), dpi=150, bbox_inches="tight")
plt.show()

corr_cols = ["hour", "day_of_week", "month", "is_weekend", "is_summer", "is_finals_week", "average_utilization"]
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.savefig(figures_path("feature_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md(
            "### EDA takeaways\n\n"
            "- Peak crowding occurs on weekday afternoons (especially 4–6 PM).\n"
            "- Track and 2nd Floor areas run hotter than 1st Floor and Lower Exercise Room.\n"
            "- Summer and winter break periods are materially quieter.\n"
            "- Nonlinear time interactions justify tree/boosting models over a simple linear baseline."
        ),
    ]
    save("02_eda.ipynb", cells)


def notebook_03():
    cells = [
        md(
            "# 03 — Regression Modeling\n\n"
            "Predict `average_utilization` using a time-aware split. Compare baselines, linear, tree/boosting, "
            "neural network, and a stacking ensemble. Includes individual-learning sections for calendar ablation, "
            "Optuna tuning, and ensemble analysis."
        ),
        code(
            setup_cells()
            + """

import json
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rec_center_utils import RANDOM_STATE, get_feature_frame

METRICS_PATH = ROOT / "data" / "metrics.json"
"""
        ),
        code(
            """
df = load_clean_data()
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

feature_cols = get_feature_frame(train_df).columns.tolist()
numeric_cols = [c for c in feature_cols if c != "location"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("num", "passthrough", numeric_cols),
])

X_train = get_feature_frame(train_df)
X_val = get_feature_frame(val_df)
X_test = get_feature_frame(test_df)
y_train = train_df["average_utilization"].values
y_val = val_df["average_utilization"].values
y_test = test_df["average_utilization"].values

preprocessor.fit(X_train)
X_train_mat = preprocessor.transform(X_train)
X_val_mat = preprocessor.transform(X_val)
X_test_mat = preprocessor.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mat)
X_val_scaled = scaler.transform(X_val_mat)
X_test_scaled = scaler.transform(X_test_mat)

def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
"""
        ),
        md("## Baseline: historical mean by location × weekday × hour"),
        code(
            """
means = train_df.groupby(["location", "day_of_week", "hour"])["average_utilization"].mean().reset_index(name="pred")
base = test_df.merge(means, on=["location", "day_of_week", "hour"], how="left")
base["pred"] = base["pred"].fillna(train_df["average_utilization"].mean())
baseline_metrics = metrics(y_test, base["pred"].values)
baseline_metrics
"""
        ),
        md("## Linear and tree/boosting models with tuning"),
        code(
            """
results = []

ridge = Pipeline([
    ("prep", preprocessor),
    ("model", Ridge()),
])
ridge_search = RandomizedSearchCV(ridge, {"model__alpha": np.logspace(-3, 2, 20)}, n_iter=10, scoring="neg_root_mean_squared_error", cv=3, random_state=RANDOM_STATE)
ridge_search.fit(X_train, y_train)
ridge_best = ridge_search.best_estimator_
results.append({"model": "ridge", **metrics(y_test, ridge_best.predict(X_test))})

rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
])
rf_search = RandomizedSearchCV(
    rf,
    {"model__n_estimators": [100, 200, 300], "model__max_depth": [8, 12, 16, None], "model__min_samples_leaf": [1, 2, 5]},
    n_iter=12,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
results.append({"model": "random_forest", **metrics(y_test, rf_best.predict(X_test))})

lgbm = Pipeline([
    ("prep", preprocessor),
    ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)),
])
lgbm_search = RandomizedSearchCV(
    lgbm,
    {"model__n_estimators": [200, 400, 600], "model__learning_rate": [0.03, 0.05, 0.1], "model__num_leaves": [31, 63, 127], "model__subsample": [0.7, 0.9, 1.0]},
    n_iter=12,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
lgbm_search.fit(X_train, y_train)
lgbm_best = lgbm_search.best_estimator_
results.append({"model": "lightgbm", **metrics(y_test, lgbm_best.predict(X_test))})

cat = Pipeline([
    ("prep", preprocessor),
    ("model", CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)),
])
cat_search = RandomizedSearchCV(
    cat,
    {"model__depth": [6, 8, 10], "model__learning_rate": [0.03, 0.05, 0.1], "model__iterations": [300, 500, 700], "model__l2_leaf_reg": [1, 3, 5]},
    n_iter=12,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
cat_search.fit(X_train, y_train)
cat_best = cat_search.best_estimator_
results.append({"model": "catboost", **metrics(y_test, cat_best.predict(X_test))})

pd.DataFrame(results).sort_values("rmse")
"""
        ),
        md(
            "## Individual learning — Member A: Academic calendar ablation\n\n"
            "Compare LightGBM with and without academic calendar flags to quantify their value."
        ),
        code(
            """
no_cal_cols = get_feature_frame(train_df, include_calendar=False).columns.tolist()
no_cal_numeric = [c for c in no_cal_cols if c != "location"]
no_cal_prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("num", "passthrough", no_cal_numeric),
])
no_cal_pipe = Pipeline([
    ("prep", no_cal_prep),
    ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **{k.replace("model__", ""): v for k, v in lgbm_search.best_params_.items()})),
])
no_cal_pipe.fit(get_feature_frame(train_df, include_calendar=False), y_train)
ablation = pd.DataFrame([
    {"features": "with_calendar", **metrics(y_test, lgbm_best.predict(X_test))},
    {"features": "without_calendar", **metrics(y_test, no_cal_pipe.predict(get_feature_frame(test_df, include_calendar=False)))},
])
ablation
"""
        ),
        md(
            "## Individual learning — Member B: Optuna tuning for Keras MLP\n\n"
            "Neural networks benefit from architecture search; Optuna explores units, dropout, and learning rate efficiently."
        ),
        code(
            """
import optuna
import tensorflow as tf
from tensorflow.keras import layers

optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.keras.utils.set_random_seed(RANDOM_STATE)

def build_mlp(params):
    model = tf.keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(params["units_1"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["units_2"], activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="mse")
    return model

def objective(trial):
    params = {
        "units_1": trial.suggest_int("units_1", 32, 128, step=32),
        "units_2": trial.suggest_int("units_2", 16, 64, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }
    model = build_mlp(params)
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=15, batch_size=1024, verbose=0)
    preds = model.predict(X_val_scaled, verbose=0).ravel()
    return np.sqrt(mean_squared_error(y_val, preds))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
nn_best = build_mlp(study.best_params)
nn_best.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=20, batch_size=1024, verbose=0)
nn_preds = nn_best.predict(X_test_scaled, verbose=0).ravel()
results.append({"model": "keras_mlp", **metrics(y_test, nn_preds)})
study.best_params
"""
        ),
        md(
            "## Individual learning — Member C: Stacking ensemble\n\n"
            "Stack diverse base learners (LightGBM, CatBoost, Keras MLP) with a Ridge meta-learner fit on validation predictions."
        ),
        code(
            """
val_stack = np.column_stack([
    lgbm_best.predict(X_val),
    cat_best.predict(X_val),
    nn_best.predict(X_val_scaled, verbose=0).ravel(),
])
test_stack = np.column_stack([
    lgbm_best.predict(X_test),
    cat_best.predict(X_test),
    nn_preds,
])
meta = Ridge(alpha=1.0)
meta.fit(val_stack, y_val)
ensemble_preds = meta.predict(test_stack)
results.append({"model": "stacking_ensemble", **metrics(y_test, ensemble_preds)})

regression_table = pd.DataFrame(results).sort_values("rmse")
regression_table
"""
        ),
        md("## Neural vs non-neural comparison (peak weekday hours)"),
        code(
            """
peak = test_df[(test_df["hour"].between(17, 18)) & (test_df["day_of_week"] < 5)].copy()
peak_true = peak["average_utilization"].values
peak_lgbm = lgbm_best.predict(get_feature_frame(peak))
peak_nn = nn_best.predict(scaler.transform(preprocessor.transform(get_feature_frame(peak))), verbose=0).ravel()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(peak_true, peak_lgbm, alpha=0.3, label="LightGBM", s=12)
ax.scatter(peak_true, peak_nn, alpha=0.3, label="Keras MLP", s=12)
lims = [0, max(peak_true.max(), peak_lgbm.max(), peak_nn.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("Actual Utilization")
ax.set_ylabel("Predicted Utilization")
ax.set_title("Peak Hour Predictions (Weekdays 5-6 PM)")
ax.legend()
fig.savefig(figures_path("peak_hour_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
    ]
    save("03_regression_modeling.ipynb", cells)


def notebook_04():
    cells = [
        md(
            "# 04 — Classification Modeling\n\n"
            "Predict low / medium / high usage categories derived from average utilization thresholds "
            "(<0.30, 0.30–0.60, >0.60) and compare linear, tree/boosting, and neural classifiers."
        ),
        code(
            setup_cells()
            + """

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from rec_center_utils import CLASS_HIGH, CLASS_LOW, RANDOM_STATE, get_feature_frame
"""
        ),
        code(
            """
df = load_clean_data()
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

le = LabelEncoder()
le.fit(df["usage_level"].astype(str))
y_train = le.transform(train_df["usage_level"].astype(str))
y_val = le.transform(val_df["usage_level"].astype(str))
y_test = le.transform(test_df["usage_level"].astype(str))

feature_cols = get_feature_frame(train_df).columns.tolist()
numeric_cols = [c for c in feature_cols if c != "location"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("num", "passthrough", numeric_cols),
])

X_train = get_feature_frame(train_df)
X_val = get_feature_frame(val_df)
X_test = get_feature_frame(test_df)
preprocessor.fit(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(preprocessor.transform(X_train))
X_test_scaled = scaler.transform(preprocessor.transform(X_test))

print(f"Thresholds: low < {CLASS_LOW}, high > {CLASS_HIGH}")
"""
        ),
        md("## Model comparison"),
        code(
            """
results = []

logistic = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
logistic.fit(X_train, y_train)
log_pred = logistic.predict(X_test)
results.append({"model": "logistic_regression", "accuracy": accuracy_score(y_test, log_pred), "macro_f1": f1_score(y_test, log_pred, average="macro")})

rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])
rf_search = RandomizedSearchCV(rf, {"model__n_estimators": [200, 300], "model__max_depth": [8, 12, None], "model__min_samples_leaf": [1, 2, 5]}, n_iter=10, scoring="f1_macro", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
rf_pred = rf_best.predict(X_test)
results.append({"model": "random_forest", "accuracy": accuracy_score(y_test, rf_pred), "macro_f1": f1_score(y_test, rf_pred, average="macro")})

lgbm = Pipeline([
    ("prep", preprocessor),
    ("model", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
])
lgbm_search = RandomizedSearchCV(lgbm, {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1], "model__num_leaves": [31, 63]}, n_iter=10, scoring="f1_macro", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
lgbm_search.fit(X_train, y_train)
lgbm_best = lgbm_search.best_estimator_
lgbm_pred = lgbm_best.predict(X_test)
results.append({"model": "lightgbm", "accuracy": accuracy_score(y_test, lgbm_pred), "macro_f1": f1_score(y_test, lgbm_pred, average="macro")})

cat = Pipeline([
    ("prep", preprocessor),
    ("model", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)),
])
cat_search = RandomizedSearchCV(cat, {"model__depth": [6, 8], "model__learning_rate": [0.05, 0.1], "model__iterations": [300, 500]}, n_iter=8, scoring="f1_macro", cv=3, random_state=RANDOM_STATE, n_jobs=-1)
cat_search.fit(X_train, y_train)
cat_best = cat_search.best_estimator_
cat_pred = cat_best.predict(X_test)
results.append({"model": "catboost", "accuracy": accuracy_score(y_test, cat_pred), "macro_f1": f1_score(y_test, cat_pred, average="macro")})

import optuna
import tensorflow as tf
from tensorflow.keras import layers

optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.keras.utils.set_random_seed(RANDOM_STATE)

def build_classifier(params):
    model = tf.keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(params["units_1"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["units_2"], activation="relu"),
        layers.Dense(len(le.classes_), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def cls_objective(trial):
    params = {
        "units_1": trial.suggest_int("units_1", 32, 128, step=32),
        "units_2": trial.suggest_int("units_2", 16, 64, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    }
    model = build_classifier(params)
    model.fit(X_train_scaled, y_train, validation_data=(scaler.transform(preprocessor.transform(X_val)), y_val), epochs=12, batch_size=1024, verbose=0)
    preds = model.predict(scaler.transform(preprocessor.transform(X_val)), verbose=0).argmax(axis=1)
    return 1 - f1_score(y_val, preds, average="macro")

cls_study = optuna.create_study(direction="minimize")
cls_study.optimize(cls_objective, n_trials=8)
nn_cls = build_classifier(cls_study.best_params)
nn_cls.fit(X_train_scaled, y_train, validation_data=(scaler.transform(preprocessor.transform(X_val)), y_val), epochs=15, batch_size=1024, verbose=0)
nn_pred = nn_cls.predict(X_test_scaled, verbose=0).argmax(axis=1)
results.append({"model": "keras_mlp", "accuracy": accuracy_score(y_test, nn_pred), "macro_f1": f1_score(y_test, nn_pred, average="macro")})

classification_table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
classification_table
"""
        ),
        md("## Confusion matrix for best gradient boosting classifier"),
        code(
            """
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, lgbm_pred, display_labels=le.classes_, cmap="Blues", ax=ax, colorbar=False)
ax.set_title("LightGBM Confusion Matrix (Test Set)")
fig.savefig(figures_path("classification_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md(
            "### Metric justification\n\n"
            "- **Macro-F1** is the primary metric because it treats low/medium/high classes equally.\n"
            "- **Accuracy** is reported for business readability but can hide poor performance on high-crowd periods."
        ),
    ]
    save("04_classification_modeling.ipynb", cells)


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
