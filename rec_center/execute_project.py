"""Execute the full Rec Center modeling pipeline and save artifacts."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rec_center_utils import (  # noqa: E402
    RANDOM_STATE,
    add_academic_calendar_features,
    add_cyclical_features,
    clean_data,
    figures_path,
    get_feature_frame,
    load_raw_data,
    save_clean_data,
)

sns.set_theme(style="whitegrid", context="notebook")

METRICS_PATH = ROOT / "data" / "metrics.json"


def split_df(df: pd.DataFrame):
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "validation"].copy(),
        df[df["split"] == "test"].copy(),
    )


def make_preprocessor(feature_columns):
    numeric_cols = [c for c in feature_columns if c != "location"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
            ("num", "passthrough", numeric_cols),
        ]
    )


def baseline_predictions(train_df, eval_df):
    means = (
        train_df.groupby(["location", "day_of_week", "hour"])["average_utilization"]
        .mean()
        .rename("pred")
        .reset_index()
    )
    merged = eval_df.merge(means, on=["location", "day_of_week", "hour"], how="left")
    global_mean = train_df["average_utilization"].mean()
    merged["pred"] = merged["pred"].fillna(global_mean)
    return merged["pred"].values


def eval_regression(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def eval_classification(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def tune_model(name, estimator, param_distributions, X_train, y_train, X_val, y_val, n_iter=12):
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    preds = best.predict(X_val)
    return best, search.best_params_, eval_regression(y_val, preds)


def tune_classifier(name, estimator, param_distributions, X_train, y_train, X_val, y_val, n_iter=12):
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    preds = best.predict(X_val)
    return best, search.best_params_, eval_classification(y_val, preds)


def build_keras_regressor(input_dim, params):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(params["units_1"], activation="relu"),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["units_2"], activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="mse")
    return model


def build_keras_classifier(input_dim, params, n_classes=3):
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(params["units_1"], activation="relu"),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["units_2"], activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def optuna_regression(X_train, y_train, X_val, y_val, n_trials=15):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "units_1": trial.suggest_int("units_1", 32, 128, step=32),
            "units_2": trial.suggest_int("units_2", 16, 64, step=16),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        }
        model = build_keras_regressor(X_train.shape[1], params)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=1024,
            verbose=0,
        )
        preds = model.predict(X_val, verbose=0).ravel()
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_model = build_keras_regressor(X_train.shape[1], best_params)
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=1024, verbose=0)
    preds = best_model.predict(X_val, verbose=0).ravel()
    return best_model, best_params, eval_regression(y_val, preds)


def transform_frame(preprocessor, df, include_calendar=True):
    x = get_feature_frame(df, include_calendar=include_calendar)
    return preprocessor.transform(x)


FEATURE_LABELS = {
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "week_of_year": "Week of year",
    "is_weekend": "Weekend",
    "hour_sin": "Hour (cyclical)",
    "hour_cos": "Hour (cyclical)",
    "month_sin": "Month (cyclical)",
    "month_cos": "Month (cyclical)",
    "is_summer": "Summer",
    "is_winter_break": "Winter break",
    "is_spring_break": "Spring break",
    "is_finals_week": "Finals week",
    "capacity": "Capacity",
}


def aggregate_feature_importances(model_pipeline) -> pd.DataFrame:
    """Aggregate CatBoost importances; sum location one-hot columns into Location."""
    prep = model_pipeline.named_steps["prep"]
    model = model_pipeline.named_steps["model"]
    raw_names = prep.get_feature_names_out()
    importances = model.feature_importances_

    grouped: dict[str, float] = {}
    for name, importance in zip(raw_names, importances):
        if name.startswith("cat__location"):
            grouped["Location"] = grouped.get("Location", 0.0) + float(importance)
        else:
            key = name.replace("num__", "")
            label = FEATURE_LABELS.get(key, key)
            grouped[label] = grouped.get(label, 0.0) + float(importance)

    out = pd.DataFrame({"feature": list(grouped.keys()), "importance": list(grouped.values())})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(model_pipeline) -> dict:
    """Save feature importance bar chart and return top features for metrics.json."""
    importance_df = aggregate_feature_importances(model_pipeline)
    top = importance_df.head(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#4C72B0")
    ax.set_title("What Drives Rec Center Crowding (CatBoost Feature Importance)")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_path("feature_importance_regression.png"), dpi=150)
    plt.close(fig)

    return {
        "top_features": [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in importance_df.head(10).iterrows()
        ]
    }


def run_split_sensitivity(train_df, val_df, test_df, cat_params: dict, current_test_rmse: float) -> dict:
    """Retrain CatBoost on train+validation with fixed params; compare test RMSE."""
    from rec_center_utils import TRAIN_END, VAL_END

    extended_df = pd.concat([train_df, val_df], ignore_index=True)
    ext_preprocessor = make_preprocessor(get_feature_frame(extended_df).columns.tolist())
    ext_preprocessor.fit(get_feature_frame(extended_df))

    cat_kwargs = {k.replace("model__", ""): v for k, v in cat_params.items()}
    ext_pipe = Pipeline(
        [
            ("prep", ext_preprocessor),
            ("model", CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **cat_kwargs)),
        ]
    )
    ext_pipe.fit(get_feature_frame(extended_df), extended_df["average_utilization"].values)
    y_test = test_df["average_utilization"].values
    ext_test = eval_regression(y_test, ext_pipe.predict(get_feature_frame(test_df)))

    fig, ax = plt.subplots(figsize=(5, 4))
    comparison = pd.DataFrame(
        {
            "scenario": ["Current train\n(May 2023–Jun 2024)", "Extended train\n(May 2023–Dec 2024)"],
            "test_rmse": [current_test_rmse, ext_test["rmse"]],
        }
    )
    sns.barplot(data=comparison, x="scenario", y="test_rmse", ax=ax, color="#55A868")
    ax.set_title("Training Window Sensitivity (CatBoost Test RMSE)")
    ax.set_ylabel("Test RMSE")
    ax.set_xlabel("")
    for bar, rmse in zip(ax.patches, comparison["test_rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{rmse:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_path("train_window_sensitivity.png"), dpi=150)
    plt.close(fig)

    return {
        "current_train_end": str(TRAIN_END.date()),
        "validation_end": str(VAL_END.date()),
        "baseline_train_rmse": current_test_rmse,
        "extended_train_rmse": ext_test["rmse"],
        "delta_rmse": float(ext_test["rmse"] - current_test_rmse),
        "extended_train_mae": ext_test["mae"],
        "extended_train_r2": ext_test["r2"],
    }


def run_eda(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["average_utilization"], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Average Utilization")
    ax.set_xlabel("Average Utilization")
    fig.tight_layout()
    fig.savefig(figures_path("average_utilization_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    hourly = df.groupby("hour")["average_utilization"].mean().reset_index()
    sns.barplot(data=hourly, x="hour", y="average_utilization", ax=ax, color="#4C72B0")
    ax.set_title("Average Utilization by Hour of Day")
    fig.tight_layout()
    fig.savefig(figures_path("utilization_by_hour.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    weekday = df.groupby("day_of_week")["average_utilization"].mean().reset_index()
    sns.barplot(data=weekday, x="day_of_week", y="average_utilization", ax=ax, color="#55A868")
    ax.set_title("Average Utilization by Day of Week")
    fig.tight_layout()
    fig.savefig(figures_path("utilization_by_weekday.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    loc = df.groupby("location")["average_utilization"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=loc, y="location", x="average_utilization", ax=ax, color="#C44E52")
    ax.set_title("Average Utilization by Location")
    fig.tight_layout()
    fig.savefig(figures_path("utilization_by_location.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    monthly = df.groupby(df["timestamp"].dt.to_period("M"))["average_utilization"].mean()
    monthly.index = monthly.index.astype(str)
    monthly.plot(ax=ax)
    ax.set_title("Monthly Average Utilization Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Utilization")
    fig.tight_layout()
    fig.savefig(figures_path("monthly_utilization_trend.png"), dpi=150)
    plt.close(fig)

    pivot = df.pivot_table(
        values="average_utilization",
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Average Utilization Heatmap by Weekday and Hour")
    fig.tight_layout()
    fig.savefig(figures_path("weekday_hour_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    period = np.where(df["is_summer"] == 1, "Summer", np.where(df["is_winter_break"] == 1, "Winter Break", "In Quarter"))
    period_df = df.assign(academic_period=period)
    sns.barplot(
        data=period_df.groupby("academic_period")["average_utilization"].mean().reset_index(),
        x="academic_period",
        y="average_utilization",
        ax=ax,
        order=["In Quarter", "Summer", "Winter Break"],
        palette="Set2",
    )
    ax.set_title("Utilization by Academic Period")
    fig.tight_layout()
    fig.savefig(figures_path("utilization_by_academic_period.png"), dpi=150)
    plt.close(fig)

    corr_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_summer",
        "is_finals_week",
        "average_utilization",
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figures_path("feature_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    print("Loading and cleaning data...")
    raw = load_raw_data()
    cleaned, clean_report = clean_data(raw)
    save_clean_data(cleaned)
    train_df, val_df, test_df = split_df(cleaned)

    preprocessor = make_preprocessor(get_feature_frame(train_df).columns.tolist())
    preprocessor.fit(get_feature_frame(train_df))

    X_train = transform_frame(preprocessor, train_df)
    X_val = transform_frame(preprocessor, val_df)
    X_test = transform_frame(preprocessor, test_df)
    X_test_scaled = X_test

    y_train = train_df["average_utilization"].values
    y_val = val_df["average_utilization"].values
    y_test = test_df["average_utilization"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("Running EDA figures...")
    run_eda(cleaned)

    metrics = {"cleaning": clean_report, "regression": {}, "classification": {}, "individual_learning": {}}

    print("Baseline...")
    base_pred = baseline_predictions(train_df, test_df)
    metrics["regression"]["historical_mean"] = eval_regression(y_test, base_pred)

    print("Ridge...")
    ridge_pipe = Pipeline([("prep", clone(preprocessor)), ("model", Ridge())])
    ridge_search = RandomizedSearchCV(
        ridge_pipe,
        {"model__alpha": np.logspace(-3, 2, 20)},
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
    )
    ridge_search.fit(get_feature_frame(train_df), y_train)
    ridge_model = ridge_search.best_estimator_
    metrics["regression"]["ridge"] = {
        "params": ridge_search.best_params_,
        "test": eval_regression(y_test, ridge_model.predict(get_feature_frame(test_df))),
    }

    print("Random Forest...")
    rf_pipe = Pipeline([("prep", clone(preprocessor)), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    rf_best, rf_params, rf_val = tune_model(
        "rf",
        rf_pipe,
        {"model__n_estimators": [100, 200, 300], "model__max_depth": [8, 12, 16, None], "model__min_samples_leaf": [1, 2, 5]},
        get_feature_frame(train_df),
        y_train,
        get_feature_frame(val_df),
        y_val,
    )
    metrics["regression"]["random_forest"] = {"params": rf_params, "validation": rf_val, "test": eval_regression(y_test, rf_best.predict(get_feature_frame(test_df)))}

    print("LightGBM...")
    lgbm_pipe = Pipeline([("prep", clone(preprocessor)), ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1))])
    lgbm_best, lgbm_params, lgbm_val = tune_model(
        "lgbm",
        lgbm_pipe,
        {
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [31, 63, 127],
            "model__subsample": [0.7, 0.9, 1.0],
        },
        get_feature_frame(train_df),
        y_train,
        get_feature_frame(val_df),
        y_val,
    )
    metrics["regression"]["lightgbm"] = {"params": lgbm_params, "validation": lgbm_val, "test": eval_regression(y_test, lgbm_best.predict(get_feature_frame(test_df)))}

    print("CatBoost...")
    cat_pipe = Pipeline([("prep", clone(preprocessor)), ("model", CatBoostRegressor(random_state=RANDOM_STATE, verbose=0))])
    cat_best, cat_params, cat_val = tune_model(
        "catboost",
        cat_pipe,
        {
            "model__depth": [6, 8, 10],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__iterations": [300, 500, 700],
            "model__l2_leaf_reg": [1, 3, 5],
        },
        get_feature_frame(train_df),
        y_train,
        get_feature_frame(val_df),
        y_val,
    )
    cat_test_metrics = eval_regression(y_test, cat_best.predict(get_feature_frame(test_df)))
    metrics["regression"]["catboost"] = {"params": cat_params, "validation": cat_val, "test": cat_test_metrics}

    print("Feature importance (CatBoost)...")
    metrics["feature_importance"] = plot_feature_importance(cat_best)

    print("Training window sensitivity...")
    metrics["split_sensitivity"] = run_split_sensitivity(
        train_df, val_df, test_df, cat_params, cat_test_metrics["rmse"]
    )

    print("Calendar ablation (LightGBM without calendar flags)...")
    no_cal_preprocessor = make_preprocessor(get_feature_frame(train_df, include_calendar=False).columns.tolist())
    no_cal_preprocessor.fit(get_feature_frame(train_df, include_calendar=False))
    no_cal_pipe = Pipeline(
        [
            ("prep", no_cal_preprocessor),
            ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **{k.replace("model__", ""): v for k, v in lgbm_params.items()})),
        ]
    )
    no_cal_pipe.fit(get_feature_frame(train_df, include_calendar=False), y_train)
    with_cal = eval_regression(y_test, lgbm_best.predict(get_feature_frame(test_df)))
    without_cal = eval_regression(y_test, no_cal_pipe.predict(get_feature_frame(test_df, include_calendar=False)))
    metrics["individual_learning"]["calendar_ablation"] = {"with_calendar": with_cal, "without_calendar": without_cal}

    print("Keras MLP regression (Optuna)...")
    nn_reg, nn_reg_params, nn_reg_val = optuna_regression(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=12)
    nn_reg_test_preds = nn_reg.predict(X_test_scaled, verbose=0).ravel()
    metrics["regression"]["keras_mlp"] = {
        "params": nn_reg_params,
        "validation": nn_reg_val,
        "test": eval_regression(y_test, nn_reg_test_preds),
    }
    metrics["individual_learning"]["optuna_regression"] = {"best_params": nn_reg_params, "validation": nn_reg_val}

    print("Stacking ensemble...")
    val_preds = np.column_stack(
        [
            lgbm_best.predict(get_feature_frame(val_df)),
            cat_best.predict(get_feature_frame(val_df)),
            nn_reg.predict(X_val_scaled, verbose=0).ravel(),
        ]
    )
    test_preds_base = np.column_stack(
        [
            lgbm_best.predict(get_feature_frame(test_df)),
            cat_best.predict(get_feature_frame(test_df)),
            nn_reg_test_preds,
        ]
    )
    meta = Ridge(alpha=1.0)
    meta.fit(val_preds, y_val)
    ensemble_test = meta.predict(test_preds_base)
    metrics["regression"]["stacking_ensemble"] = {
        "meta_weights": meta.coef_.tolist(),
        "test": eval_regression(y_test, ensemble_test),
        "base_models": ["lightgbm", "catboost", "keras_mlp"],
    }
    metrics["individual_learning"]["stacking_ensemble"] = metrics["regression"]["stacking_ensemble"]

    # Peak-hour diagnostic plot
    peak_mask = (test_df["hour"].between(17, 18)) & (test_df["day_of_week"] < 5)
    peak_df = test_df.loc[peak_mask].copy()
    if len(peak_df) > 0:
        peak_true = peak_df["average_utilization"].values
        peak_lgbm = lgbm_best.predict(get_feature_frame(peak_df))
        peak_nn = nn_reg.predict(scaler.transform(transform_frame(preprocessor, peak_df)), verbose=0).ravel()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(peak_true, peak_lgbm, alpha=0.25, label="LightGBM", s=12)
        ax.scatter(peak_true, peak_nn, alpha=0.25, label="Keras MLP", s=12)
        lims = [0, max(peak_true.max(), peak_lgbm.max(), peak_nn.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("Actual Utilization")
        ax.set_ylabel("Predicted Utilization")
        ax.set_title("Peak Hour Predictions (Weekdays 5-6 PM)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_path("peak_hour_pred_vs_actual.png"), dpi=150)
        plt.close(fig)

    print("Classification models...")
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned["usage_level"].astype(str))
    y_train_cls = label_encoder.transform(train_df["usage_level"].astype(str))
    y_val_cls = label_encoder.transform(val_df["usage_level"].astype(str))
    y_test_cls = label_encoder.transform(test_df["usage_level"].astype(str))
    class_labels = list(label_encoder.classes_)

    log_pipe = Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    log_pipe.fit(get_feature_frame(train_df), y_train_cls)
    log_pred = log_pipe.predict(get_feature_frame(test_df))
    metrics["classification"]["logistic_regression"] = eval_classification(y_test_cls, log_pred)

    rf_cls_pipe = Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    rf_cls_best, rf_cls_params, rf_cls_val = tune_classifier(
        "rf_cls",
        rf_cls_pipe,
        {"model__n_estimators": [200, 300], "model__max_depth": [8, 12, None], "model__min_samples_leaf": [1, 2, 5]},
        get_feature_frame(train_df),
        y_train_cls,
        get_feature_frame(val_df),
        y_val_cls,
    )
    rf_cls_pred = rf_cls_best.predict(get_feature_frame(test_df))
    metrics["classification"]["random_forest"] = {
        "params": rf_cls_params,
        "validation": rf_cls_val,
        "test": eval_classification(y_test_cls, rf_cls_pred),
    }

    lgbm_cls_pipe = Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("model", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
        ]
    )
    lgbm_cls_best, lgbm_cls_params, lgbm_cls_val = tune_classifier(
        "lgbm_cls",
        lgbm_cls_pipe,
        {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31, 63],
        },
        get_feature_frame(train_df),
        y_train_cls,
        get_feature_frame(val_df),
        y_val_cls,
    )
    lgbm_cls_pred = lgbm_cls_best.predict(get_feature_frame(test_df))
    metrics["classification"]["lightgbm"] = {
        "params": lgbm_cls_params,
        "validation": lgbm_cls_val,
        "test": eval_classification(y_test_cls, lgbm_cls_pred),
    }

    cat_cls_pipe = Pipeline(
        [
            ("prep", clone(preprocessor)),
            ("model", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)),
        ]
    )
    cat_cls_best, cat_cls_params, cat_cls_val = tune_classifier(
        "cat_cls",
        cat_cls_pipe,
        {
            "model__depth": [6, 8],
            "model__learning_rate": [0.05, 0.1],
            "model__iterations": [300, 500],
        },
        get_feature_frame(train_df),
        y_train_cls,
        get_feature_frame(val_df),
        y_val_cls,
    )
    cat_cls_pred = cat_cls_best.predict(get_feature_frame(test_df))
    metrics["classification"]["catboost"] = {
        "params": cat_cls_params,
        "validation": cat_cls_val,
        "test": eval_classification(y_test_cls, cat_cls_pred),
    }

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def cls_objective(trial):
        params = {
            "units_1": trial.suggest_int("units_1", 32, 128, step=32),
            "units_2": trial.suggest_int("units_2", 16, 64, step=16),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        }
        model = build_keras_classifier(X_train_scaled.shape[1], params)
        model.fit(X_train_scaled, y_train_cls, validation_data=(X_val_scaled, y_val_cls), epochs=12, batch_size=1024, verbose=0)
        preds = model.predict(X_val_scaled, verbose=0).argmax(axis=1)
        return 1 - f1_score(y_val_cls, preds, average="macro")

    cls_study = optuna.create_study(direction="minimize")
    cls_study.optimize(cls_objective, n_trials=10, show_progress_bar=False)
    nn_cls = build_keras_classifier(X_train_scaled.shape[1], cls_study.best_params)
    nn_cls.fit(X_train_scaled, y_train_cls, validation_data=(X_val_scaled, y_val_cls), epochs=20, batch_size=1024, verbose=0)
    nn_cls_pred = nn_cls.predict(X_test_scaled, verbose=0).argmax(axis=1)
    metrics["classification"]["keras_mlp"] = {
        "params": cls_study.best_params,
        "test": eval_classification(y_test_cls, nn_cls_pred),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test_cls,
        lgbm_cls_pred,
        display_labels=class_labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("LightGBM Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(figures_path("classification_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Regression comparison table
    reg_rows = []
    for name, payload in metrics["regression"].items():
        test_metrics = payload.get("test", payload)
        reg_rows.append({"model": name, **test_metrics})
    reg_table = pd.DataFrame(reg_rows).sort_values("rmse")
    reg_table.to_csv(ROOT / "data" / "regression_results.csv", index=False)

    cls_rows = []
    for name, payload in metrics["classification"].items():
        test_metrics = payload.get("test", payload)
        cls_rows.append({"model": name, **test_metrics})
    cls_table = pd.DataFrame(cls_rows).sort_values("macro_f1", ascending=False)
    cls_table.to_csv(ROOT / "data" / "classification_results.csv", index=False)

    metrics["summary"] = {
        "best_regression_model": reg_table.iloc[0]["model"],
        "best_regression_rmse": reg_table.iloc[0]["rmse"],
        "best_classification_model": cls_table.iloc[0]["model"],
        "best_classification_macro_f1": cls_table.iloc[0]["macro_f1"],
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved metrics to", METRICS_PATH)
    print(reg_table)
    print(cls_table)


if __name__ == "__main__":
    main()
