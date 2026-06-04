"""Fast path: regenerate figures and new metrics using saved hyperparameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings_import = __import__("warnings")
warnings_import.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from execute_project import (  # noqa: E402
    METRICS_PATH,
    baseline_predictions,
    build_keras_classifier,
    build_keras_regressor,
    eval_classification,
    eval_regression,
    figures_path,
    get_feature_frame,
    make_preprocessor,
    plot_feature_importance,
    run_eda,
    run_split_sensitivity,
    split_df,
    transform_frame,
)
from rec_center_utils import RANDOM_STATE, clean_data, load_raw_data, save_clean_data  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _strip_params(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


def main():
    print("Loading and cleaning data...")
    raw = load_raw_data()
    cleaned, clean_report = clean_data(raw)
    save_clean_data(cleaned)
    train_df, val_df, test_df = split_df(cleaned)

    print("Running EDA figures...")
    run_eda(cleaned)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics["cleaning"] = clean_report

    y_train = train_df["average_utilization"].values
    y_val = val_df["average_utilization"].values
    y_test = test_df["average_utilization"].values

    preprocessor = make_preprocessor(get_feature_frame(train_df).columns.tolist())
    preprocessor.fit(get_feature_frame(train_df))
    scaler = StandardScaler()
    X_train_mat = transform_frame(preprocessor, train_df)
    X_val_mat = transform_frame(preprocessor, val_df)
    X_test_mat = transform_frame(preprocessor, test_df)
    X_train_scaled = scaler.fit_transform(X_train_mat)
    X_val_scaled = scaler.transform(X_val_mat)
    X_test_scaled = scaler.transform(X_test_mat)

    reg = metrics["regression"]
    cat_params = reg["catboost"]["params"]
    cat_pipe = Pipeline(
        [
            ("prep", make_preprocessor(get_feature_frame(train_df).columns.tolist())),
            ("model", CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **_strip_params(cat_params))),
        ]
    )
    cat_pipe.named_steps["prep"].fit(get_feature_frame(train_df))
    cat_pipe.fit(get_feature_frame(train_df), y_train)
    cat_test = eval_regression(y_test, cat_pipe.predict(get_feature_frame(test_df)))

    print("Feature importance...")
    metrics["feature_importance"] = plot_feature_importance(cat_pipe)

    print("Training window sensitivity...")
    metrics["split_sensitivity"] = run_split_sensitivity(
        train_df, val_df, test_df, cat_params, cat_test["rmse"]
    )

    print("Peak-hour diagnostic plot...")
    lgbm_params = reg["lightgbm"]["params"]
    lgbm_pipe = Pipeline(
        [
            ("prep", make_preprocessor(get_feature_frame(train_df).columns.tolist())),
            ("model", LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **_strip_params(lgbm_params))),
        ]
    )
    lgbm_pipe.named_steps["prep"].fit(get_feature_frame(train_df))
    lgbm_pipe.fit(get_feature_frame(train_df), y_train)

    nn_params = reg["keras_mlp"]["params"]
    nn_reg = build_keras_regressor(X_train_scaled.shape[1], nn_params)
    nn_reg.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=25,
        batch_size=1024,
        verbose=0,
    )

    peak_mask = (test_df["hour"].between(17, 18)) & (test_df["day_of_week"] < 5)
    peak_df = test_df.loc[peak_mask].copy()
    if len(peak_df) > 0:
        peak_true = peak_df["average_utilization"].values
        peak_lgbm = lgbm_pipe.predict(get_feature_frame(peak_df))
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

    print("Classification confusion matrix...")
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned["usage_level"].astype(str))
    y_test_cls = label_encoder.transform(test_df["usage_level"].astype(str))
    cls_params = metrics["classification"]["catboost"]["params"]
    cat_cls = Pipeline(
        [
            ("prep", make_preprocessor(get_feature_frame(train_df).columns.tolist())),
            ("model", CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, **_strip_params(cls_params))),
        ]
    )
    cat_cls.named_steps["prep"].fit(get_feature_frame(train_df))
    cat_cls.fit(
        get_feature_frame(train_df),
        label_encoder.transform(train_df["usage_level"].astype(str)),
    )
    cat_pred = cat_cls.predict(get_feature_frame(test_df))
    class_order = ["low", "medium", "high"]
    class_labels_display = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test_cls,
        cat_pred,
        labels=[list(label_encoder.classes_).index(c) for c in class_order],
        display_labels=class_labels_display,
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("CatBoost Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(figures_path("classification_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Updated", METRICS_PATH)
    print("split_sensitivity:", metrics["split_sensitivity"])
    print("top features:", metrics["feature_importance"]["top_features"][:3])


if __name__ == "__main__":
    main()
