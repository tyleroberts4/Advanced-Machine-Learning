"""Inject feature-importance and sensitivity outputs into notebook 03 without full re-tune."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "notebooks" / "03_regression_modeling.ipynb"


def image_output(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return new_output(output_type="display_data", data={"image/png": data}, metadata={})


def stream_output(text: str) -> dict:
    return new_output(output_type="stream", name="stdout", text=text)


def main():
    metrics = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))
    top = metrics["feature_importance"]["top_features"]
    sens = metrics["split_sensitivity"]

    importance_text = "Top features (CatBoost):\n" + "\n".join(
        f"  {row['feature']}: {row['importance']:.2f}" for row in top[:10]
    )
    sensitivity_text = (
        "Training window sensitivity (CatBoost test RMSE):\n"
        f"  Current train (through {sens['current_train_end']}): {sens['baseline_train_rmse']:.4f}\n"
        f"  Extended train (through {sens['validation_end']}): {sens['extended_train_rmse']:.4f}\n"
        f"  Delta RMSE: {sens['delta_rmse']:.4f}\n"
    )

    nb = nbformat.read(NB_PATH, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if "aggregate_importances" in src:
            cell.outputs = [
                stream_output(importance_text + "\n"),
                image_output(ROOT / "figures" / "feature_importance_regression.png"),
            ]
            cell.execution_count = 1
        elif "extended_pipe.fit" in src:
            cell.outputs = [
                stream_output(sensitivity_text),
                image_output(ROOT / "figures" / "train_window_sensitivity.png"),
            ]
            cell.execution_count = 1

    nbformat.write(nb, NB_PATH)
    print("Patched outputs in", NB_PATH)


if __name__ == "__main__":
    main()
