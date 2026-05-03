import json
import os
import pathlib

import mlflow


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    checkpoints_dir = pathlib.Path("/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("demo-training")

    x_value = 3.0
    prediction = (2.0 * x_value) + 1.0

    checkpoint_path = checkpoints_dir / "demo-checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "model": "linear-regression-demo",
                "formula": "y = 2x + 1",
                "input": x_value,
                "prediction": prediction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="demo-training-job"):
        mlflow.log_param("model_type", "linear_regression_demo")
        mlflow.log_param("formula", "y = 2x + 1")
        mlflow.log_metric("example_input", x_value)
        mlflow.log_metric("example_prediction", prediction)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    print(f"Logged MLflow run with prediction={prediction}")


if __name__ == "__main__":
    main()