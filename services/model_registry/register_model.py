import os
import time

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException  # type: ignore[import-not-found]


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        client.get_registered_model(model_name)


def select_best_run(client: MlflowClient, experiment_id: str, metric_name: str):
    order_by = [f"metrics.{metric_name} DESC", "attributes.start_time DESC"]
    filter_string = f"attributes.status = 'FINISHED' and metrics.{metric_name} > -1"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=order_by,
        max_results=1,
    )
    if runs:
        return runs[0]

    raise RuntimeError("No finished runs with the selection metric were found to register")


def wait_for_model_version(client: MlflowClient, model_name: str, version: str) -> None:
    for _ in range(36):
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        time.sleep(5)

    raise RuntimeError(f"Model version {model_name}/{version} did not become READY")


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "colorflow")
    model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "colorflow-demo-model")
    metric_name = os.environ.get("SELECTION_METRIC", "selection_score")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' does not exist")

    best_run = select_best_run(client, experiment.experiment_id, metric_name)
    run_id = best_run.info.run_id
    metric_value = best_run.data.metrics.get(metric_name)

    ensure_registered_model(client, model_name)

    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_for_model_version(client, model_name, model_version.version)

    client.set_model_version_tag(model_name, model_version.version, "selected_metric", str(metric_value))
    client.set_model_version_tag(model_name, model_version.version, "selected_run_id", run_id)
    client.set_registered_model_alias(model_name, "champion", model_version.version)

    print(
        f"Registered model '{model_name}' version {model_version.version} "
        f"from run {run_id} using {metric_name}={metric_value}"
    )


if __name__ == "__main__":
    main()