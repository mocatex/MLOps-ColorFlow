import os
import time

import mlflow
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException  # type: ignore[import-not-found]


def resolve_tracking_uri() -> str:
    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit_tracking_uri:
        return explicit_tracking_uri

    for candidate in ("http://mlflow:5000", "http://localhost:5001"):
        try:
            response = requests.get(f"{candidate}/health", timeout=3)
            response.raise_for_status()
        except requests.RequestException:
            continue
        return candidate

    raise RuntimeError(
        "Could not reach MLflow at http://mlflow:5000 or http://localhost:5001. "
        "Set MLFLOW_TRACKING_URI explicitly before running register.py."
    )


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        client.get_registered_model(model_name)


def run_has_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> bool:
    try:
        response = requests.get(
            f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/artifacts/list",
            params={"run_id": run_id, "path": ""},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return False

    artifacts = response.json().get("files", [])
    return any(artifact.get("path") == artifact_path for artifact in artifacts)


def select_best_run(
    client: MlflowClient,
    tracking_uri: str,
    experiment_id: str,
    metric_name: str,
    artifact_path: str,
):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )

    best_run = None
    best_key = None
    best_metric_name = None
    best_metric_value = None

    for run in runs:
        if not run_has_artifact(tracking_uri, run.info.run_id, artifact_path):
            continue

        metrics = run.data.metrics

        if metric_name in metrics:
            candidate_metric_name = metric_name
            candidate_metric_value = float(metrics[metric_name])
            candidate_key = (2, candidate_metric_value, run.info.start_time)
        elif "best_val_loss_G_L1" in metrics:
            candidate_metric_name = "best_val_loss_G_L1"
            candidate_metric_value = float(metrics["best_val_loss_G_L1"])
            candidate_key = (1, -candidate_metric_value, run.info.start_time)
        elif "val_loss_G_L1" in metrics:
            candidate_metric_name = "val_loss_G_L1"
            candidate_metric_value = float(metrics["val_loss_G_L1"])
            candidate_key = (0, -candidate_metric_value, run.info.start_time)
        else:
            continue

        if best_key is None or candidate_key > best_key:
            best_run = run
            best_key = candidate_key
            best_metric_name = candidate_metric_name
            best_metric_value = candidate_metric_value

    if best_run is not None:
        return best_run, best_metric_name, best_metric_value

    raise RuntimeError("No finished runs with the selection metric were found to register")


def wait_for_model_version(client: MlflowClient, model_name: str, version: str) -> None:
    for _ in range(36):
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        time.sleep(5)

    raise RuntimeError(f"Model version {model_name}/{version} did not become READY")


def find_existing_model_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
    artifact_path: str,
):
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        tags = model_version.tags or {}
        tagged_run_id = tags.get("selected_run_id")
        tagged_artifact_path = tags.get("selected_artifact_path")

        if tagged_run_id == run_id and tagged_artifact_path == artifact_path:
            return model_version

        if tagged_run_id == run_id and tagged_artifact_path is None:
            return model_version

        if model_version.run_id == run_id:
            return model_version

    return None


def main() -> None:
    tracking_uri = resolve_tracking_uri()
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "colorflow")
    model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "colorflow-model")
    metric_name = os.environ.get("SELECTION_METRIC", "selection_score")
    model_artifact_path = os.environ.get("MLFLOW_MODEL_ARTIFACT_PATH", "generator")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' does not exist")

    best_run, selected_metric_name, metric_value = select_best_run(
        client,
        tracking_uri,
        experiment.experiment_id,
        metric_name,
        model_artifact_path,
    )
    run_id = best_run.info.run_id

    ensure_registered_model(client, model_name)

    model_version = find_existing_model_version(
        client,
        model_name,
        run_id,
        model_artifact_path,
    )
    reused_existing_version = model_version is not None
    if model_version is None:
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    wait_for_model_version(client, model_name, model_version.version)

    client.set_model_version_tag(model_name, model_version.version, "selected_metric_name", selected_metric_name)
    client.set_model_version_tag(model_name, model_version.version, "selected_metric", str(metric_value))
    client.set_model_version_tag(model_name, model_version.version, "selected_run_id", run_id)
    client.set_model_version_tag(model_name, model_version.version, "selected_artifact_path", model_artifact_path)
    client.set_registered_model_alias(model_name, "champion", model_version.version)

    action = "Reused" if reused_existing_version else "Registered"
    print(
        f"{action} model '{model_name}' version {model_version.version} "
        f"for run {run_id} using {selected_metric_name}={metric_value} "
        f"from artifact path '{model_artifact_path}'"
    )


if __name__ == "__main__":
    main()