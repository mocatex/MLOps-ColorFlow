import os
import time
from pathlib import Path

import mlflow
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException  # type: ignore[import-not-found]


def resolve_tracking_uri(experiment_name: str) -> str:
    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit_tracking_uri:
        return explicit_tracking_uri

    local_store = Path(__file__).resolve().parents[2] / "storage" / "mlops-flow"
    if local_store.exists():
        local_tracking_uri = local_store.as_uri()
        client = MlflowClient(tracking_uri=local_tracking_uri)
        if client.get_experiment_by_name(experiment_name) is not None:
            return local_tracking_uri

    for candidate in ("http://mlflow:5000", "http://localhost:5001"):
        try:
            response = requests.get(f"{candidate}/health", timeout=3)
            response.raise_for_status()
        except requests.RequestException:
            continue
        return candidate

    raise RuntimeError(
        "Could not find a local file-backed MLflow store or reach MLflow at http://mlflow:5000 or http://localhost:5001. "
        "Set MLFLOW_TRACKING_URI explicitly before running register.py."
    )


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        client.get_registered_model(model_name)


def run_has_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> bool:
    try:
        artifacts = client.list_artifacts(run_id)
    except Exception:
        return False

    return any(artifact.path == artifact_path for artifact in artifacts)


def metric_sort_key(metric_name: str, metric_value: float, tie_breaker: int = 0):
    if metric_name == "selection_score":
        return (2, metric_value, tie_breaker)
    if metric_name == "best_val_loss_G_L1":
        return (1, -metric_value, tie_breaker)
    if metric_name == "val_loss_G_L1":
        return (0, -metric_value, tie_breaker)
    raise ValueError(f"Unsupported metric for ranking: {metric_name}")


def get_model_version_metric(model_version):
    tags = model_version.tags or {}
    metric_name = tags.get("selected_metric_name")
    metric_value = tags.get("selected_metric")
    if metric_name is None or metric_value is None:
        return None

    try:
        metric_value_float = float(metric_value)
    except ValueError:
        return None

    try:
        ranking_key = metric_sort_key(metric_name, metric_value_float)
    except ValueError:
        return None

    return metric_name, metric_value_float, ranking_key


def get_best_existing_model_version(
    client: MlflowClient,
    model_name: str,
    exclude_version: str,
):
    best_version = None
    best_metric = None

    for model_version in client.search_model_versions(f"name='{model_name}'"):
        if model_version.version == exclude_version:
            continue

        model_metric = get_model_version_metric(model_version)
        if model_metric is None:
            continue

        if best_metric is None or model_metric[2] > best_metric[2]:
            best_version = model_version
            best_metric = model_metric

    return best_version, best_metric


def iter_finished_runs(client: MlflowClient, experiment_id: str):
    page_token = None

    while True:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1000,
            page_token=page_token,
        )

        for run in runs:
            yield run

        page_token = runs.token
        if not page_token:
            break


def select_best_run(
    client: MlflowClient,
    experiment_id: str,
    metric_name: str,
    artifact_path: str,
):
    best_run = None
    best_key = None
    best_metric_name = None
    best_metric_value = None

    for run in iter_finished_runs(client, experiment_id):
        if not run_has_artifact(client, run.info.run_id, artifact_path):
            continue

        metrics = run.data.metrics

        if metric_name in metrics:
            candidate_metric_name = metric_name
            candidate_metric_value = float(metrics[metric_name])
            candidate_key = metric_sort_key(
                candidate_metric_name,
                candidate_metric_value,
                run.info.start_time,
            )
        elif "best_val_loss_G_L1" in metrics:
            candidate_metric_name = "best_val_loss_G_L1"
            candidate_metric_value = float(metrics["best_val_loss_G_L1"])
            candidate_key = metric_sort_key(
                candidate_metric_name,
                candidate_metric_value,
                run.info.start_time,
            )
        elif "val_loss_G_L1" in metrics:
            candidate_metric_name = "val_loss_G_L1"
            candidate_metric_value = float(metrics["val_loss_G_L1"])
            candidate_key = metric_sort_key(
                candidate_metric_name,
                candidate_metric_value,
                run.info.start_time,
            )
        else:
            continue

        if best_key is None or candidate_key > best_key:
            best_run = run
            best_key = candidate_key
            best_metric_name = candidate_metric_name
            best_metric_value = candidate_metric_value

    if best_run is not None:
        return best_run, best_metric_name, best_metric_value, best_key

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
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "colorflow")
    tracking_uri = resolve_tracking_uri(experiment_name)
    model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "colorflow-model")
    metric_name = os.environ.get("SELECTION_METRIC", "selection_score")
    model_artifact_path = os.environ.get("MLFLOW_MODEL_ARTIFACT_PATH", "generator")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' does not exist")

    best_run, selected_metric_name, metric_value, candidate_key = select_best_run(
        client,
        experiment.experiment_id,
        metric_name,
        model_artifact_path,
    )
    run_id = best_run.info.run_id
    selected_checkpoint_uri = best_run.data.tags.get("checkpoint_uri")

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
    if selected_checkpoint_uri:
        client.set_model_version_tag(
            model_name,
            model_version.version,
            "selected_checkpoint_uri",
            selected_checkpoint_uri,
        )

    candidate_model_metric = get_model_version_metric(model_version)
    best_existing_version, best_existing_metric = get_best_existing_model_version(
        client,
        model_name,
        exclude_version=model_version.version,
    )

    champion_version = model_version
    if candidate_model_metric is not None and best_existing_metric is not None:
        existing_metric_name, existing_metric_value, existing_key = best_existing_metric
        if existing_key >= candidate_model_metric[2]:
            champion_version = best_existing_version
            print(
                f"Kept champion version {best_existing_version.version} using "
                f"{existing_metric_name}={existing_metric_value}; "
                f"candidate run {run_id} scored {selected_metric_name}={metric_value}."
            )

    client.set_registered_model_alias(model_name, "champion", champion_version.version)

    action = "Reused" if reused_existing_version else "Registered"
    alias_message = (
        "and promoted to champion"
        if champion_version.version == model_version.version
        else f"and kept champion on version {champion_version.version}"
    )
    print(
        f"{action} model '{model_name}' version {model_version.version} "
        f"for run {run_id} using {selected_metric_name}={metric_value} "
        f"from artifact path '{model_artifact_path}' {alias_message}"
    )


if __name__ == "__main__":
    main()