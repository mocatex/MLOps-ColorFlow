import argparse
import os
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import urlparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a locally registered champion model into a target MLflow registry.",
    )
    parser.add_argument(
        "--source-tracking-uri",
        default="http://localhost:5001",
        help="MLflow tracking URI that contains the local champion model.",
    )
    parser.add_argument(
        "--target-tracking-uri",
        default="http://localhost:5002",
        help="MLflow tracking URI that should serve the promoted model.",
    )
    parser.add_argument(
        "--model-name",
        default="colorflow-model",
        help="Registered model name in both the source and target registries.",
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Model alias to promote.",
    )
    parser.add_argument(
        "--target-experiment-name",
        default=os.environ.get("MLFLOW_TARGET_EXPERIMENT_NAME", "colorflow-gke"),
        help="Target MLflow experiment name for the promotion run.",
    )
    parser.add_argument(
        "--artifact-root",
        default="/outputs/mlruns",
        help="Kept for backward compatibility; no longer needed for upload-based promotion.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/outputs/checkpoints",
        help="Root path for mirrored raw checkpoints in the cluster.",
    )
    return parser.parse_args()


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        client.get_registered_model(model_name)


def is_remote_tracking_uri(tracking_uri: str) -> bool:
    return tracking_uri.startswith(("http://", "https://"))


def is_non_proxied_local_artifact_location(artifact_location: str | None) -> bool:
    if not artifact_location:
        return False
    return artifact_location.startswith("/") or artifact_location.startswith("file://")


def ensure_experiment(
    client: MlflowClient,
    experiment_name: str,
    tracking_uri: str,
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
        if experiment is None:
            raise RuntimeError(
                f"Target experiment '{experiment_name}' was created but could not be reloaded"
            )

    if is_remote_tracking_uri(tracking_uri) and is_non_proxied_local_artifact_location(
        experiment.artifact_location
    ):
        raise RuntimeError(
            f"Target experiment '{experiment_name}' uses non-proxied artifact location "
            f"'{experiment.artifact_location}'. Redeploy the target MLflow server with "
            "'--serve-artifacts --artifacts-destination ...' and use a fresh target "
            "experiment name, or clear the old cluster MLflow metadata before retrying."
        )

    return experiment.experiment_id


def remap_checkpoint_uri(checkpoint_uri: str | None, checkpoint_root: str) -> str | None:
    if not checkpoint_uri:
        return None

    if checkpoint_uri.startswith("gs://"):
        return checkpoint_uri

    parsed = urlparse(checkpoint_uri)
    checkpoint_name = PurePosixPath(parsed.path).name
    if not checkpoint_name:
        return None

    return f"{checkpoint_root.rstrip('/')}/{checkpoint_name}"


def resolve_source_model_uri(
    source_tracking_uri: str,
    artifact_root: str,
    experiment_id: str,
    run_id: str,
    artifact_path: str,
) -> str:
    if source_tracking_uri.startswith("file://"):
        mirrored_path = (
            Path(artifact_root)
            / experiment_id
            / run_id
            / "artifacts"
            / artifact_path
        )
        if mirrored_path.exists():
            return str(mirrored_path)

    return f"runs:/{run_id}/{artifact_path}"


def find_existing_target_version(
    client: MlflowClient,
    model_name: str,
    promoted_run_id: str,
    promoted_from_tracking_uri: str,
):
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        tags = model_version.tags or {}
        if (
            tags.get("promoted_run_id") == promoted_run_id
            and tags.get("promoted_from_tracking_uri") == promoted_from_tracking_uri
        ):
            return model_version
    return None


def wait_for_model_version(client: MlflowClient, model_name: str, version: str) -> None:
    for _ in range(36):
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        import time

        time.sleep(5)

    raise RuntimeError(f"Model version {model_name}/{version} did not become READY")


def main() -> None:
    args = parse_args()

    source_client = MlflowClient(tracking_uri=args.source_tracking_uri)
    target_client = MlflowClient(tracking_uri=args.target_tracking_uri)

    source_version = source_client.get_model_version_by_alias(args.model_name, args.alias)
    source_tags = source_version.tags or {}
    run_id = source_tags.get("selected_run_id") or source_version.run_id
    artifact_path = source_tags.get("selected_artifact_path", "generator")

    if not run_id:
        raise RuntimeError("Source champion model has no run_id to promote")

    source_run = source_client.get_run(run_id)
    source_experiment = source_client.get_experiment(source_run.info.experiment_id)
    if source_experiment is None:
        raise RuntimeError(
            f"Source experiment '{source_run.info.experiment_id}' does not exist"
        )

    selected_checkpoint_uri = remap_checkpoint_uri(
        source_tags.get("selected_checkpoint_uri") or source_run.data.tags.get("checkpoint_uri"),
        args.checkpoint_root,
    )

    ensure_registered_model(target_client, args.model_name)
    target_version = find_existing_target_version(
        target_client,
        args.model_name,
        run_id,
        args.source_tracking_uri,
    )

    if target_version is None:
        promotion_tags = {
            "promoted_from_tracking_uri": args.source_tracking_uri,
            "promoted_run_id": run_id,
            "promoted_source_model_version": source_version.version,
            "selected_run_id": run_id,
            "selected_artifact_path": artifact_path,
            **(
                {"selected_metric_name": source_tags["selected_metric_name"]}
                if "selected_metric_name" in source_tags
                else {}
            ),
            **(
                {"selected_metric": source_tags["selected_metric"]}
                if "selected_metric" in source_tags
                else {}
            ),
            **(
                {"selected_checkpoint_uri": selected_checkpoint_uri}
                if selected_checkpoint_uri
                else {}
            ),
        }

        mlflow.set_tracking_uri(args.source_tracking_uri)
        source_model_uri = resolve_source_model_uri(
            args.source_tracking_uri,
            args.artifact_root,
            source_run.info.experiment_id,
            run_id,
            artifact_path,
        )
        model = mlflow.pytorch.load_model(source_model_uri, map_location="cpu")

        target_experiment_id = ensure_experiment(
            target_client,
            args.target_experiment_name,
            args.target_tracking_uri,
        )
        mlflow.set_tracking_uri(args.target_tracking_uri)
        with mlflow.start_run(
            experiment_id=target_experiment_id,
            run_name=f"promote-{args.model_name}-{run_id[:8]}",
        ) as promotion_run:
            mlflow.set_tags(promotion_tags)
            selected_metric = source_tags.get("selected_metric")
            if selected_metric is not None:
                try:
                    mlflow.log_metric("selected_metric", float(selected_metric))
                except ValueError:
                    pass
            mlflow.pytorch.log_model(model, artifact_path=artifact_path)

        model_uri = f"runs:/{promotion_run.info.run_id}/{artifact_path}"
        target_version = mlflow.register_model(model_uri=model_uri, name=args.model_name)
        wait_for_model_version(target_client, args.model_name, target_version.version)
        for key, value in promotion_tags.items():
            target_client.set_model_version_tag(
                args.model_name,
                target_version.version,
                key,
                value,
            )
        action = "Created"
    else:
        action = "Reused"

    target_client.set_registered_model_alias(args.model_name, args.alias, target_version.version)

    print(
        f"{action} {args.model_name} version {target_version.version} on {args.target_tracking_uri} "
        f"from source run {run_id} and set alias '{args.alias}'"
    )


if __name__ == "__main__":
    main()