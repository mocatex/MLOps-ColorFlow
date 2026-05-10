import argparse
from pathlib import PurePosixPath
from urllib.parse import urlparse

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
        "--artifact-root",
        default="/outputs/mlruns",
        help="Root path that contains the mirrored MLflow artifact tree in the cluster.",
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


def build_artifact_source(artifact_root: str, experiment_id: str, run_id: str, artifact_path: str) -> str:
    base = artifact_root.rstrip("/")
    return f"{base}/{experiment_id}/{run_id}/artifacts/{artifact_path.strip('/')}"


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


def find_existing_target_version(
    client: MlflowClient,
    model_name: str,
    promoted_run_id: str,
    artifact_source: str,
):
    for model_version in client.search_model_versions(f"name='{model_name}'"):
        tags = model_version.tags or {}
        if tags.get("promoted_run_id") == promoted_run_id:
            return model_version
        if model_version.source == artifact_source:
            return model_version
    return None


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
    artifact_source = build_artifact_source(
        args.artifact_root,
        source_run.info.experiment_id,
        run_id,
        artifact_path,
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
        artifact_source,
    )

    if target_version is None:
        target_version = target_client.create_model_version(
            name=args.model_name,
            source=artifact_source,
            tags={
                "promoted_from_tracking_uri": args.source_tracking_uri,
                "promoted_run_id": run_id,
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
            },
        )
        action = "Created"
    else:
        action = "Reused"

    target_client.set_registered_model_alias(args.model_name, args.alias, target_version.version)

    print(
        f"{action} {args.model_name} version {target_version.version} on {args.target_tracking_uri} "
        f"from {artifact_source} and set alias '{args.alias}'"
    )


if __name__ == "__main__":
    main()