from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """Minimal experiment-tracking surface used by the training loop.

    Implementations should be context managers so resources (e.g. an MLflow run)
    open and close around training. The loop never imports MLflow directly —
    this lets HPO trials swap in NoopTracker, and lets Dagster supply its own.
    """

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def log_pytorch_model(self, model, artifact_path: str) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
    def __enter__(self) -> "Tracker": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...


class NoopTracker:
    def log_params(self, params): pass
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, path, artifact_path=None): pass
    def log_pytorch_model(self, model, artifact_path): pass
    def set_tags(self, tags): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return False


class MLflowTracker:
    def __init__(
        self,
        experiment: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run_name = run_name
        self._run = None

    def __enter__(self):
        self._run = self._mlflow.start_run(run_name=self._run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self._mlflow.end_run(status=status)
        return False

    def log_params(self, params):
        flat = _flatten(params)
        # MLflow caps individual param values at 500 chars and rejects None — coerce.
        clean = {k: ("null" if v is None else str(v)[:500]) for k, v in flat.items()}
        self._mlflow.log_params(clean)

    def log_metrics(self, metrics, step=None):
        scalar = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._mlflow.log_metrics(scalar, step=step)

    def log_artifact(self, path, artifact_path=None):
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_pytorch_model(self, model, artifact_path):
        # MLflow 3.x renamed `artifact_path` -> `name` for log_model.
        self._mlflow.pytorch.log_model(model, name=artifact_path)

    def set_tags(self, tags):
        self._mlflow.set_tags(tags)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None


def build_tracker(cfg) -> Tracker:
    backend = cfg.backend
    if backend == "mlflow":
        return MLflowTracker(
            experiment=cfg.experiment,
            run_name=cfg.get("run_name"),
            tracking_uri=cfg.get("tracking_uri"),
        )
    if backend == "noop":
        return NoopTracker()
    raise ValueError(f"Unknown tracker backend: {backend}")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
