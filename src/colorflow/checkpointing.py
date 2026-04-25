from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class Checkpointer:
    """Saves training checkpoints with three retention slots: latest, best, epoch_N.

    Checkpoint payloads are arbitrary dicts — typically containing model + optimizer
    state dicts, the resolved config, and the metrics for that epoch. Keeping the
    config in the payload makes checkpoints self-describing: a serving runtime
    (e.g. MLServer) can rebuild the architecture from the checkpoint alone.
    """

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        monitor: str,
        mode: str = "min",
        save_every: int = 1,
        keep_last: int = 3,
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.keep_last = keep_last
        self.best_metric: float = float("inf") if mode == "min" else float("-inf")
        self._epoch_paths: list[Path] = []

    def _is_better(self, metric: float) -> bool:
        return metric < self.best_metric if self.mode == "min" else metric > self.best_metric

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metrics: dict[str, float],
    ) -> dict[str, Path]:
        """Persist ``state``. Returns a dict of {slot: path} for slots written this call."""
        saved: dict[str, Path] = {}

        latest_path = self.output_dir / f"{self.prefix}_latest.pt"
        torch.save(state, latest_path)
        saved["latest"] = latest_path

        if self.save_every > 0 and epoch % self.save_every == 0:
            epoch_path = self.output_dir / f"{self.prefix}_epoch_{epoch:03d}.pt"
            torch.save(state, epoch_path)
            self._epoch_paths.append(epoch_path)
            self._prune_old()
            saved["epoch"] = epoch_path

        if self.monitor in metrics:
            metric = float(metrics[self.monitor])
            if self._is_better(metric):
                self.best_metric = metric
                best_path = self.output_dir / f"{self.prefix}_best.pt"
                torch.save(state, best_path)
                saved["best"] = best_path

        return saved

    def _prune_old(self) -> None:
        if self.keep_last <= 0:
            return
        while len(self._epoch_paths) > self.keep_last:
            old = self._epoch_paths.pop(0)
            old.unlink(missing_ok=True)

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return torch.load(path, map_location=map_location)
