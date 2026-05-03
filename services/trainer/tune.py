"""Optuna HPO driver for the colorflow training pipeline.

Each trial composes a fresh Hydra config (configs/train.yaml + overrides) and
calls ``train.run(cfg)`` in-process. The minimised metric is whatever
``run`` returns — currently ``best_val_loss_G_L1``.

Examples
--------
    # 20 trials, default search space, in-memory study
    uv run python tune.py

    # Persistent study + more trials
    uv run python tune.py --n-trials 50 --study-name colorflow_v2 \\
        --storage sqlite:///optuna.db

    # Narrow a search range, fix epochs/batch_size for all trials
    # (everything after `--` is forwarded to Hydra as fixed overrides)
    uv run python tune.py --gen-lr 1e-4 5e-4 -- training.epochs=3 data.batch_size=8

    # Restrict the categorical sweep
    uv run python tune.py --gan-modes vanilla
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from train import run

CONFIGS_DIR = Path(__file__).parent / "configs"


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv on ``--``: tune flags before, Hydra overrides after."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def parse_args(tune_argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for colorflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Study control
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="Wall-clock budget in seconds")
    p.add_argument("--study-name", default="colorflow")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (e.g. sqlite:///optuna.db). None = in-memory.")
    p.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    p.add_argument("--seed", type=int, default=42)

    # Search-space defaults — pass MIN MAX as a pair to override
    p.add_argument("--gen-lr", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=[1e-5, 1e-3])
    p.add_argument("--disc-lr", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=[1e-5, 1e-3])
    p.add_argument("--lambda-l1", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=[10.0, 200.0])
    p.add_argument("--beta1", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=[0.0, 0.9])
    p.add_argument("--gan-modes", nargs="+", default=["vanilla", "lsgan"])

    return p.parse_args(tune_argv)


def make_objective(args: argparse.Namespace, hydra_overrides: list[str]):
    # Per-trial defaults: silence MLflow (override with `-- tracking=mlflow`)
    # and isolate each trial's outputs so checkpoints don't collide.
    base_overrides = ["tracking=noop", *hydra_overrides]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "training.gan.gen_lr": trial.suggest_float("gen_lr", *args.gen_lr, log=True),
            "training.gan.disc_lr": trial.suggest_float("disc_lr", *args.disc_lr, log=True),
            "training.gan.beta1": trial.suggest_float("beta1", *args.beta1),
            "model.loss.lambda_l1": trial.suggest_float("lambda_l1", *args.lambda_l1),
            "model.loss.gan_mode": trial.suggest_categorical("gan_mode", args.gan_modes),
        }
        trial_overrides = [f"{k}={v}" for k, v in params.items()]
        trial_overrides.append(
            f"output_dir=outputs/optuna/{args.study_name}/trial_{trial.number:03d}"
        )

        # Hydra's GlobalHydra is a singleton; clear any state from a prior trial
        # before composing the next one.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIGS_DIR.resolve()), version_base=None):
            cfg = compose(config_name="train", overrides=base_overrides + trial_overrides)

        try:
            return run(cfg)
        except Exception as e:
            print(f"[trial {trial.number}] failed: {e!r}", file=sys.stderr)
            raise optuna.TrialPruned() from e

    return objective


def main() -> None:
    tune_argv, hydra_overrides = split_argv(sys.argv[1:])
    args = parse_args(tune_argv)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
        direction=args.direction,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(args, hydra_overrides),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\nBest trial:")
    print(f"  value : {study.best_value:.5f}")
    print(f"  number: {study.best_trial.number}")
    print("  params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
