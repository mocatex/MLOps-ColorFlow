# syntax=docker/dockerfile:1.7
# Shared dependency layer for every colorflow service image.
# Source code is added by the per-service Dockerfiles that FROM this one,
# so dep installs cache across source-only iterations.
#
# CPU build by default. For CUDA, swap the FROM line to e.g.
#   nvidia/cuda:12.4.1-runtime-ubuntu22.04
# and `apt-get install python3.12 python3.12-venv` before installing uv.

FROM python:3.12-slim

# uv (Astral's package manager) — pulled from its own image so we don't curl-install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System libs that some Python wheels need at runtime.
# git is required by DVC; libgl1 by opencv-style image stacks pulled transitively.
# Cache apt indices and downloads across builds.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0

WORKDIR /app

# Copy lockfile + project metadata first so the dep install layer caches.
COPY pyproject.toml uv.lock ./

# `--no-install-project` skips installing the local `colorflow` package
# (the source isn't here yet — it's added by per-service images).
# `--extra hpo` includes optuna so the same image powers both train.py and tune.py.
# UV_LINK_MODE=copy is required when the cache is on a different filesystem
# (which it is, with a buildkit cache mount).
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Cache mount keeps downloaded wheels between builds — turns a 5-min rebuild
# into <1 min when only some deps change.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra hpo

# Make the venv the default Python and put the source dir on PYTHONPATH so
# `import colorflow` works without an editable install.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1
