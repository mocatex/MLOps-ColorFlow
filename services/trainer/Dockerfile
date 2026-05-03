# syntax=docker/dockerfile:1.7
# Training and tuning workloads share this image.
# Override `command:` to switch entry script:
#   command: ["train.py"]                       # default
#   command: ["tune.py", "--n-trials", "20"]
#
# Training data must be materialized manually before building the image
# (for example via `uv run dvc pull`). The image then bakes in `data/images`.

FROM python:3.12-slim

# uv (Astral's package manager) — pulled from its own image so we don't curl-install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System libs that some Python wheels need at runtime.
# git is required by DVC; libgl1 by opencv-style image stacks pulled transitively.
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

# Copy lockfile + project metadata first so the dependency install layer caches.
COPY pyproject.toml uv.lock ./

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Cache mount keeps downloaded wheels between builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra hpo

# Source code + Hydra configs + entry scripts.
COPY src/ src/
COPY configs/ configs/
COPY train.py tune.py ./

# DVC metadata + materialized dataset. The cache and config.local are excluded
# by .dockerignore, so this only ships tracked metadata and pulled data.
COPY .dvc/ .dvc/
COPY .dvcignore ./
COPY data/ data/

RUN if [ ! -d data/images ] || [ -z "$(ls -A data/images 2>/dev/null)" ]; then \
            echo "data/images is empty. Run 'uv run dvc pull' manually before building the trainer image." >&2; \
            exit 1; \
        fi

# Runtime entrypoint: fail fast if the expected dataset is missing.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train.py"]