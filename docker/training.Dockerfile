# Training and tuning workloads share this image.
# Override `command:` to switch entry script:
#   command: ["train.py"]                       # default
#   command: ["tune.py", "--n-trials", "20"]
#
# The entrypoint runs `dvc pull` automatically when /app/data/images is empty
# (fresh K8s pod) and skips it when the directory is already populated
# (host bind-mount in local dev). Force the behaviour with DVC_PULL=force|skip.

ARG BASE_IMAGE=mlops-colorflow-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Source code + Hydra configs + entry scripts.
COPY src/ src/
COPY configs/ configs/
COPY train.py tune.py ./

# DVC machinery so the container can `dvc pull` data at startup.
# The cache and config.local are excluded by .dockerignore.
COPY .dvc/ .dvc/
COPY .dvcignore ./
COPY data/*.dvc data/

# Hybrid entrypoint: bind-mounted data → skip pull; empty volume → pull.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train.py"]
