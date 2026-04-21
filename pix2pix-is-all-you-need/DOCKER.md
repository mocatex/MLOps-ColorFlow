# Docker Usage

This project includes a `Dockerfile` and `docker-compose.yml` that install all
Python dependencies from `requirements.txt`.

## 1) Start with Docker Compose (recommended)

From the repository root:

```bash
docker compose build
docker compose up -d
```

If the docker build fails, in `requirements.txt` change/add the following line:

```yml
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.9.0+cpu 
torchvision==0.24.0+cpu
```

`torch==2.6.0` is pulling CUDA-enabled NVIDIA wheels by default, which are huge (~100–500 MB each) and sometimes can cause timeout or fail in Docker builds. The `+cpu` suffix pulls the CPU-only version, which is much smaller (~50 MB) and faster to install. The `--extra-index-url` is needed to find the CPU wheels.


Then in VS Code (from your local folder), run:

1. **Dev Containers: Reopen in Container**

This uses `.devcontainer/devcontainer.json` and opens `/workspace`
automatically.

## 2) Attach VS Code to the running container

In VS Code:
1. Open Command Palette
2. Run **Dev Containers: Attach to Running Container...**
3. Select `mlops-project` 

If VS Code asks for a project folder, open `/workspace`.

## 3) Verify dependencies (optional)

```bash
docker exec -it mlops-project python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## 4) Stop and remove the container

Compose:

```bash
docker compose down
```

Plain Docker:

```bash
docker rm -f mlops-project
```

## Notes

- File changes are synced both ways because of `-v "$(pwd)":/workspace`.
- If you change `requirements.txt`, rebuild the image:

```bash
docker build -t mlops-project:latest .
```

# Troubleshooting build failures (`HTTP 000`, `No space left on device`)

If a build fails with errors like:
- `CondaHTTPError: HTTP 000 CONNECTION FAILED`
- `OSError: [Errno 28] No space left on device`

```sh
# 1) Check Docker disk usage
docker system df
# TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
# Images          5         4         26.36GB   25.38GB (96%)
# Containers      4         0         8.46GB    8.46GB (100%)
# Local Volumes   1         0         0B        0B
# Build Cache     52        0         46.08GB   21.23GB

# 2) Remove build cache (usually the biggest cleanup)
docker builder prune -af

# 3) Optional: Remove stopped containers
docker container prune -f

# 2) Retry build
docker compose build --no-cache

# 5) If it still fails, also remove unused images
docker image prune -a -f

# 6) If space problems continue, use a stronger cleanup (removes unused volumes too):
docker system prune -a --volumes -f
# Note: this can remove local data in unused Docker volumes.
```
