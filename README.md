# Kubernetes Cluster Scaffold

This folder contains a minimal Kubernetes scaffold for this project.

It now deploys a minimal platform slice and a demo application:

- one namespace,
- basic resource defaults,
- one persistent volume claim for model checkpoints,
- one PostgreSQL deployment for MLflow metadata,
- one MLflow deployment with a persistent artifact store,
- one local `kind` cluster configuration,
- one Kustomize overlay for local development,
- one Kustomize overlay for GKE.

## Layout

- `kind/cluster.yaml`: local cluster definition for `kind`
- `base/`: shared Kubernetes resources
- `overlays/local/`: local settings for `kind`
- `overlays/gke/`: GKE settings for production hosting

## Local Setup With kind

Prerequisites:

- Docker
- `kind`
- `kubectl`

Create the local cluster:

```bash
kind create cluster --name colorflow --config k8s/kind/cluster.yaml
```

Set the context and apply the base resources:

```bash
kubectl cluster-info --context kind-colorflow
kubectl apply -k k8s/overlays/local
```

Validate:

```bash
kubectl get ns colorflow
kubectl get pvc -n colorflow
kubectl describe limitrange default-resource-defaults -n colorflow
kubectl get pods -n colorflow
```

Note: the `model-checkpoints` PVC can remain `Pending` until the first pod mounts it. That is expected when the storage class uses `WaitForFirstConsumer`.

Optional: install NGINX ingress locally so later you can expose the frontend on `localhost`:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl patch deployment ingress-nginx-controller \
  -n ingress-nginx \
  --type merge \
  --patch-file k8s/kind/ingress-nginx-controller-patch.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s
```

The patch forces the ingress controller onto the `kind` control-plane node, which is where the local host port mappings live.


## GKE Setup

This repo does not provision GKE automatically yet. That is intentional.

GKE needs project-specific values:

- Google Cloud project ID
- region or zone
- node sizing
- billing-enabled project

Create the cluster manually first:

```bash
gcloud services enable container.googleapis.com
gcloud container clusters create-auto colorflow \
  --region europe-west6
gcloud container clusters get-credentials colorflow --region europe-west6
kubectl apply -k k8s/overlays/gke
```

Validate:

```bash
kubectl get ns colorflow
kubectl get pvc -n colorflow
```

The same PVC behavior can happen on GKE if the selected storage class delays binding until a consuming pod is scheduled.

## Demo Apps

This repo now includes a minimal end-to-end demo:

- `services/model_registry/`: registration script that selects the best run and creates a model version in MLflow Model Registry
- `services/trainer/`: one-shot training script that logs a single MLflow run
- `services/ui/`: static frontend served by NGINX
- `services/mlserver/`: minimal inference API that serves the MLflow `champion` model alias

### Local build and deploy flow

```bash
# materialize the training dataset manually before building the trainer image
cd services/trainer
uv run dvc pull # see services/trainer/DVC.MD for details
# build images
docker build -t colorflow-mlflow:local services/mlflow
docker build -t colorflow-model-registry:local services/model_registry
docker build -t colorflow-trainer:local services/trainer
docker build -t colorflow-ui:local services/ui
docker build -t colorflow-mlserver:local services/mlserver
# load images into kind cluster
kind load docker-image colorflow-mlflow:local --name colorflow
kind load docker-image colorflow-model-registry:local --name colorflow
kind load docker-image colorflow-trainer:local --name colorflow
kind load docker-image colorflow-ui:local --name colorflow
kind load docker-image colorflow-mlserver:local --name colorflow
# apply the local overlay to create the resources
kubectl apply -k k8s/overlays/local
```

The trainer image now expects `services/trainer/data/images` to already be present. It does not run `dvc pull` inside the container.

Once ingress is ready, open `http://localhost` and the UI will call `http://localhost/v2/models/linear-regression/infer` through the same ingress.

The serving pod resolves `models:/colorflow-demo-model@champion` from MLflow. If no champion model exists yet, the pod stays live but not ready until the training and registry flow completes.

### Access MLflow locally

To access MLflow locally, use port forwarding instead of public ingress:

```bash
kubectl port-forward -n colorflow svc/mlflow 5000:5000
```

Then open `http://localhost:5000`.

PostgreSQL is kept internal only. MLflow uses PostgreSQL for metadata and the `mlflow-artifacts` persistent volume claim for artifacts.

### Inspect the demo training job

```bash
kubectl get jobs -n colorflow
kubectl logs job/demo-trainer -n colorflow
kubectl logs job/demo-model-registry -n colorflow
```

To inspect a running training job in more detail:

```bash
# job-level status
kubectl get job demo-trainer -n colorflow
kubectl describe job demo-trainer -n colorflow

# pod-level status for the trainer job
kubectl get pods -n colorflow -l job-name=demo-trainer
kubectl describe pod -n colorflow -l job-name=demo-trainer

# stream live trainer logs
kubectl logs -f job/demo-trainer -n colorflow

# watch status changes live
kubectl get job demo-trainer -n colorflow -w
kubectl get pods -n colorflow -l job-name=demo-trainer -w
```

### Run a smoke test

Verify before running training, print the resolved config and exit:

```bash
python train.py tracking=noop --cfg job
```

To get a shell inside the container:

```bash
docker compose -f services/trainer/docker-compose.yml run --rm --entrypoint sh training
```

Then inside the container you should run:

```bash
python train.py tracking=noop training.epochs=10 training.pretrain.enabled=false data.external_data_size=20 data.train_size=16 data.batch_size=2 data.image_size_1=256 data.image_size_2=256 data.pin_memory=false model.generator.use_pretrained_backbone=false
```

If you want to log to MLflow, use the following command instead:

```bash
export MLFLOW_TRACKING_URI=http://host.docker.internal:5000
python train.py training.epochs=10 training.pretrain.enabled=false data.external_data_size=20 data.train_size=16 data.batch_size=2 data.image_size_1=256 data.image_size_2=256 data.pin_memory=false model.generator.use_pretrained_backbone=false
```

## Shutdown

1. Stop the deployed services but keep Kubernetes running:

```sh
kubectl delete --ignore-not-found -k k8s/overlays/local
```
2. Tear everything down, including the local cluster:

```sh
kind delete cluster --name colorflow
```

## Trigger Training Job

If you changed the trainer or registry code, use this workflow before running the jobs:

```bash
# materialize trainer data manually on the host
cd services/trainer
uv run dvc pull # see services/trainer/DVC.MD for details
# rebuild the images
docker build -t colorflow-model-registry:local services/model_registry
docker build -t colorflow-trainer:local services/trainer
# load the rebuilt images into kind
kind load docker-image colorflow-model-registry:local --name colorflow
kind load docker-image colorflow-trainer:local --name colorflow
```

Then run the ordered flow:

```bash
./scripts/run_training_and_register.sh
```

The script assumes the current `colorflow-trainer:local` image was built after you manually materialized `services/trainer/data/images`.

In other words, the trainer no longer runs `dvc pull` inside the container. Data preparation is a manual host-side step, and the pulled `services/trainer/data/images` directory is baked into the trainer image at build time.

The script does four things in order:

- deletes any old `demo-trainer` and `demo-model-registry` jobs,
- applies the local overlay so the trainer job is created,
- waits for `demo-trainer` to complete and prints its logs,
- creates `demo-model-registry`, waits for it to complete, and prints its logs.

Because the serving layer resolves the `champion` alias from MLflow, it will start serving the newly registered best model after the alias changes.

If you want to run the same steps manually instead of using the script:

```bash
kubectl delete job demo-trainer demo-model-registry -n colorflow --ignore-not-found
kubectl apply -k k8s/overlays/local
kubectl logs job/demo-trainer -n colorflow
kubectl apply -f k8s/jobs/model-registry/local/job.yaml
kubectl logs job/demo-model-registry -n colorflow
```

If you want to watch the run appear in MLflow while the job is running:

```bash
kubectl port-forward -n colorflow svc/mlflow 5000:5000
```

Then open `http://localhost:5000`, inspect the `demo-training` experiment, and open the `Models` tab to inspect `colorflow-demo-model` and the `champion` alias.

## Inspect Persistent Storage

The `model-checkpoints` persistent volume claim is backed locally by the `kind` storage provisioner. In this setup, the easiest way to inspect it is directly on the worker node container.

```bash
docker exec colorflow-worker ls -la /var/local-path-provisioner
docker exec colorflow-worker cat /var/local-path-provisioner/pvc-823f4bfe-04ee-471e-9566-00813734c6d1_colorflow_model-checkpoints/demo-checkpoint.json
```

This direct path is specific to the current local cluster. In this setup, the checkpoint PVC is backed by:

```bash
/var/local-path-provisioner/pvc-823f4bfe-04ee-471e-9566-00813734c6d1_colorflow_model-checkpoints
```

On GKE, the inspection flow will differ because the storage backend will not use the local-path provisioner.

If you want to run it again after it completes:

```bash
./scripts/run_training_and_register.sh
```

The delete can fail with `NotFound` because the job is configured with `ttlSecondsAfterFinished: 300`, so Kubernetes removes it automatically about five minutes after it completes.
