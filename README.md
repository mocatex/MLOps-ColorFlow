<<<<<<< HEAD
# Kubernetes Cluster Scaffold

<<<<<<< HEAD
```bash
# create python environment and install dependencies
uv sync
# activate the python environment
source .venv/bin/activate
```

# Local Setup

**See [DVC](./DVC.MD) for Training set File download**

**Run once locally for GCloud project setup**
gcloud config set project mlops-colorflow

Build the dep layer (~5 min)
> docker compose build base
*(if building fails because of oauth)*
> docker logout ghcr.io

Build & Start MLFlow with postgres tracking stack
> docker compose up -d postgres mlflow

MLflow UI
> open http://localhost:5001

# Training & Hyper Parameter Search
one-shot training
> docker compose run --rm training
one-shot HPO     
> docker compose run --rm tuning

Override Hydra config from the CLI:
> docker compose run --rm training python train.py training.epochs=2 data.batch_size=8
=======
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
=======
# Layout
>>>>>>> b16e984 (fixes)

- `kind/cluster.yaml`: local cluster definition for `kind`
- `base/`: shared Kubernetes resources
- `overlays/local/`: local settings for `kind`
- `overlays/gke/`: GKE settings for production hosting

# Prerequisites

- Docker
- `kind`
- `kubectl`

# Deploy Locally

```bash
# materialize the trainer dataset on the host first
# you need GCS credentials set up locally for this to work; see services/trainer/DVC.MD for details
cd services/trainer
uv run dvc pull 

# in case you have an old cluster or resources running, clean up first:
kubectl delete --ignore-not-found -k k8s/overlays/local
kind delete cluster --name colorflow

# create the local cluster:
kind create cluster --name colorflow --config k8s/kind/cluster.yaml

# build all images referenced by the Kubernetes manifests
docker build -t colorflow-mlflow:local services/mlflow
docker build -t colorflow-model-registry:local services/model_registry
docker build -t colorflow-trainer:local services/trainer
docker build -t colorflow-ui:local services/ui
docker build -t colorflow-mlserver:local services/mlserver

# load those host images into the new kind cluster
kind load docker-image colorflow-mlflow:local --name colorflow
kind load docker-image colorflow-ui:local --name colorflow
kind load docker-image colorflow-mlserver:local --name colorflow

# set the context and apply the platform resources:
kubectl cluster-info --context kind-colorflow
kubectl apply -k k8s/overlays/local

# install NGINX ingress locally so you can expose the frontend on `localhost`:
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl patch deployment ingress-nginx-controller \
  -n ingress-nginx \
  --type merge \
  --patch-file k8s/kind/ingress-nginx-controller-patch.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s

# validate:
kubectl get ns colorflow
kubectl get pvc -n colorflow
kubectl describe limitrange default-resource-defaults -n colorflow
kubectl get pods -n colorflow
```

`kubectl apply -k k8s/overlays/local` alone is not enough after `kind delete cluster`, because deleting the cluster also removes all previously loaded local images.

Note: the `model-checkpoints` PVC can remain `Pending` until the first pod mounts it. That is expected when the storage class uses `WaitForFirstConsumer`.

The patch forces the ingress controller onto the `kind` control-plane node, which is where the local host port mappings live.

The trainer image now expects `services/trainer/data/images` to already be present. It does not run `dvc pull` inside the container.

Use `k8s/overlays/local` when you want MLflow, UI, MLServer, PostgreSQL, services, ingress, and PVCs without automatically starting training.

Once ingress is ready, open `http://localhost` and the UI will call `http://localhost/v2/models/colorflow/infer-image` through the same ingress.

The serving pod resolves `models:/colorflow-model@champion` from MLflow. If no champion model exists yet, the pod stays live but not ready until the training and registry flow completes.


# List all Jobs or Pods

```bash
# to see the jobs:
kubectl get jobs -n colorflow
# to see the pods created by the jobs:
kubectl get pods -n colorflow -o wide
```

# Shutdown

```bash
# Stop the deployed services but keep Kubernetes running:
kubectl delete --ignore-not-found -k k8s/overlays/local
# Tear everything down, including the local cluster:
kind delete cluster --name colorflow
```

# Trigger Training Job

Applying `k8s/overlays/local` only creates the platform resources. Trigger training separately with the trainer job manifest.

To run the ordered flow automatically, use the following script:

```bash
./scripts/run_training_and_register.sh
```

The script does four things in order:

- deletes any old `trainer` and `model-registry` jobs,
- applies the local overlay so the platform is up,
- creates the `trainer` job explicitly,
- waits for `trainer` to complete and prints its logs,
- creates `model-registry`, waits for it to complete, and prints its logs.

If you want to run the same steps manually instead of using the script:

```bash
# clean up any old jobs first
kubectl delete job trainer model-registry -n colorflow --ignore-not-found
# make sure the platform is up
kubectl apply -k k8s/overlays/local
# load the images into kind
kind load docker-image colorflow-model-registry:local --name colorflow
kind load docker-image colorflow-trainer:local --name colorflow
# create the trainer job
kubectl apply -f k8s/jobs/trainer/local/job.yaml
# wait for the trainer to complete
kubectl wait --for=condition=complete job/trainer -n colorflow --timeout=1200s
kubectl logs job/trainer -n colorflow
# create the model registry job
kubectl apply -f k8s/jobs/model-registry/local/job.yaml 
# wait for the model registry to complete
kubectl wait --for=condition=complete job/model-registry -n colorflow --timeout=600s
kubectl logs job/model-registry -n colorflow 
```

If your goal is platform first, training later, the intended order is:

```bash
# start the platform
kubectl apply -k k8s/overlays/local

# later, when you want to run training
kubectl delete job trainer model-registry -n colorflow --ignore-not-found
kind load docker-image colorflow-trainer:local --name colorflow
kind load docker-image colorflow-model-registry:local --name colorflow
kubectl apply -f k8s/jobs/trainer/local/job.yaml
kubectl wait --for=condition=complete job/trainer -n colorflow --timeout=1200s
kubectl apply -f k8s/jobs/model-registry/local/job.yaml
kubectl wait --for=condition=complete job/model-registry -n colorflow --timeout=600s
```

If you want to watch the run appear in MLflow while the job is running. To access MLflow locally, use port forwarding instead of public ingress:

```bash
kubectl port-forward -n colorflow svc/mlflow 5000:5000
# then open `http://localhost:5000`
```

## Inspect the training job

To inspect a running training job in more detail:

```bash
# to see the pods created by the jobs:
kubectl logs job/trainer -n colorflow

# job-level status
kubectl get job trainer -n colorflow
kubectl describe job trainer -n colorflow

# pod-level status for the trainer job
kubectl get pods -n colorflow -l job-name=trainer
kubectl describe pod -n colorflow -l job-name=trainer

# stream live trainer logs
kubectl logs -f job/trainer -n colorflow

# watch status changes live
kubectl get job trainer -n colorflow -w
kubectl get pods -n colorflow -l job-name=trainer -w

# to see the model registry logs:
kubectl logs job/model-registry -n colorflow
```

## Run a smoke test

Verify before running training, print the resolved config and exit:

```bash
python train.py tracking=noop --cfg job

# To get a shell inside the container:
docker compose -f services/trainer/docker-compose.yml run --rm --entrypoint sh training

# then inside the container you should run:
python train.py tracking=noop training.epochs=10 training.pretrain.enabled=false data.external_data_size=20 data.train_size=16 data.batch_size=2 data.image_size_1=256 data.image_size_2=256 data.pin_memory=false model.generator.use_pretrained_backbone=false

# if you want to log to MLflow, use the following command instead:
export MLFLOW_TRACKING_URI=http://host.docker.internal:5000
python train.py training.epochs=10 training.pretrain.enabled=false data.external_data_size=20 data.train_size=16 data.batch_size=2 data.image_size_1=256 data.image_size_2=256 data.pin_memory=false model.generator.use_pretrained_backbone=false
```

# Inspect Persistent Storage

The `model-checkpoints` persistent volume claim is backed locally by the `kind` storage provisioner. In this setup, the easiest way to inspect it is directly on the worker node container.

```bash
docker exec -it colorflow-worker sh
cd /var/local-path-provisioner
# find the most recently modified checkpoint file
find /var/local-path-provisioner -type f -name "*.pt" -exec ls -t {} + | head -n 1

# to copy the latest checkpoint file to your host machine, exit the container and run:
exit
docker cp colorflow-worker:/var/local-path-provisioner/dir/to/gan_latest.pt ~/Downloads/gan_latest.pt
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
<<<<<<< HEAD
>>>>>>> 3982e25 (smoke test)
=======

# Google Kubernetes Engine Setup

This repo does not provision GKE automatically yet. That is intentional.

GKE needs project-specific values:

- Google Cloud project ID
- region or zone
- node sizing
- billing-enabled project

Create the cluster manually first:

```bash
gcloud services enable container.googleapis.com
gcloud container clusters create-auto colorflow --region europe-west6
gcloud container clusters get-credentials colorflow --region europe-west6
kubectl apply -k k8s/overlays/gke

# Validate:
kubectl get ns colorflow
kubectl get pvc -n colorflow
```

The same PVC behavior can happen on GKE if the selected storage class delays binding until a consuming pod is scheduled.
>>>>>>> b16e984 (fixes)
