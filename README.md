# Kubernetes Cluster Scaffold

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

For host-based Docker Compose runs, local MLflow artifacts are written to `storage/mlops-flow/` and raw checkpoints are written to `storage/mlops-checkpoints/`.

Override Hydra config from the CLI:
> docker compose run --rm training python train.py training.epochs=2 data.batch_size=8

This folder contains a minimal Kubernetes scaffold for this project.

It now deploys a minimal platform slice and a demo application:

- one namespace,
- basic resource defaults,
- one persistent volume claim for shared in-cluster outputs on local Kubernetes,
- one PostgreSQL deployment for MLflow metadata,
- one MLflow deployment with a persistent artifact store,
- one local `kind` cluster configuration,
- one Kustomize overlay for local development,
- one Kustomize overlay for GKE.

# Layout

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
docker build -t colorflow-registry:local services/registry
docker build -t colorflow-trainer:local -f services/trainer/Dockerfile .
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

- deletes any old `trainer` and `registry` jobs,
- applies the local overlay so the platform is up,
- creates the `trainer` job explicitly,
- waits for `trainer` to complete and prints its logs,
- creates `registry`, waits for it to complete, and prints its logs.

If you want to run the same steps manually instead of using the script:

```bash
# clean up any old jobs first
kubectl delete job trainer registry -n colorflow --ignore-not-found
# make sure the platform is up
kubectl apply -k k8s/overlays/local

# load the images into kind
kind load docker-image colorflow-registry:local --name colorflow
kind load docker-image colorflow-trainer:local --name colorflow

# create both jobs from the local job overlay
kubectl apply -k k8s/jobs/local
# wait for the trainer to complete
kubectl wait --for=condition=complete job/trainer -n colorflow
# or to watch it live:
kubectl logs -f job/trainer -n colorflow

# wait for the model registry to complete
kubectl wait --for=condition=complete job/registry -n colorflow
# or to watch it live:
kubectl logs -f job/registry -n colorflow 
```



If you want to watch the run appear in MLflow while the job is running. To access MLflow locally, use port forwarding instead of public ingress:

```bash
kubectl port-forward -n colorflow svc/mlflow 5000:5000
# then open `http://localhost:5000`
```

Check, whether the Python process is alive, how much memory it uses:

```bash
kubectl exec -n colorflow trainer-t5stt -- sh -lc '
  echo "status:";
  grep -E "State|VmRSS|VmSize|Threads|voluntary_ctxt_switches|nonvoluntary_ctxt_switches" /proc/1/status'
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
kubectl logs job/registry -n colorflow
```

# Google Kubernetes Engine Setup

Create the cluster manually first:

```bash
# creata and configure your GCP project and GKE cluster (if you haven't already):
gcloud services enable container.googleapis.com
gcloud container clusters create-auto colorflow --region europe-west6
gcloud container clusters get-credentials colorflow --region europe-west6
```

# Build and Deploy to GKE

```bash
# check if you have set the project 
gcloud config get project
# if not, set it to the correct one
gcloud config set project mlops-colorflow

# enable Artifact Registry in that project
gcloud services enable artifactregistry.googleapis.com --project mlops-colorflow

# create the docker repository once if it does not exist yet
gcloud artifacts repositories create colorflow \
  --repository-format=docker \
  --location=europe-west6 \
  --project=mlops-colorflow

# make sure you have gke.env filled out with the correct values
PROJECT_ID=mlops-colorflow
REGION=europe-west6
REPOSITORY=colorflow
IMAGE_TAG=your-new-tag
APP_HOST=
MLFLOW_ARTIFACT_ROOT=gs://mlops-flow
GCP_SERVICE_ACCOUNT_EMAIL=our-real-service-account@mlops-colorflow.iam.gserviceaccount.com

# build and push all service images to Artifact Registry using the same values
unset TAG
set -a # start exporting all variables to the environment
. ./scripts/gke.env
# use a fresh tag whenever you change the image; this avoids reusing cached `latest`
export TAG=mlserver-oom-fix-20260507
set +a # stop exporting all variables

# configure all GKE overlay placeholders from that one file
./scripts/configure_gke_overlay.sh scripts/gke.env
# make sure docker daemon is running
# then build and push the images referenced by the GKE overlay to Artifact Registry:
./scripts/build_and_push_gke_images.sh scripts/gke.env
# start MLflow first:
kubectl apply -k k8s/stages/gke/mlflow
# both should return "successfully rolled out" before you proceed
kubectl rollout status deployment/postgres -n colorflow 
kubectl rollout status deployment/mlflow -n colorflow

# run local mlflow
docker compose up -d postgres mlflow
# run local registration 
uv run python services/registry/register.py
# mirror the local model files into GCS
gcloud storage rsync --recursive storage/mlops-flow gs://mlops-flow
gcloud storage cp storage/mlops-checkpoints/gan_best.pt gs://mlops-checkpoints/gan_best.pt

# expose MLflow through port forwarding so you can inspect it locally:
kubectl port-forward -n colorflow svc/mlflow 5002:5000
# In another terminal, promote the local champion into the cluster MLflow registry
# It should say soemthing like "set alias 'champion'"
uv run python services/registry/promote_local_model.py \
  --source-tracking-uri "file://$PWD/storage/mlops-flow" \
  --target-tracking-uri http://localhost:5002 \
  --artifact-root gs://mlops-flow \
  --checkpoint-root gs://mlops-checkpoints

# start MLServer after the target champion alias exists:
kubectl apply -k k8s/stages/gke/mlserver
# should return "successfully rolled out" before you proceed:
kubectl rollout status deployment/mlserver -n colorflow
# start the UI and ingress last:
kubectl apply -k k8s/stages/gke/ui
# should return "successfully rolled out"
kubectl rollout status deployment/ui -n colorflow

# Validate:
kubectl get ns colorflow # should show the namespace
kubectl get ingress -n colorflow # should show the ingress with an ADDRESS once it's ready; this is the public entry point to the app on GKE
kubectl get ingress -n colorflow -w # watch it
# then open http://<ADDRESS>/
```

The GKE overlay now targets the native `gce` ingress class instead of `ingress-nginx`.

If `APP_HOST` is empty, the generated ingress omits the host match and you can call the app directly through the load balancer IP. If you later add a real DNS host, set `APP_HOST` and rerun `./scripts/configure_gke_overlay.sh`.

The public entry point on GKE is the `Ingress` address, not any of the `ClusterIP` service addresses. 

If the address appears but the browser still resets or hangs for a short time, GKE is usually still finishing load balancer health checks. Give it another minute or two and retry.

`./scripts/configure_gke_overlay.sh` writes the GKE overlay files from one configuration source. The GKE overlay stores MLflow artifacts in `gs://mlops-flow` and creates a bucket-backed `model-checkpoints` PVC for `gs://mlops-checkpoints`.

For the clean startup order above, use the staged overlays `k8s/stages/gke/mlflow`, `k8s/stages/gke/mlserver`, and `k8s/stages/gke/ui`. `k8s/overlays/gke` still exists if you want to start the whole platform at once.

The GKE path also assumes Workload Identity for GCS access. Configure `GCP_SERVICE_ACCOUNT_EMAIL` in `scripts/gke.env`, then grant that service account access to the MLflow artifact bucket, the checkpoint bucket, and the DVC dataset bucket.

The `colorflow-runtime` Kubernetes service account is used by MLflow, MLServer, the trainer job, and the registry job. On GKE, bind it to your Google service account before you deploy. At a minimum, that Google service account needs storage object access to the buckets used by MLflow artifacts, trainer checkpoints, and DVC data.

The GKE trainer job overlay enables `DVC_PULL_DATA=true`, so the trainer can fetch `data/images.dvc` at startup instead of requiring the dataset to be baked into the image. The same overlay mounts the checkpoint bucket at `/checkpoints` and sets `COLORFLOW_CHECKPOINT_DIR=/checkpoints` for raw `.pt` files.

Training and model registration are on-demand on GKE. Trigger jobs explicitly when you want them:

```bash
# start a training run
kubectl apply -k k8s/jobs/gke/trainer

# after training completes, register the best model
kubectl apply -k k8s/jobs/gke/registry

# list all jobs to see their status:
kubectl get jobs -n colorflow
# to see the pods created by the jobs:
kubectl get pods -n colorflow -o wide
```

# Promote a local champion to GKE

If the current `champion` model was trained locally, mirror its files to GCS and promote it into the GKE MLflow registry like this:

```bash
# assuming you first ran locally:
uv run train.py
uv run register.py

# mirror the local model files into GCS
gcloud storage rsync --recursive storage/mlops-flow gs://mlops-flow

# optional: keep the raw checkpoint bucket in sync too
gcloud storage cp storage/mlops-checkpoints/gan_best.pt gs://mlops-checkpoints/gan_best.pt

# expose GKE MLflow locally, then promote the mirrored artifact into the GKE registry
# open a tunnel to the mlflow service in GKE
kubectl port-forward -n colorflow svc/mlflow 5002:5000
# then in another terminal, promote the local champion into the GKE registry:
uv run python services/registry/promote_local_model.py

# if MLServer and UI are not deployed yet, start them now:
kubectl apply -k k8s/stages/gke/mlserver
kubectl rollout status deployment/mlserver -n colorflow
kubectl apply -k k8s/stages/gke/ui
kubectl rollout status deployment/ui -n colorflow
kubectl get pods -n colorflow

```

# Workflow

Fix workload identity: 

```bash
PROJECT_ID=mlops-colorflow
GSA=260943884277-compute@developer.gserviceaccount.com

gcloud iam service-accounts add-iam-policy-binding "$GSA" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[colorflow/colorflow-runtime]"

gcloud storage buckets add-iam-policy-binding gs://mlops-flow \
  --member="serviceAccount:${GSA}" \
  --role="roles/storage.objectAdmin"

gcloud storage buckets add-iam-policy-binding gs://mlops-checkpoints \
  --member="serviceAccount:${GSA}" \
  --role="roles/storage.objectAdmin"

gcloud storage buckets add-iam-policy-binding gs://mlops-coco \
  --member="serviceAccount:${GSA}" \
  --role="roles/storage.objectViewer"

# restart the pods
kubectl rollout restart deployment/mlflow deployment/mlserver -n colorflow
kubectl rollout status deployment/mlflow -n colorflow
kubectl rollout status deployment/mlserver -n colorflow

# verify:
kubectl port-forward -n colorflow svc/mlserver 8088:8080
curl http://127.0.0.1:8088/v2/health/ready
```


```bash
# Deploy platform only:
kubectl apply -k k8s/overlays/gke

# Trigger training only when you want it:
kubectl apply -k k8s/jobs/gke/trainer
kubectl logs job/trainer -n colorflow

# Trigger model registration only after training completes:
kubectl apply -k k8s/jobs/gke/registry
kubectl logs job/registry -n colorflow

# stop the training and model registry jobs if they are still running:
kubectl delete job trainer registry -n colorflow --ignore-not-found

# list all running pods
kubectl get pods -n colorflow
```

# Clean slate

```bash
# delete all jobs, deployments, and pods in the colorflow namespace
kubectl delete job --all -n colorflow
kubectl delete deployment --all -n colorflow
kubectl delete pod --all -n colorflow

# delete all platform resources and jobs (but keep the cluster running):
kubectl delete --ignore-not-found -k k8s/overlays/gke
# then apply again if you want to restart with a clean slate:
kubectl apply -k k8s/overlays/gke

# if you also want to wipe persistent data too
kubectl delete pvc --all -n colorflow
```


# Serve UI from GKE

```bash
# serve the ui service from the cluster. 
kubectl port-forward -n colorflow svc/ui 8080:80
# then open `http://localhost:8080`
# localhost:8080 is the cluster UI service, not your local mlserver.
```
