# Overview

## Locally

Storage locations:

1. `PostgreSQL`: local MLflow metadata. Used by local MLflow.
2. `storage/mlops-flow`: local MLflow artifact files. The trainer sends artifact/model files to MLflow through the MLflow tracking API, and MLflow saves those files here. Registry and MLServer access them through MLflow.
3. `storage/mlops-checkpoints`: local raw training checkpoints. Written by trainer. Not read directly by MLServer.
4. `storage/mlops-coco`: local training dataset. Read by trainer. Populated through DVC and mirrors the `gs://mlops-coco` bucket on GCS.

Who uses what locally:

1. `Trainer`: reads `storage/mlops-coco`, writes `storage/mlops-checkpoints`, and sends params/metrics/tags/model files to MLflow through the MLflow tracking API.
2. `MLflow`: writes metadata to local `PostgreSQL` and writes artifact/model files to `storage/mlops-flow`.
3. `Registry job`: reads runs and artifacts through MLflow, registers the serving model.
4. `MLServer`: asks MLflow for `champion`, loads the registered model from `storage/mlops-flow` through MLflow.
5. `UI`: calls MLServer only.

## On GKE

Storage locations:

1. `PostgreSQL`: MLflow metadata. Used by cluster MLflow.
2. `/outputs`: shared persistent volume mounted to the `colorflow-filestore` filestore on GKE.
3. `/outputs/mlruns`: MLflow model artifacts. The trainer sends artifact/model files to MLflow through the MLflow tracking API, and MLflow saves those files here. Registry and MLServer access them through MLflow.
4. `/outputs/checkpoints`: raw training checkpoints. Written by trainer. Not read directly by MLServer.
5. `gs://mlops-coco`: training dataset bucket. Read by trainer.

Who uses what on GKE:

1. `Trainer`: reads `gs://mlops-coco`, writes `/outputs/checkpoints`, and sends params/metrics/tags/model files to MLflow through the MLflow tracking API.
2. `MLflow`: writes metadata to `PostgreSQL` and writes artifact/model files to `/outputs/mlruns`.
3. `Registry job`: reads runs and artifacts through MLflow, registers the serving model.
4. `MLServer`: asks MLflow for `champion`, loads the registered model from `/outputs/mlruns` through MLflow.
5. `UI`: calls MLServer only.

## If You Train Locally First And Then Promote To GKE

1. Local training writes to `storage/mlops-flow` and `storage/mlops-checkpoints`.
2. You copy those into GKE Filestore at `/outputs/mlruns` and `/outputs/checkpoints`.
3. You promote the local champion into the GKE MLflow registry.
4. MLServer then loads the registered model from `/outputs/mlruns`.

# Prerequisites

- Docker
- `kubectl`

# Activate Python Environment

```bash
# create python environment and install dependencies
uv sync
# activate the python environment
source .venv/bin/activate
```

# List all Jobs or Pods

```bash
# to see the jobs:
kubectl get jobs -n colorflow
# to see the pods created by the jobs:
kubectl get pods -n colorflow -o wide
```

# Google Kubernetes Engine Setup

Create the cluster manually first:

```bash
# creata and configure your GCP project and GKE cluster (if you haven't already):
gcloud services enable container.googleapis.com
gcloud container clusters create-auto colorflow --region europe-west6
gcloud container clusters get-credentials colorflow --region europe-west6
```

# Create Filestore Intance

```bash
# enable the file API for GKE to use Filestore as a shared filesystem for MLflow and MLServer artifacts
gcloud services enable file.googleapis.com \
  --project mlops-colorflow

# create a Filestore instance with a 1TiB shared volume
# this is where MLflow and MLServer will read and write artifacts on GKE
# the filestore should then show up under console.cloud.google.com/filestore/instances
gcloud filestore instances create colorflow-filestore \
  --project mlops-colorflow \
  --location europe-west6-b \
  --tier BASIC_HDD \
  --file-share name=colorflow,capacity=1TiB \
  --network name=default 
  # check your VPC network name (should be "default" if you haven't created any custom VPCs)

# get the private IP with:
gcloud filestore instances describe colorflow-filestore \
  --project mlops-colorflow \
  --location europe-west6-b \
  --format="value(networks[0].ipAddresses[0])"

# Then put that value into scripts/gke.env:
MLFLOW_ARTIFACT_NFS_SERVER=10.x.x.x
MLFLOW_ARTIFACT_NFS_PATH=/colorflow
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
MLFLOW_ARTIFACT_STORAGE_MODE=filesystem
MLFLOW_ARTIFACT_ROOT=/outputs/mlruns
MLFLOW_ARTIFACT_NFS_SERVER=<your-filestore-or-nfs-ip>
MLFLOW_ARTIFACT_NFS_PATH=/colorflow
MLFLOW_ARTIFACT_PVC_SIZE=10Gi
GCP_SERVICE_ACCOUNT_EMAIL=our-real-service-account@mlops-colorflow.iam.gserviceaccount.com

# build and push all service images to Artifact Registry using the same values
unset TAG
set -a # start exporting all variables to the environment
. ./scripts/gke.env
# use a fresh tag whenever you change the image; this avoids reusing cached `latest`
export TAG=mlserver-deps-fix-20260508
set +a # stop exporting all variables

# configure all GKE overlay placeholders from that one file
./scripts/configure_gke_overlay.sh scripts/gke.env
# make sure docker daemon is running
# then build and push the images referenced by the GKE overlay to Artifact Registry:
./scripts/build_and_push_gke_images.sh scripts/gke.env
# start MLflow first:
kubectl apply -k k8s/stages/gke/mlflow
# both should return "successfully rolled out" before you proceed. this can take a few minutes:
kubectl rollout status deployment/postgres -n colorflow 
kubectl rollout status deployment/mlflow -n colorflow

# run local mlflow so that the promote script can find the local champion
docker compose up -d postgres mlflow
# run local registration 
uv run python services/registry/register.py

# start a temporary uploader pod that mounts the shared artifact volume
kubectl apply -k k8s/tools/gke/uploader
kubectl wait --for=condition=Ready pod/artifact-uploader -n colorflow --timeout=180s
kubectl exec -n colorflow artifact-uploader -- mkdir -p /outputs/mlruns /outputs/checkpoints
# copy your local MLflow artifact tree and checkpoints into the shared persistent volume
kubectl cp storage/mlops-flow/. colorflow/artifact-uploader:/outputs/mlruns
kubectl cp storage/mlops-checkpoints/. colorflow/artifact-uploader:/outputs/checkpoints

# expose MLflow through port forwarding so you can inspect it locally:
kubectl port-forward -n colorflow svc/mlflow 5002:5000
# In another terminal, promote the local champion into the cluster MLflow registry
# It should say "set alias 'champion'"
uv run python services/registry/promote_local_model.py \
  --source-tracking-uri "file://$PWD/storage/mlops-flow" \
  --target-tracking-uri http://localhost:5002 \
  --artifact-root /outputs/mlruns \
  --checkpoint-root /outputs/checkpoints

# once the files are copied, you can remove the temporary uploader pod
kubectl delete -k k8s/tools/gke/uploader

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

# Promote a local champion to GKE

If the current `champion` model was trained locally, copy its files into the shared persistent volume and promote it into the GKE MLflow registry like this:

```bash
# assuming you first ran locally and already have a local champion model:
uv run python services/registry/register.py

# in order to use kubectl, you need to have the cluster credentials set up locally with:
gcloud container clusters get-credentials colorflow --region europe-west6

# start a temporary uploader pod that mounts the shared artifact volume
kubectl apply -k k8s/tools/gke/uploader
# wait for the uploader pod to be ready before copying files
kubectl wait --for=condition=Ready pod/artifact-uploader -n colorflow --timeout=120s
# make sure the target directories exist before copying files
kubectl exec -n colorflow artifact-uploader -- mkdir -p /outputs/mlruns /outputs/checkpoints
# copy the local model files into the shared persistent volume
kubectl cp storage/mlops-flow/. colorflow/artifact-uploader:/outputs/mlruns
kubectl cp storage/mlops-checkpoints/. colorflow/artifact-uploader:/outputs/checkpoints

# get the list of mlserver pods
kubectl get pods -n colorflow -l app=mlserver
# take the pod name from that output, then:
pod=mlserver-856cdf84df-mnjdw

# copy the promote script into the mlserver pod so it can access the files
# under /outputs and talk to the in-cluster MLflow service
kubectl cp services/registry/promote_local_model.py \
  colorflow/${pod}:/tmp/promote_local_model.py

# then run it inside the pod:
kubectl exec -n colorflow ${pod} -- python /tmp/promote_local_model.py \
  --source-tracking-uri file:///outputs/mlruns \
  --target-tracking-uri http://mlflow:5000 \
  --artifact-root /outputs/mlruns \
  --checkpoint-root /outputs/checkpoints

# remove the temporary uploader pod when you are done
kubectl delete -k k8s/tools/gke/uploader

# if MLServer and UI are not deployed yet, start them now:
kubectl apply -k k8s/stages/gke/mlserver
kubectl rollout status deployment/mlserver -n colorflow
kubectl apply -k k8s/stages/gke/ui
kubectl rollout status deployment/ui -n colorflow
kubectl get pods -n colorflow
# all pods shuld have READY set to 1/1 and STATUS to Running.
```

# Update MLServer to GKE

```bash
# in the project root

# set the environment variables for your GKE cluster and Artifact Registry
set -a
. ./scripts/gke.env
export TAG=mlserver-app-fix-20260513
set +a

# authenticate your local docker client to push to Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
```


For repo-native flow:

```bash
# this builds all images and pushes them to Artifact Registry, then updates the GKE deployment:
./scripts/build_and_push_gke_images.sh scripts/gke.env
kubectl apply -k k8s/overlays/gke
kubectl rollout status deployment/mlserver -n colorflow
kubectl get pods -n colorflow -l app=mlserver

# use an existing base image TAG
docker buildx build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-mlflow:mlserver-mps-fix-20260508" \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-mlserver:${TAG}" \
  --push \
  services/mlserver

kubectl set image deployment/mlserver \
  mlserver="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-mlserver:${TAG}" \
  -n colorflow

# this can take a few minutes and even be unavailable for a short time during the rollout:
kubectl rollout status deployment/mlserver -n colorflow
# check the new image is running:
kubectl get pods -n colorflow -l app=mlserver
kubectl logs -n colorflow <new-pod-name>
```

# Update UI to GKE

```bash
# in the project root

# set the environment variables for your GKE cluster and Artifact Registry
set -a
. ./scripts/gke.env
export TAG=mlserver-app-fix-20260513
set +a

# authenticate your local docker client to push to Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# build and push the new UI image to Artifact Registry:
docker buildx build \
  --platform linux/amd64 \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-ui:${TAG}" \
  --push \
  services/ui

# update the GKE deployment to use the new image:
kubectl set image deployment/ui \
  ui="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-ui:${TAG}" \
  -n colorflow

# this can take a few minutes and even be unavailable for a short time during the rollout:
kubectl rollout status deployment/ui -n colorflow
# check the new image is running:
kubectl get pods -n colorflow -l app=ui
kubectl logs -n colorflow <new-pod-name>
```

# Update the trainer job on GKE

```bash
# in the project root
set -a
. ./scripts/gke.env
export TAG=trainer-new-tag-20260513
set +a

# Update `k8s/jobs/gke/trainer/kustomization.yaml` to set `newTag: ${TAG}` before running `kubectl apply -k k8s/jobs/gke/trainer`.

docker buildx build \
  --platform linux/amd64 \
  --file services/trainer/Dockerfile \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-trainer:${TAG}" \
  --push \
  .

```

# Update the registry job on GKE

```bash
# in the project root
set -a
. ./scripts/gke.env
export TAG=registry-new-tag-20260513
set +a

# Update `k8s/jobs/gke/registry/kustomization.yaml` to set `newTag: ${TAG}` 
# before running `kubectl apply -k k8s/jobs/gke/registry`.

docker buildx build \
  --platform linux/amd64 \
  --file services/registry/Dockerfile \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/colorflow-registry:${TAG}" \
  --push \
  services/registry
```

# Start the trainer and registry jobs on GKE:

```bash
# run trainer first, wait for it to complete, then run registry:
kubectl delete job trainer registry -n colorflow --ignore-not-found && \
kubectl apply -k k8s/jobs/gke/trainer && \
kubectl wait --for=condition=complete job/trainer -n colorflow && \
kubectl logs job/trainer -n colorflow

# then trigger the registry job to register the champion model after training completes:
kubectl delete job registry -n colorflow --ignore-not-found && \
kubectl apply -k k8s/jobs/gke/registry && \
kubectl wait --for=condition=complete job/registry -n colorflow --timeout=5m && \
kubectl logs job/registry -n colorflow

# watch the trainer job status separately if needed:
kubectl get job trainer -n colorflow -w
# or watch the pod status:
kubectl get pods -n colorflow -l job-name=trainer -w
```

To view the experiment in MLflow UI in the cluster:

```bash
# confirm the mlflow service exists first:
kubectl get svc mlflow -n colorflow
# port forward the in-cluster MLflow service to your local machine:
kubectl port-forward -n colorflow svc/mlflow 5002:5000
# then open the experiment page in your browser:
#http://127.0.0.1:5002/#/experiments/1
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

# Apply model registry job (after training completes)

```bash
# 1. stop the training and model registry jobs if they are still running:
# 2. Trigger model registration only after training completes:
# 3. watch the registry job logs (this will block):
kubectl delete job registry -n colorflow --ignore-not-found && \
kubectl apply -k k8s/jobs/gke/registry && \
kubectl wait --for=condition=complete job/registry -n colorflow --timeout=5m && \
kubectl logs job/registry -n colorflow

# watch the job status:
kubectl get job registry -n colorflow -w
```

# Serve UI from GKE

This will forward the cluster UI service to your local machine so you can access it at `http://localhost:8080`. 

```bash
# serve the ui service from the cluster. 
kubectl port-forward -n colorflow svc/ui 8080:80
# then open `http://localhost:8080`
# localhost:8080 is the cluster UI service, not your local mlserver.
```

# General GKE Troubleshooting

Fix workload identity when cluster workloads need Google Cloud access:

```bash
PROJECT_ID=mlops-colorflow
GSA=260943884277-compute@developer.gserviceaccount.com

gcloud iam service-accounts add-iam-policy-binding "$GSA" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[colorflow/colorflow-runtime]"

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

Delete the cluster:

```bash
# list all clusters in the current project and region to verify your cluster is running:
gcloud container clusters list --region europe-west6
# delete the whole cluster if you want to tear down everything:
gcloud container clusters delete colorflow-cluster --region europe-west6 --project mlops-colorflow
```
