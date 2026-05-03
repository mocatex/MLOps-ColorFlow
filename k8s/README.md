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

Delete the local cluster:

```bash
kind delete cluster --name colorflow
```

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

## What This Gives You

After applying an overlay, the cluster is ready for later workloads such as:

- training `Job` or `CronJob`
- MLflow deployment
- PostgreSQL deployment
- MLServer deployment
- frontend deployment
- internal API deployment

The base scaffold now already includes a minimal PostgreSQL and MLflow setup.

It also includes a minimal one-shot training `Job` named `demo-trainer` that logs a single run to MLflow and writes a tiny checkpoint file onto the `model-checkpoints` volume.

## Demo Apps

This repo now includes a minimal end-to-end demo:

- `services/trainer/`: one-shot training script that logs a single MLflow run
- `services/ui/`: static frontend served by NGINX
- `services/mlserver/`: custom MLServer runtime serving a toy linear model

Local build and deploy flow:

```bash
docker build -t colorflow-mlflow:local services/mlflow
docker build -t colorflow-trainer:local services/trainer
docker build -t colorflow-ui:local services/ui
docker build -t colorflow-mlserver:local services/mlserver
kind load docker-image colorflow-mlflow:local --name colorflow
kind load docker-image colorflow-trainer:local --name colorflow
kind load docker-image colorflow-ui:local --name colorflow
kind load docker-image colorflow-mlserver:local --name colorflow
kubectl apply -k k8s/overlays/local
```

Once ingress is ready, open `http://localhost` and the UI will call `http://localhost/v2/models/linear-regression/infer` through the same ingress.

To access MLflow locally, use port forwarding instead of public ingress:

```bash
kubectl port-forward -n colorflow svc/mlflow 5000:5000
```

Then open `http://localhost:5000`.

PostgreSQL is kept internal only. MLflow uses PostgreSQL for metadata and the `mlflow-artifacts` persistent volume claim for artifacts.

To inspect the demo training job:

```bash
kubectl get jobs -n colorflow
kubectl logs job/demo-trainer -n colorflow
```

If you want to run it again after it completes:

```bash
kubectl delete job demo-trainer -n colorflow
kubectl apply -k k8s/overlays/local
```

## Why This Is Minimal

This scaffold intentionally avoids:

- Helm charts
- Terraform
- GitOps controllers
- workflow orchestration
- public ingress manifests for your application

Those can be added after the first real services exist.