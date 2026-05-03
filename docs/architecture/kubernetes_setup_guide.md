# Kubernetes Setup Guide

This project's architecture already points in a sensible direction:

- Kubernetes hosts training, tracking, storage, serving, and the website.
- MLflow tracks experiments and model metadata.
- Persistent volumes store checkpoints and optionally artifacts.
- MLServer serves inference behind an API.
- The frontend calls the inference API through an ingress.

## Recommended MVP

If this is your first Kubernetes setup, do not start with a large production-style platform. Start with one small cluster and a minimal set of components:

1. One Kubernetes cluster.
2. One namespace for the application, for example `colorflow`.
3. One PostgreSQL database for MLflow metadata.
4. One object store or persistent volume for MLflow artifacts and checkpoints.
5. One training job image.
6. One MLServer deployment for inference.
7. One frontend deployment.
8. One ingress controller for external access.

For a student project, this is enough to train, track, store models, and serve predictions.

## Suggested Architecture

### 1. Cluster Base

Use a managed cluster if possible:

- Google Kubernetes Engine if you are already using GCS.
- Azure Kubernetes Service or Amazon EKS if your team already uses those clouds.

If you only need local learning and demos first:

- Use `kind` or `minikube` locally.
- Move to a managed cluster once the containers and manifests work.

### 2. Training Layer

Run training as a Kubernetes `Job` or `CronJob`, not as a long-running deployment.

Why:

- Training is batch work.
- Jobs restart cleanly when they fail.
- You can allocate more CPU, memory, or GPU only when needed.

Training container responsibilities:

- Pull dataset references from your storage layer.
- Train the model.
- Log parameters, metrics, and artifacts to MLflow.
- Save checkpoints to persistent storage.
- Optionally register the best model in the MLflow Model Registry.

### 3. Tracking Layer

Deploy MLflow with:

- PostgreSQL for backend metadata.
- S3-compatible object storage, GCS bucket, or Azure Blob for artifacts.

Avoid storing MLflow artifacts only on a pod filesystem. Pods are disposable.

If you already use GCS, the cleanest setup is usually:

- MLflow metadata in PostgreSQL.
- MLflow artifacts in GCS.
- Training checkpoints in either GCS or a mounted persistent volume.

### 4. Persistent Storage

You said you want to save checkpoints to persistent storage. That is correct.

Choose one of these patterns:

- Best for cloud-native simplicity: store checkpoints in an object store like GCS or S3.
- Best for shared file semantics: use a PersistentVolumeClaim backed by a cloud disk or network file share.

For ML work, object storage is usually easier long term because:

- it survives pod replacement naturally,
- it is easier to version,
- it is easier to share between training and deployment pipelines.

### 5. Serving Layer

Deploy MLServer as a regular `Deployment` plus a `Service`.

Recommended flow:

1. Training finishes.
2. A chosen model is promoted in MLflow.
3. The serving deployment pulls the selected model artifact.
4. MLServer loads the model and exposes inference over HTTP.

For the first version, keep it simple:

- one model,
- one inference endpoint,
- synchronous HTTP requests,
- CPU serving unless you know you need GPU inference.

### 6. Frontend Layer

Deploy the website separately from MLServer.

Recommended frontend flow:

- Frontend is served as its own deployment.
- Frontend calls a backend API route.
- Backend route forwards the request to MLServer.

Do not let the browser call MLServer directly unless you have a strong reason. A small backend API gives you:

- authentication later,
- request validation,
- rate limiting,
- easier API evolution.

### 7. Networking

Use:

- `Ingress` or a cloud load balancer for public traffic.
- internal cluster `Service` objects for pod-to-pod traffic.

Typical exposure pattern:

- Frontend exposed publicly.
- Backend API exposed publicly or behind the frontend domain.
- MLServer internal only.
- MLflow optionally internal only, unless your team needs external access.

## Do You Need Dagster?

Probably not at the start.

Dagster is useful when you need orchestration:

- scheduled retraining,
- multi-step data pipelines,
- model promotion workflows,
- lineage and operational visibility.

For a first Kubernetes setup, Dagster is optional overhead. Start without it if your immediate goal is:

- train a model in a container,
- log to MLflow,
- persist checkpoints,
- serve inference,
- host a small website.

Add Dagster later if you need repeatable pipelines or scheduled jobs. For an MVP, Kubernetes `Job` or `CronJob` plus GitHub Actions is enough.

## What I Suggest For This Project

Use this rollout order:

1. Containerize the training code.
2. Containerize the inference service with MLServer.
3. Containerize the frontend and its backend API.
4. Deploy MLflow with PostgreSQL and artifact storage.
5. Add a persistent checkpoint location.
6. Create Kubernetes manifests or Helm charts.
7. Expose the frontend through ingress.
8. Keep MLServer internal.
9. Add Dagster only after the basic flow works.

## Practical First Stack

For a beginner-friendly stack, I would use:

- Kubernetes: `kind` locally, then GKE for hosted deployment.
- Registry: GitHub Container Registry.
- Tracking: MLflow.
- Metadata DB: PostgreSQL.
- Artifact store: GCS if available, otherwise MinIO.
- Training execution: Kubernetes `Job`.
- Serving: MLServer `Deployment`.
- Website: separate frontend `Deployment` with a small API service.
- Public entrypoint: NGINX Ingress.

## Minimal Namespace Layout

Keep everything in one namespace first:

- `mlflow`
- `postgres`
- `trainer-job`
- `mlserver`
- `frontend`
- `api`
- `ingress`

Later, if the project grows, split into separate namespaces such as `platform`, `training`, and `serving`.

## Security And Operations Basics

Do not skip these, even in a student project:

- Store secrets in Kubernetes `Secret` objects, not in images.
- Use persistent storage for anything you cannot lose.
- Add readiness and liveness probes to MLServer and the frontend.
- Add resource requests and limits to every workload.
- Keep MLServer private inside the cluster unless there is a clear reason to expose it.
- Back up MLflow metadata if the experiments matter.

## What Not To Overbuild Yet

Avoid these in version one unless your course explicitly requires them:

- service mesh,
- multiple clusters,
- GPU autoscaling,
- complex workflow orchestration,
- model canary deployments,
- full GitOps platform.

## Recommended Next Steps

1. Decide where the cluster will run: local only, or managed cloud.
2. Choose artifact storage: GCS is preferable if available.
3. Define three images: trainer, inference, frontend or API.
4. Create a first local deployment with `kind`.
5. After that works, add hosted ingress and DNS.

## Decision Summary

Yes, you should use Kubernetes for:

- training jobs,
- MLflow,
- persistent storage access,
- MLServer inference,
- hosted website.

No, you do not need Dagster on day one.

Start with Kubernetes plus MLflow plus persistent storage plus MLServer plus a small API and frontend. That gives you the simplest architecture that still matches a real MLOps workflow.