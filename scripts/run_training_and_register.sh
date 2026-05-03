#!/usr/bin/env bash

# This script runs the training and model registry jobs sequentially.

# Exit immediately if a command exits with a non-zero status
set -euo pipefail 

# Change to the repository root directory
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# Cleanup any existing jobs and apply the new ones
kubectl delete job demo-trainer demo-model-registry -n colorflow --ignore-not-found
# Apply the training job
kubectl apply -k k8s/overlays/local
# Wait for the training job to complete and print its logs
kubectl wait --for=condition=complete job/demo-trainer -n colorflow
# Print the logs of the training job
kubectl logs job/demo-trainer -n colorflow
# Apply the model registry job
kubectl apply -f k8s/jobs/model-registry/local/job.yaml
# Wait for the model registry job to complete and print its logs
kubectl wait --for=condition=complete job/demo-model-registry -n colorflow
# Print the logs of the model registry job
kubectl logs job/demo-model-registry -n colorflow
