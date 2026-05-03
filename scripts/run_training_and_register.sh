#!/usr/bin/env bash

# This script runs the training and model registry jobs sequentially.

# Exit immediately if a command exits with a non-zero status
set -euo pipefail 

# Change to the repository root directory
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

namespace="colorflow"

print_job_debug() {
	job_name="$1"
	kubectl describe job "$job_name" -n "$namespace" || true
	pod_name="$(kubectl get pods -n "$namespace" -l job-name="$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
	if [ -n "$pod_name" ]; then
		kubectl describe pod "$pod_name" -n "$namespace" || true
		kubectl logs "$pod_name" -n "$namespace" --all-containers=true || true
	fi
}

wait_for_job() {
	job_name="$1"
	timeout_seconds="${2:-1200}"
	elapsed=0

	while [ "$elapsed" -lt "$timeout_seconds" ]; do
		complete_status="$(kubectl get job "$job_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)"
		failed_status="$(kubectl get job "$job_name" -n "$namespace" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)"

		if [ "$complete_status" = "True" ]; then
			kubectl logs "job/$job_name" -n "$namespace"
			return 0
		fi

		if [ "$failed_status" = "True" ]; then
			echo "Job $job_name failed." >&2
			print_job_debug "$job_name"
			return 1
		fi

		sleep 5
		elapsed=$((elapsed + 5))
	done

	echo "Timed out waiting for job $job_name to complete." >&2
	print_job_debug "$job_name"
	return 1
}

# Cleanup any existing jobs and apply the new ones
kubectl delete job demo-trainer demo-model-registry -n "$namespace" --ignore-not-found
# Apply the training job
kubectl apply -k k8s/overlays/local
wait_for_job demo-trainer
# Apply the model registry job
kubectl apply -f k8s/jobs/model-registry/local/job.yaml
wait_for_job demo-model-registry
