
# Model Registry Service

MLflow already has the actual model registry. This extra model registry component is not a second registry backend, it is a thin automation job around MLflow.

It talks directly to MLflow, then adds tags and moves the selected version to the `champion` alias. 

The extra logic is:

1. find the best finished training run by metric and artifact presence
2. register that run’s model artifact in MLflow
3. tag the version with the selection metadata
4. promote it to the serving alias

This logic could be removed and moved to the training job, but this is more decoupled and allows for more complex selection logic in the future.
