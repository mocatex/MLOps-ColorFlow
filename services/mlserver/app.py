import math
import os
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel


class InferenceInput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[float]


class InferenceRequest(BaseModel):
    inputs: list[InferenceInput]


class ResponseOutput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[float]


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    outputs: list[ResponseOutput]


class ChampionModelServer:
    def __init__(self) -> None:
        self.tracking_uri = "http://mlflow:5000"
        self.registered_model_name = "colorflow-demo-model"
        self.registered_model_alias = "champion"
        self.served_model_name = "linear-regression"
        self.client: MlflowClient | None = None
        self.model: Any = None
        self.loaded_version: str | None = None
        self.load_error: str | None = None

    def configure(self, app: FastAPI) -> None:
        self.tracking_uri = app.state.tracking_uri
        self.registered_model_name = app.state.registered_model_name
        self.registered_model_alias = app.state.registered_model_alias
        self.served_model_name = app.state.served_model_name
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def ensure_model(self) -> None:
        if self.client is None:
            raise RuntimeError("MLflow client is not configured")

        try:
            alias_version = self.client.get_model_version_by_alias(
                self.registered_model_name,
                self.registered_model_alias,
            )
            target_version = alias_version.version
            if self.loaded_version == target_version and self.model is not None:
                return

            model_uri = f"models:/{self.registered_model_name}/{target_version}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.loaded_version = target_version
            self.load_error = None
        except Exception as error:
            self.model = None
            self.loaded_version = None
            self.load_error = str(error)
            raise

    def predict(self, values: list[float]) -> tuple[list[float], str]:
        self.ensure_model()
        frame = pd.DataFrame({"x": values})
        raw_predictions = self.model.predict(frame)
        predictions = [float(value) for value in raw_predictions]
        if any(math.isnan(value) or math.isinf(value) for value in predictions):
            raise RuntimeError("Model returned a non-finite prediction")

        return predictions, self.loaded_version or "unknown"


app = FastAPI(title="ColorFlow Inference API")
app.state.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
app.state.registered_model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "colorflow-demo-model")
app.state.registered_model_alias = os.environ.get("MLFLOW_REGISTERED_MODEL_ALIAS", "champion")
app.state.served_model_name = os.environ.get("SERVED_MODEL_NAME", "linear-regression")
server = ChampionModelServer()


@app.on_event("startup")
def startup() -> None:
    server.configure(app)
    try:
        server.ensure_model()
    except Exception:
        # Keep the container alive so readiness can recover once a champion exists.
        pass


@app.get("/v2/health/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/v2/health/ready")
def ready() -> dict[str, str]:
    try:
        server.ensure_model()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Model not ready: {error}") from error

    return {"status": "ready", "model_version": server.loaded_version or "unknown"}


@app.post("/v2/models/{model_name}/infer", response_model=InferenceResponse)
def infer(model_name: str, payload: InferenceRequest) -> InferenceResponse:
    if model_name != app.state.served_model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")

    if not payload.inputs:
        raise HTTPException(status_code=400, detail="At least one input tensor is required")

    values = [float(value) for value in payload.inputs[0].data]

    try:
        predictions, version = server.predict(values)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Inference failed: {error}") from error

    return InferenceResponse(
        model_name=model_name,
        model_version=version,
        outputs=[
            ResponseOutput(
                name="predictions",
                shape=[len(predictions)],
                datatype="FP64",
                data=predictions,
            )
        ],
    )