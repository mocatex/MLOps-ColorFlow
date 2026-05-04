import os
from io import BytesIO
from typing import Any

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from mlflow import MlflowClient
from PIL import Image
from pydantic import BaseModel
from skimage.color import lab2rgb, rgb2lab


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
        self.registered_model_name = "colorflow-model"
        self.registered_model_alias = "champion"
        self.served_model_name = "colorflow"
        self.client: MlflowClient | None = None
        self.model: Any = None
        self.loaded_version: str | None = None
        self.load_error: str | None = None
        self.device = torch.device(os.environ.get("MODEL_DEVICE", "cpu"))
        self.image_size = int(os.environ.get("MODEL_IMAGE_SIZE", "256"))

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
            self.model = mlflow.pytorch.load_model(model_uri, dst_path=None)
            self.model.to(self.device)
            self.model.eval()
            self.loaded_version = target_version
            self.load_error = None
        except Exception as error:
            self.model = None
            self.loaded_version = None
            self.load_error = str(error)
            raise

    def predict(self, l_channel: np.ndarray) -> tuple[np.ndarray, str]:
        self.ensure_model()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        l_tensor = torch.from_numpy(l_channel).to(self.device)
        with torch.no_grad():
            ab_tensor = self.model(l_tensor)
        rgb = lab_to_rgb(l_tensor, ab_tensor)

        if not np.isfinite(rgb).all():
            raise RuntimeError("Model returned a non-finite RGB image")

        return rgb.astype(np.float32), self.loaded_version or "unknown"


def lab_to_rgb(l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    l_tensor = (l_tensor.detach().cpu() + 1.0) * 50.0
    ab_tensor = ab_tensor.detach().cpu() * 128.0
    lab = torch.cat([l_tensor, ab_tensor], dim=1).permute(0, 2, 3, 1).numpy()
    rgb = [lab2rgb(sample) for sample in lab]
    return np.stack(rgb, axis=0)


def image_bytes_to_l_tensor(image_bytes: bytes, image_size: int) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {error}") from error

    image = image.resize((image_size, image_size))
    rgb = np.asarray(image, dtype=np.float32)
    lab = rgb2lab(rgb).astype(np.float32)
    l_channel = (lab[..., 0] / 50.0) - 1.0
    return l_channel[np.newaxis, np.newaxis, ...]


def rgb_batch_to_png_bytes(rgb_batch: np.ndarray) -> bytes:
    rgb = np.clip(rgb_batch[0] * 255.0, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_l_tensor(model_name: str, payload: InferenceRequest) -> np.ndarray:
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="At least one input tensor is required")

    tensor = payload.inputs[0]
    expected_rank = {3, 4}
    if len(tensor.shape) not in expected_rank:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' expects shape [1,H,W] or [N,1,H,W] for the L channel, "
                f"got {tensor.shape}"
            ),
        )

    array = np.asarray(tensor.data, dtype=np.float32)
    expected_size = int(np.prod(tensor.shape))
    if array.size != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Input data length {array.size} does not match shape {tensor.shape}",
        )

    array = array.reshape(tensor.shape)
    if array.ndim == 3:
        array = np.expand_dims(array, axis=1)

    if array.shape[1] != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' expects a single L channel, got shape {array.shape}",
        )

    return np.clip(array, -1.0, 1.0)


app = FastAPI(title="ColorFlow Inference API")
app.state.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
app.state.registered_model_name = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "colorflow-model")
app.state.registered_model_alias = os.environ.get("MLFLOW_REGISTERED_MODEL_ALIAS", "champion")
app.state.served_model_name = os.environ.get("SERVED_MODEL_NAME", "colorflow")
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

    l_channel = decode_l_tensor(model_name, payload)

    try:
        rgb_batch, version = server.predict(l_channel)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Inference failed: {error}") from error

    output = rgb_batch[0] if rgb_batch.shape[0] == 1 else rgb_batch

    return InferenceResponse(
        model_name=model_name,
        model_version=version,
        outputs=[
            ResponseOutput(
                name="rgb",
                shape=list(output.shape),
                datatype="FP32",
                data=output.reshape(-1).astype(float).tolist(),
            )
        ],
    )


@app.post("/v2/models/{model_name}/infer-image")
async def infer_image(model_name: str, image: UploadFile = File(...)) -> Response:
    if model_name != app.state.served_model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image was empty")

    l_channel = image_bytes_to_l_tensor(image_bytes, server.image_size)

    try:
        rgb_batch, version = server.predict(l_channel)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Inference failed: {error}") from error

    return Response(
        content=rgb_batch_to_png_bytes(rgb_batch),
        media_type="image/png",
        headers={
            "X-Model-Name": app.state.registered_model_name,
            "X-Model-Version": version,
            "X-Served-Model": model_name,
        },
    )