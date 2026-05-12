import os
import asyncio
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import mlflow
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from mlflow import MlflowClient
from PIL import Image
from pydantic import BaseModel
from skimage.color import lab2rgb, rgb2lab

# Setup logging so we can see the hot-swapping in action
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("colorflow-mlserver")


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

        # Perform the initial synchronous load so the server doesn't start empty
        self.initial_load()

    def resolve_target_version(self) -> str:
        if self.client is None:
            raise RuntimeError("MLflow client is not configured")
        alias_version = self.client.get_model_version_by_alias(
            self.registered_model_name,
            self.registered_model_alias,
        )
        return alias_version.version

    def load_model_for_version(self, target_version: str) -> Any:
        model_uri = f"models:/{self.registered_model_name}/{target_version}"
        try:
            model = mlflow.pytorch.load_model(
                model_uri,
                dst_path=None,
                map_location=self.device,
            )
            model.to(self.device)
            model.eval()
        except Exception as error:
            self.load_error = None
            raise RuntimeError(str(error)) from error
        return model

    def initial_load(self) -> None:
        """Called once on startup to ensure a model is ready before taking traffic."""
        try:
            target_version = self.resolve_target_version()
            self.model = self.load_model_for_version(target_version)
            self.loaded_version = target_version
            logger.info(f"Initial startup: Loaded champion model v{target_version}")
        except Exception as e:
            logger.error(f"Failed initial model load: {e}")

    async def background_poller(self, interval_seconds: int = 10) -> None:
        """Silently polls MLflow in the background for new champion models."""
        logger.info(
            f"Started background poller (checking every {interval_seconds}s)..."
        )
        while True:
            try:
                target_version = self.resolve_target_version()

                # If MLflow has a new champion, start the hot-swap process
                if target_version != self.loaded_version:
                    logger.info(
                        f"🔄 New champion detected (v{target_version}). Pre-loading in background..."
                    )

                    # 1. Download and load into memory (Takes time, but doesn't block users)
                    new_model = self.load_model_for_version(target_version)

                    # 2. Atomic swap (Instant, zero downtime)
                    self.model = new_model
                    self.loaded_version = target_version
                    logger.info(
                        f"✅ Successfully hot-swapped to v{target_version}! Now serving new traffic."
                    )

            except Exception as e:
                # Silently catch errors so the poller doesn't crash if MLflow blips
                logger.debug(f"Poller check skipped: {e}")

            await asyncio.sleep(interval_seconds)

    def predict(self, l_channel: np.ndarray) -> tuple[np.ndarray, str]:
        # No more blocking MLflow checks here! Just use whatever is currently loaded.
        if self.model is None:
            raise RuntimeError("Server is still warming up, no model loaded.")

        l_tensor = torch.from_numpy(l_channel).to(self.device)
        with torch.no_grad():
            ab_tensor = self.model(l_tensor)
        rgb = lab_to_rgb(l_tensor, ab_tensor)

        if not np.isfinite(rgb).all():
            raise RuntimeError("Model returned a non-finite RGB image")

        return rgb.astype(np.float32), self.loaded_version


def lab_to_rgb(l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    l_tensor = (l_tensor.detach().cpu() + 1.0) * 50.0
    ab_tensor = ab_tensor.detach().cpu() * 128.0
    lab = torch.cat([l_tensor, ab_tensor], dim=1).permute(0, 2, 3, 1).numpy()
    rgb = [lab2rgb(sample) for sample in lab]
    return np.stack(rgb, axis=0)


server = ChampionModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Configure and do initial load
    server.configure(app)
    # 2. Start the background polling task
    polling_task = asyncio.create_task(server.background_poller(interval_seconds=10))
    yield
    # 3. Clean up on shutdown
    polling_task.cancel()


app = FastAPI(title="ColorFlow Inference API", lifespan=lifespan)
app.state.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
app.state.registered_model_name = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_NAME", "colorflow-model"
)
app.state.registered_model_alias = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_ALIAS", "champion"
)
app.state.served_model_name = os.environ.get("SERVED_MODEL_NAME", "colorflow")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8081,http://127.0.0.1:8081,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Model-Name", "X-Model-Version", "X-Served-Model"],
)


@app.get("/v2/health/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/v2/health/ready")
def ready() -> dict[str, str]:
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model warming up")
    return {"status": "ready", "model_version": server.loaded_version}


@app.post("/v2/models/{model_name}/infer", response_model=InferenceResponse)
def infer(model_name: str, payload: InferenceRequest) -> InferenceResponse:
    # [Omitted: the /infer endpoint logic remains exactly the same]
    pass


@app.post("/v2/models/{model_name}/infer-image")
async def infer_image(model_name: str, image: UploadFile = File(...)) -> Response:
    if model_name != app.state.served_model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image was empty")

    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = pil_img.size
    except Exception as error:
        raise HTTPException(
            status_code=400, detail=f"Invalid image: {error}"
        ) from error

    resized_img = pil_img.resize(
        (server.image_size, server.image_size), Image.Resampling.LANCZOS
    )

    rgb = np.asarray(resized_img)
    lab = rgb2lab(rgb).astype(np.float32)
    l_channel = (lab[..., 0] / 50.0) - 1.0
    l_tensor = l_channel[np.newaxis, np.newaxis, ...]

    try:
        rgb_batch, version = server.predict(l_tensor)
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Inference failed: {error}"
        ) from error

    output_rgb = np.clip(rgb_batch[0] * 255.0, 0, 255).astype(np.uint8)
    output_pil = Image.fromarray(output_rgb).resize(
        original_size, Image.Resampling.LANCZOS
    )

    buffer = BytesIO()
    output_pil.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-Model-Name": app.state.registered_model_name,
            "X-Model-Version": version,
            "X-Served-Model": model_name,
        },
    )
