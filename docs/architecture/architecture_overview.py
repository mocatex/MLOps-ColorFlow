# type: ignore

import os
import tempfile
from PIL import Image, UnidentifiedImageError
from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.vcs import Github
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.gcp.storage import GCS
from diagrams.programming.language import Python
from diagrams.k8s.storage import PV
from diagrams.programming.framework import React
from diagrams.custom import Custom
from urllib.request import urlretrieve


temp_dir = tempfile.TemporaryDirectory()


def get_icon(url, name):
    path = os.path.join(temp_dir.name, f"{name}.png")
    urlretrieve(url, path)

    # trim transparent margins
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            alpha_bbox = rgba.getchannel("A").getbbox()
            if alpha_bbox:
                rgba.crop(alpha_bbox).save(path)
    except (UnidentifiedImageError, OSError):
        # if not .png or cannot be processed, use the original
        pass

    return path


# --- CONFIGURATION ---
OUTPUT_DIR = "docs/img"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Logos for better branding
DAGSTER_URL = get_icon(
    "https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/png/dagster-light.png",
    "dagster",
)

# OXEN_URL = get_icon(
#     "https://raw.githubusercontent.com/Oxen-AI/Oxen/main/images/oxen-no-margin-black.png",
#     "oxen",
# )

DVC_URL = get_icon(
    "https://repository-images.githubusercontent.com/83878269/a5c64400-8fdd-11ea-9851-ec57bc168db5",
    "dvc",
)

GCS_URL = get_icon(
    "https://holori.com/wp-content/uploads/2024/07/gcp-logo-4.png", "gcs"
)

K8S_URL = get_icon(
    "https://raw.githubusercontent.com/cncf/artwork/master/projects/kubernetes/icon/color/kubernetes-icon-color.png",
    "kubernetes",
)

ML_SERVER_URL = get_icon(
    "https://raw.githubusercontent.com/mocatex/MLOps-ColorFlow/refs/heads/main/images/mlserver.png",
    "mlserver",
)

ML_FLOW_URL = get_icon(
    "https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/png/ml-flow-wordmark-dark.png",
    "mlflow",
)


# Graph attributes for spacing and font control
graph_attr = {
    "fontsize": "25",
    "nodesep": "0.4",  # Vertical distance between nodes (Higher = more space)
    "ranksep": "0.8",  # Horizontal distance between ranks (Higher = more space)
    "pad": "0.5",
    # "bgcolor": "#f0f0f0",  # for image
    "bgcolor": "transparent",  # for website/pitch
    "dpi": "300",
    "splines": "spline",
}

# Node attributes for text size
node_attr = {
    "fontsize": "20",
    "fontname": "Arial Bold",
}

# Edge attributes for styling
edge_attr = {
    "fontsize": "42",
    "fontname": "Arial",
    "penwidth": "5",
}

CLUSTER_FONT_SIZE = "22"

with Diagram(
    name="ColorFlow - GAN Image Colorizer Architecture",
    show=False,
    filename=f"{OUTPUT_DIR}/pipeline_architecture",
    outformat="png",
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):

    github = Github("\nGitHub\n(Code & Actions)")

    with Cluster(
        "Local Data Processing",
        graph_attr={
            "fontsize": CLUSTER_FONT_SIZE,
            "margin": "30",
            "bgcolor": "#bbdcf3",
        },
    ):
        local_code = Python("Preprocessing")
        dvc = Custom("Images (DVC)", DVC_URL)
        local_code >> dvc

    gcs = Custom("GCS\nData Lake", GCS_URL)

    # Big Kubernetes Cluster to host all production components
    with Cluster(
        "Production Kubernetes Cluster",
        graph_attr={
            "fontsize": CLUSTER_FONT_SIZE,
            "bgcolor": "#bbdcf3",
            "margin": "30",
        },
    ):
        k8s_master = Custom("Kubernetes", K8S_URL)
        dagster = Custom("\nDagster\nOrchestration", DAGSTER_URL)

        # MLFlow (Experiment Tracking)
        with Cluster(
            "Training & Tracking",
            graph_attr={"fontsize": CLUSTER_FONT_SIZE, "bgcolor": "#e4a8d1"},
        ):
            mlflow = Custom("MLflow\nExperiment Tracking", ML_FLOW_URL)

        checkpoints = PV("Shared Checkpoints")

        with Cluster(
            "Serving Layer",
            graph_attr={
                "fontsize": CLUSTER_FONT_SIZE,
                "bgcolor": "#ffe0af",
                "margin": "30",
            },
        ):
            backend = Custom("MLServer", ML_SERVER_URL)
            frontend = React("Website\n(User Interface)", group="frontend")
            (
                frontend
                >> Edge(label="Inference\n\n", fontsize="20", minlen="1.5")
                >> backend
            )

    user = Users("End Users")

    # --- POSITIONING & DISTANCES (Logic) ---

    # User Interaction
    # Invisible edge to keep User aligned with Backend:
    user >> Edge(style="invis") >> backend

    (
        user
        >> Edge(xlabel="  Uses  ", constraint="false", fontsize="20", minlen="2.0")
        >> frontend
    )

    # Trigger Flow
    (
        github
        >> Edge(
            style="dashed",
            color="darkblue",
            label="automatically trigger\n\n",
            fontsize="18",
        )
        >> dagster
    )

    # Data Flow
    dvc >> gcs >> Edge(tailport="e") >> mlflow
    dagster >> mlflow >> checkpoints >> Edge(tailport="e") >> backend
