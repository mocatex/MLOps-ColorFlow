ARG BASE_IMAGE=colorflow-mlflow:local
FROM ${BASE_IMAGE}

WORKDIR /app

COPY register_model.py /app/register_model.py

CMD ["python", "/app/register_model.py"]