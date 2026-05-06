ARG BASE_IMAGE=colorflow-mlflow:local
FROM ${BASE_IMAGE}

WORKDIR /app

COPY register.py /app/register.py

CMD ["python", "/app/register.py"]