ARG BASE_IMAGE=colorflow-mlflow:local
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from fastai.vision.learner import create_body; import IPython; print('mlserver deps ok')"

COPY app.py ./
COPY colorflow ./colorflow

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]