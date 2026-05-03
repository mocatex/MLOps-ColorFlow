FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py model-settings.json settings.json ./

EXPOSE 8080

CMD ["mlserver", "start", "."]