# Start the UI

```bash
# from the project root, build the image and run it with nginx.
docker build -t colorflow-ui -f services/ui/Dockerfile services/ui
docker run --rm -p 8081:80 colorflow-ui
# Then open http://127.0.0.1:8081

# for a quick local static preview without Docker
cd services/ui
python -m http.server 8081
```

The UI calls `/v2/...` on the same origin in Kubernetes. When you run the UI locally on port `8081`, it automatically falls back to `http://127.0.0.1:8080/v2/...`, so start MLServer on `127.0.0.1:8080` and keep that process running while the UI is open.