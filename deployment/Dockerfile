FROM python:3.9-slim

WORKDIR /opt/app

# Install MLflow + runtime deps
# Add any packages your pyfunc needs at inference time (redis/valkey client, etc.)
RUN pip install --no-cache-dir mlflow-skinny==3.0.1 pandas numpy psutil redis

# Copy model into the image
COPY mwj_model/competitive_pyfunc /opt/ml/model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "--host", "0.0.0.0", "--port", "8080", "--no-conda"]
