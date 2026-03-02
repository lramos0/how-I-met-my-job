#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config
# -------------------------
export MLFLOW_TRACKING_URI="file:///Users/lramos/mlflow/mlruns"
MODEL_URI="runs:/730c65524cf548698260d8e5392d8af9/competitive_pyfunc"

AWS_REGION="us-east-2"
AWS_ACCOUNT_ID="833726713341"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/models/mwj"
TAG="v1"

LOCAL_IMAGE="mwj-competitive:${TAG}"

# Where we download the model artifacts
MODEL_DIR="mwj_model"

# -------------------------
# 1) Download model artifacts locally
# -------------------------
rm -rf "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

python - <<'EOF'
from pathlib import Path
import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

model_uri = "runs:/730c65524cf548698260d8e5392d8af9/competitive_pyfunc"
dst = Path("mwj_model")
dst.mkdir(parents=True, exist_ok=True)

local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst))
print("Downloaded model to:", local_path)
EOF

echo "Finding MLmodel..."
MLMODEL_PATH="$(find "${MODEL_DIR}" -name MLmodel -print -quit)"
if [[ -z "${MLMODEL_PATH}" ]]; then
  echo "ERROR: Could not find MLmodel under ${MODEL_DIR}"
  exit 1
fi
echo "MLmodel found at: ${MLMODEL_PATH}"

# The directory containing MLmodel is the model root
MODEL_ROOT="$(dirname "${MLMODEL_PATH}")"
echo "Model root: ${MODEL_ROOT}"

# -------------------------
# 2) Write a Dockerfile that serves the model root
# -------------------------
cat > Dockerfile <<EOF
FROM python:3.9-slim

WORKDIR /opt/app

# Install MLflow + runtime deps
# Add any packages your pyfunc needs at inference time (redis/valkey client, etc.)
RUN pip install --no-cache-dir mlflow-skinny==3.0.1 pandas numpy psutil redis

# Copy model into the image
COPY ${MODEL_ROOT} /opt/ml/model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "--host", "0.0.0.0", "--port", "8080", "--no-conda"]
EOF

# -------------------------
# 3) Build amd64 image locally (Apple Silicon -> linux/amd64)
# -------------------------
# Ensure buildx is set up
docker buildx create --use --name mwj-builder >/dev/null 2>&1 || true
docker buildx inspect --bootstrap >/dev/null

echo "Building linux/amd64 image: ${LOCAL_IMAGE}"
docker buildx build \
  --platform linux/amd64 \
  -t "${LOCAL_IMAGE}" \
  --load \
  .

# -------------------------
# 4) Login to ECR
# -------------------------
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# -------------------------
# 5) Tag + push
# -------------------------
docker tag "${LOCAL_IMAGE}" "${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo "✅ Pushed: ${ECR_URI}:${TAG}"

# -------------------------
# 6) (Optional) local architecture sanity check
# -------------------------
echo "Local image architecture:"
docker image inspect "${LOCAL_IMAGE}" --format='{{.Architecture}}'

