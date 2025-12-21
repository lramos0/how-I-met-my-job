#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-east-2"
AWS_ACCOUNT_ID="833726713341"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/models/mwj"
TAG="${TAG:-v1}"
LOCAL_IMAGE="mwj-listings-api:amd64"

# Make sure repo exists (optional, safe)
aws ecr describe-repositories --repository-names models/mwj --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name models/mwj --region "${AWS_REGION}" >/dev/null

docker buildx create --use --name mwj-builder >/dev/null 2>&1 || true
docker buildx inspect --bootstrap >/dev/null

echo "Building linux/amd64 image: ${LOCAL_IMAGE}"
docker buildx build \
  --platform linux/amd64 \
  -t "${LOCAL_IMAGE}" \
  --load \
  .

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker tag "${LOCAL_IMAGE}" "${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo "Pushed: ${ECR_URI}:${TAG}"

