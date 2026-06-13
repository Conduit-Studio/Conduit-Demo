#!/usr/bin/env bash
# Build the BYO SageMaker training container and push it to ECR, then print the
# @sha256 DIGEST to paste (digest-pinned, not a tag!) into the Train Model node.
#
# A `:latest` tag is a lie the same way a bare S3 ref is — Conduit's lineage record
# pins the training image by digest so a registered model is actually reproducible.
#
#   AWS_REGION=us-east-1 bash scripts/build_and_push.sh
#
# Env knobs: AWS_REGION (default us-east-1), REPO (default conduit-yolo), TAG (default finetune).
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO="${REPO:-conduit-yolo}"
TAG="${TAG:-finetune}"
HERE="$(cd "$(dirname "$0")" && pwd)"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
URI="${REGISTRY}/${REPO}"

echo "→ ensuring ECR repo ${REPO} in ${REGION}…"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

echo "→ logging docker into ${REGISTRY}…"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "→ building + pushing ${URI}:${TAG}…"
docker build -t "${URI}:${TAG}" "${HERE}/../container"
docker push "${URI}:${TAG}"

DIGEST="$(aws ecr describe-images \
  --repository-name "$REPO" --image-ids imageTag="$TAG" --region "$REGION" \
  --query 'imageDetails[0].imageDigest' --output text)"

echo
echo "✅ Pin THIS in the Train Model node's image field (digest, not tag):"
echo "      ${URI}@${DIGEST}"
