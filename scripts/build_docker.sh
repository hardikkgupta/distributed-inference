#!/bin/bash

# Exit on error
set -e

# Configuration
IMAGE_NAME="llm-inference"
IMAGE_TAG="latest"
REGISTRY="your-registry.com"  # Change this to your container registry

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag the image for the registry
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Push the image to the registry
echo "Pushing image to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo "Build and push completed successfully!" 