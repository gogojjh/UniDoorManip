#!/bin/bash
# Script to build the nvidia_docker:ubuntu2004_cuda117_torch1131 image

set -e

echo "Building Docker image: nvidia_docker:ubuntu2004_cuda117_torch1131"
echo "This may take several minutes..."

cd "$(dirname "$0")"

# Try to pull the base image first
echo "Pulling base CUDA image..."
docker pull nvidia/cuda:11.7.1-devel-ubuntu20.04 || {
    echo "Warning: Failed to pull nvidia/cuda:11.7.1-devel-ubuntu20.04"
    echo "Trying alternative tag: nvidia/cuda:11.7.0-devel-ubuntu20.04"
    docker pull nvidia/cuda:11.7.0-devel-ubuntu20.04 || {
        echo "Error: Could not pull CUDA base image. Please check your network connection."
        echo "You may need to configure Docker registry mirrors or check firewall settings."
        exit 1
    }
    # Update Dockerfile to use the alternative tag
    sed -i 's|nvidia/cuda:11.7.1-devel-ubuntu20.04|nvidia/cuda:11.7.0-devel-ubuntu20.04|g' Dockerfile
}

echo "Docker image built successfully!"
echo "Image: nvidia_docker:ubuntu2004_cuda117_torch1131"
