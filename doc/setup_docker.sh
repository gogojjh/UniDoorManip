#!/bin/bash
# Complete setup script for UniDoorManip Docker environment

set -e

echo "=========================================="
echo "UniDoorManip Docker Setup Script"
echo "=========================================="

# Step 3: Create the container
docker run -it \
    --gpus all --env NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged --cap-add sys_ptrace \
    -e DISPLAY -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.docker.xauth \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev/input:/dev/input \
    -v /dev/bus/usb:/dev/bus/usb:rw \
    -v ~/.Xauthority:/root/.Xauthority:rw \
    -v ~/robohike_ws/src:/Titan/code/robohike_ws/src \
    -v /Titan/dataset/:/Titan/dataset \
    --network host \
    --name unidoormanip nvidia/cuda:11.7.1-devel-ubuntu20.04 /bin/bash

echo ""
echo "=========================================="
echo "Container setup complete!"
echo "=========================================="