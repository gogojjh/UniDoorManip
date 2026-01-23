#!/bin/bash
# Installation script to run inside the unidoormanip Docker container

set -e

echo "=========================================="
echo "UniDoorManip Installation Script"
echo "=========================================="


apt update && apt install -y git vim locate wget tar curl
apt-get install -y libvulkan1 vulkan-tools
vulkaninfo | grep "GPU"

# Install Miniconda3
MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
wget -O /tmp/$MINICONDA_INSTALLER https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER
bash /tmp/$MINICONDA_INSTALLER -b -p /root/miniconda3
rm /tmp/$MINICONDA_INSTALLER

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Step 1: Create conda environment
echo ""
echo "Step 1: Creating conda environment 'unidoormanip' with Python 3.8"
echo "---------------------------------------------------"
conda create -n unidoormanip python=3.8 -y

# Step 2: Activate environment and install dependencies
echo ""
echo "Step 2: Installing dependencies"
echo "---------------------------------------------------"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unidoormanip

echo "Installing PyTorch and other packages..."
pip install torch==1.13.1 torchvision==0.14.1 ipdb trimesh

# Step 3: Install PointNet++
echo ""
echo "Step 3: Installing PointNet++"
echo "---------------------------------------------------"
cd /Titan/code/robohike_ws/src/UniDoorManip
if [ -d "Pointnet2_PyTorch" ]; then
    echo "Pointnet2_PyTorch directory already exists, skipping clone..."
else
    echo "Cloning PointNet++ repository..."
    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
fi

cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Note: You still need to install IsaacGym manually following the official guide:"
echo "  https://developer.nvidia.com/isaac-gym"
echo ""
echo "To activate the environment in future sessions:"
echo "  conda activate unidoormanip"

# Install IsaacGym
tar -xzf /Titan/code/robohike_ws/src/IsaacGym_Preview_4_Package.tar.gz -C /root/
cd /root/isaacgym/python
pip install -e .
cd examples

# Install Vulkan
mkdir -p /etc/vulkan/icd.d/
echo '{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3"
    }
}' > /etc/vulkan/icd.d/nvidia_icd.json
export vk_icd_filenames=/etc/vulkan/icd.d/nvidia_icd.json

# Test Vulkan
cd /root/isaacgym/python/examples
python joint_monkey.py

# Install claude code
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code
