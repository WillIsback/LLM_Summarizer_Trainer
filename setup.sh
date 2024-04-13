#!/bin/bash

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install the requirements
pip install -r requirements.txt

# Find CUDA version
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
# Install additional packages for PyTorch 2.2.1
PYTORCH_VERSION=$(python -c "import torch; print(torch.version.__version__)")
# Check GPU architecture
GPU_ARCH=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])")

# if PyTorch version is 2.1.0
if [[ $PYTORCH_VERSION == "2.1.0" ]]; then
    pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
    --index-url https://download.pytorch.org/whl/cu121
    # if GPU architecture is ampere of newer (RTX 30xx, RTX 40xx, A100, H100, L40)
    if [[ $GPU_ARCH -ge 8 ]]; then
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
        fi
    # else GPU architecture is older (V100, Tesla T4, RTX 20xx)
    else
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
        fi
    fi
# else if PyTorch version is 2.1.1
elif [[ $PYTORCH_VERSION == "2.1.1" ]]; then
    pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton \
    --index-url https://download.pytorch.org/whl/cu121
    # if GPU architecture is ampere of newer (RTX 30xx, RTX 40xx, A100, H100, L40)
    if [[ $GPU_ARCH -ge 8 ]]; then
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
        fi
    # else GPU architecture is older (V100, Tesla T4, RTX 20xx)
    else
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118-torch211] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121-torch211] @ git+https://github.com/unslothai/unsloth.git"
        fi
    fi
# else if PyTorch version is 2.2.0
elif [[ $PYTORCH_VERSION == "2.2.0" ]]; then
    pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
    --index-url https://download.pytorch.org/whl/cu121
    # if GPU architecture is ampere of newer (RTX 30xx, RTX 40xx, A100, H100, L40)
    if [[ $GPU_ARCH -ge 8 ]]; then
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
        fi
    # else GPU architecture is older (V100, Tesla T4, RTX 20xx)
    else
        # if cuda version is 11.8
        if [[ $CUDA_VERSION == "11.8" ]]; then
            pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
        # if cuda version is 12.1
        elif [[ $CUDA_VERSION == "12.1" ]]; then
            pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
        fi
    fi

# else if PyTorch version is 2.2.1
elif [[ $PYTORCH_VERSION == "2.2.1" ]]; then
    # if GPU architecture is ampere of newer (RTX 30xx, RTX 40xx, A100, H100, L40)
    if [[ $GPU_ARCH -ge 8 ]]; then
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
    # else GPU architecture is older (V100, Tesla T4, RTX 20xx)
    else
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps xformers trl peft accelerate bitsandbytes
    fi
fi