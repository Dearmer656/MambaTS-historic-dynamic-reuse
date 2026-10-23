#!/bin/bash
# One-time environment setup on the laptop (Linux, RTX 4080, NVIDIA driver supporting CUDA 12.8).
# Same versions as the cluster env: Python 3.12, torch 2.8.0+cu128, mamba-ssm 2.2.5, causal-conv1d 1.5.2.
# usage (repo root): bash laptop/setup_env.sh     then: conda activate mambats
set -e
ENV=${ENV:-mambats}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n $ENV python=3.12
conda activate $ENV
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install packaging ninja wheel setuptools
# both packages download a prebuilt wheel matching torch 2.8 / CUDA 12 / Python 3.12; if that fails they compile (needs nvcc 12.x)
pip install causal-conv1d==1.5.2 --no-build-isolation
pip install mamba-ssm==2.2.5 --no-build-isolation
pip install -r laptop/requirements.txt
python -c "import torch, mamba_ssm, causal_conv1d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
