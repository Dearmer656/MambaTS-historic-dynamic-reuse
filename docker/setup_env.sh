#!/bin/bash
# One-time environment setup in the docker container (one A6000, NVIDIA driver supporting CUDA 12.8), plain pip.
# Same versions as the cluster env: torch 2.8.0+cu128, mamba-ssm 2.2.5, causal-conv1d 1.5.2 (Python 3.10-3.12).
# usage (repo root): bash docker/setup_env.sh      (set PY=python3.12 etc. to choose the interpreter)
set -e
PY=${PY:-python3}
$PY -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install packaging ninja wheel setuptools
# both packages download a prebuilt wheel matching torch 2.8 / CUDA 12; if that fails they compile (needs nvcc 12.x)
$PY -m pip install causal-conv1d==1.5.2 --no-build-isolation
$PY -m pip install mamba-ssm==2.2.5 --no-build-isolation
$PY -m pip install -r docker/requirements.txt
$PY -c "import torch, mamba_ssm, causal_conv1d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
