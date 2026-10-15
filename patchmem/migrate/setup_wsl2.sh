#!/usr/bin/env bash
# PAT-304: one-shot env + code setup, run INSIDE WSL2 Ubuntu (not native Windows -- mamba-ssm/causal-conv1d
# have no official Windows wheel; see patchmem/migrate/README.md). On the Windows side, first run (in an
# elevated PowerShell, one time, then reboot if it asks):
#     wsl --install
# then open the "Ubuntu" app it installs and run this script there:
#     bash setup_wsl2.sh
# GPU passthrough uses your existing Windows NVIDIA driver automatically -- do not apt-get a driver inside
# WSL2. Only the CUDA *runtime* this script installs (via torch's own cu128 wheels) is needed in Ubuntu.
set -euo pipefail

# Must run from WSL2's own Linux filesystem, NOT /mnt/c/... (a Windows/NTFS path). git/chmod fail there
# ("Operation not permitted" on .git/config.lock) because NTFS-via-WSL doesn't support POSIX permission
# bits. cd to $HOME (ext4, real Linux fs) first; from Windows Explorer it's reachable at \\wsl$\Ubuntu\home\<you>.
case "$PWD" in /mnt/*) echo "Run this from \$HOME, not a /mnt/c/... Windows path (git chmod fails there). cd ~ && bash $0" >&2; exit 1 ;; esac

REPO_URL="https://github.com/Dearmer656/MambaTS-historic-dynamic-reuse.git"
BRANCH="dearmer656/pat-304-mambats-memory"
ENV_NAME="mamba_iso"
CLUSTER_HOST="REPLACE_ME"   # your SSH alias/hostname for the cluster login node (for the dataset scp below)

echo "== 1/5 miniconda =="
if ! command -v conda >/dev/null 2>&1; then
    curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "== 2/5 clone =="
if [ ! -d MambaTS-historic-dynamic-reuse ]; then
    git clone -b "$BRANCH" "$REPO_URL"
fi
cd MambaTS-historic-dynamic-reuse

echo "== 3/5 conda env (python 3.12, matches the cluster) =="
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

echo "== 4/5 torch (cu128) then the pinned libs (causal-conv1d/mamba-ssm need torch importable first) =="
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r patchmem/migrate/requirements-lock.txt

echo "== 5/5 verify: GPU visible, then the unit tests (no dataset needed yet) =="
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python patchmem/test_mem_retrieval.py
python patchmem/test_step_sg.py
python patchmem/test_from_scratch.py
echo "All unit tests passed -- environment matches the cluster."

echo
echo "Next (data, not automated -- fill in CLUSTER_HOST at the top of this script, or run by hand):"
echo "  mkdir -p dataset/ETT-small dataset/weather"
echo "  scp $CLUSTER_HOST:/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/dataset/ETT-small/ETTh2.csv dataset/ETT-small/"
echo "  scp $CLUSTER_HOST:/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/dataset/weather/weather.csv dataset/weather/"
