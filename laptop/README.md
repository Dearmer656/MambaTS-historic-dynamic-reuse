# Laptop runs (PAT-307, branch `laptop`)

Latest PAT-307 code (recurrent MambaTS R1 + error feedback) plus the scripts for the laptop tasks.
The datasets the tasks need (ETTh2, Weather) are committed on this branch, so nothing else needs to be downloaded.

## 1. One-time setup (Linux, RTX 4080)

```bash
git clone -b laptop git@github.com:Dearmer656/MambaTS-historic-dynamic-reuse.git && cd MambaTS-historic-dynamic-reuse
bash laptop/setup_env.sh          # conda env "mambats": Python 3.12, torch 2.8.0+cu128, mamba-ssm 2.2.5, causal-conv1d 1.5.2
conda activate mambats
bash laptop/check.sh              # unit tests + 100-iteration smoke run; must end with "CHECK OK"
```

`setup_env.sh` normally downloads prebuilt mamba-ssm / causal-conv1d wheels. If it starts compiling, it needs a CUDA 12.x `nvcc` on the PATH.

## 2. Tasks

| task | command | what | time (estimate) |
|---|---|---|---|
| 1 | `bash laptop/task1.sh` | ErrPath-shared, ETTh2, seed 3048 | 3-4 h |
| 2 | `bash laptop/task2.sh` | ErrPath-shared, Weather, seed 3048 | 4-6 h |

Run them one after the other (`bash laptop/task1.sh && bash laptop/task2.sh`); running both on one GPU at once only slows each down.
Any data/seed: `bash laptop/run_errpath_shared.sh E|W SEED`. Settings are identical to the cluster job `patchmem/slurm/vshare_job.sbatch` (seed 3047 runs on the cluster).

ErrPath-shared = O-ErrPath with the Householder path in the P-dim patch space, the same reflection for every variable (no cross-variable mixing; `R1_OBS_VSHARE=1`). See Linear PAT-307 / PAT-312.

## 3. Results

Logs: `patchmem/logs/laptop/{E,W}_OpathVs_s3048.txt`. The test MSE per (L, H) is on the `[R1Mem audit test]` lines (`normal mse`); the `noH` audit is the path removed.
Send them back with:

```bash
git add -f patchmem/logs/laptop/*.txt && git commit -m "laptop results" && git push origin laptop
```
