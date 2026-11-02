# Docker runs (PAT-315, branch `docker`, one A6000)

PAT-315 error carry (PaTH) on the frozen iTransformer, **Traffic, lookback L = 720, H = 192 / 336, 3 seeds**.
Everything the runs need is on this branch: code, `docker/data/traffic.csv.gz` (unpacked and checksummed by the
script), the 6 frozen backbone checkpoints (`checkpoints/iTransformer_base_T_L720_H{192,336}_s*`), their official test
MSE (`results/.../base_metrics.json`) and the FreshCast / B reference results (`docker/refs/`).

## 1. One-time setup

```bash
git clone -b docker git@github.com:Dearmer656/MambaTS-historic-dynamic-reuse.git && cd MambaTS-historic-dynamic-reuse
bash docker/setup_env.sh          # pip: torch 2.8.0+cu128, mamba-ssm 2.2.5, causal-conv1d 1.5.2, docker/requirements.txt
bash docker/run_traffic_dev.sh smoke   # ~10 min, must end with "SMOKE OK"
```

The repo imports MambaTS, so `mamba-ssm` / `causal-conv1d` must import even though only iTransformer runs.
`setup_env.sh` normally gets prebuilt wheels; if it starts compiling it needs CUDA 12.x `nvcc`.

## 2. Task: Traffic L720 dev (assigned 2026-11-02)

```bash
git pull origin docker
bash docker/run_traffic_dev.sh    # smoke, then 6 runs one after the other; commits + pushes after every run
```

`patchmem/eval_sr.py` v3 (`--base shrot --freeze_theta --bptt_h 168 --early_stop`, arms `s1_R,s1_D,s1_P,s1_P2`):
base = closed-form ridge on the shock history + a fixed rotary shock state; carry on its residual with state-driven
transition R (none) / D (scalar decay) / P (PaTH, 1 Householder) / P2 (2 Householders per step); theta frozen at the
weekly harmonics, truncated BPTT one week, epoch chosen on val. Configs are compared on **val only**.

Resources per run: GPU < 24 GB; **CPU RAM ~35 GB (H192) / ~55 GB (H336)** (error stream 862 channels x H on the CPU);
time roughly 4-6 h per run on an A6000 (24-36 h total). Interrupted? Run the same command again (finished runs are skipped).
Results + logs: `patchmem/logs/docker/sr3/T_H{192,336}_L720_s{seed}.{json,txt}`, pushed to `origin docker`.
If a push is rejected: `git pull --rebase origin docker && git push origin docker`.
