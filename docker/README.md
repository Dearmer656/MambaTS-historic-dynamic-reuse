# Docker runs (branch `docker`, one A6000)

Tasks: **3. PAT-320 task-embedding control (assigned 2026-11-11, run this one first)**; 2. PAT-315 Traffic L720 dev.

# PAT-315

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

## 3. Task: PAT-320 task-embedding control (assigned 2026-11-11, run first)

```bash
git pull origin docker
bash docker/run_p320_te.sh        # smoke (~15 min, must print "SMOKE OK"), then 3 seeds; commits + pushes after every run
```

Code: `openltm/` (minimal copy of OpenLTM: official Timer-XL + `pat320/`). One Timer-XL is trained jointly on ETTh1/2,
ETTm1/2, Weather, ECL, Traffic (channel independent, official splits, 672 -> 96 tokens, lr 5e-4 cosine, 10 epochs), with a
learned per-dataset vector (zero init) added to every input-token embedding. Then the error-carry heads `path` (d) and
`nopath` (c) are fitted on that frozen backbone (val only) and evaluated on every official test window, H 96/192/336/720.
This is the control the carry head must beat: does simply telling the model the dataset close the joint-vs-single gap?
Data: `docker/data/p320/*.csv.gz` + `docker/data/traffic.csv.gz`, unpacked into `openltm/dataset/` and checksummed.

Per seed: stage 1 ~1.5-2 h, each carry head ~1-2 h (rough, from the cluster 6000). GPU < 24 GB, CPU RAM < 32 GB.
Results + logs + the 3 backbone checkpoints (63 MB each): `openltm/pat_logs/docker_te/`, pushed to `origin docker`.
No mamba-ssm needed for this task (plain torch + numpy/pandas/scikit-learn/einops).
