# PAT-307 task 6 — TEFL compiled into ICL: version registry

One line per code version and per run batch. Every result quoted in Linear names the code commit below. Branch
`dearmer656/pat-304-mambats-memory` (cluster); laptop branch `laptop` (merges of it + `laptop/run_tefl_icl.sh`).

## Code versions

| commit | what changed | affects |
|---|---|---|
| `7f6c812` | R1TeflICL (`patchmem/r1_tefl_icl.py`): arms id / tied / qk / hyb, demos inside the lookback (origin j -> j+K), K = Hmax = 7; `R1_EVAL_FROM` test-only loading; icloff / iclperm audits; `FROZEN_MOD` (frozen A1 + ICL params) | all task 6 runs |
| (same) | exp_r1 training: `backward()` skipped when the loss has no gradient (frozen backbone + no demos at L < 2K+1) | frozen tied runs |
| `0f960dc` | K = H for every variant (`R1_ICL_KH=1`: error length = the forecast's H, zero-padded to 336) ; id with K = H is numerically identical to the earlier id-only K = H version (zero padding leaves dot products and norms unchanged) | idKH0, KH-trained arms, H192 array |
| `5be946e` | EFC-CET baseline (`patchmem/efc_cet.py`, arXiv 2610.22872 CED + CET + out-of-sample alpha, no TRL GRU) on frozen A1; cold / warm stream | efc_cet_* |
| (next) | `ridge` variant: closed-form ridge (lam 1) on the same in-window demos as id (solver ablation); `icl_kh.sbatch` takes the variant | iclridgeKH0 |

## Fixed choices (set before results)

- ICL step size: delta = eta * sum_j v_j <phi(k_j), phi(q)> / (|phi(q)| sum_j |phi(k_j)|); offline arms eta = 1 (not fitted).
- EFC-CET (paper leaves open): W shared across variables, NV = H validation origins, abar starts at 0, lam = 1, rho = 0.95, errors
  in data units; warm stream prefix 2880 steps.
- Judged on test MSE, 3/3 seeds; iclperm (values shuffled across the window's demos) is the pairing control.

## Run batches

| batch | where | code | logs | status |
|---|---|---|---|---|
| smoke (tied / hyb / qk / tiedFz, offline id0 / hyb0), ETTh2 s3047 | 3090 jobs 6006, 6028 | `2704130` + working tree = `7f6c812` | `slurm_6006_iclsmk.txt`, `slurm_6028_iclsmk.txt` | done |
| offline id, K = H, ETTh2 s3047 | 3090 job 6048 | `7f6c812` + id-only K = H (working tree) | `E_iclIdKH0_s3047.txt` | done |
| offline id, K = H, ETTh2 s3048 / s3049 | 3090 jobs 6083, 6084 | same as 6048 | `E_iclIdKH0_s{3048,3049}.txt` | done |
| offline id, K = H, Weather s3047-9 | 3090 jobs 6085-6087 | same as 6048 | `W_iclIdKH0_s*.txt` | running |
| KH-trained smoke (tefl / tied / hyb / tiedFz, random H) | 3090 job 6132 | `0f960dc` | `slurm_6132_khsmk.txt` | queued |
| ETTh2 H = 192 only, 6 arms x 3 seeds | 3090 array 6144 | `0f960dc` | `E_H192_{arm}_s*.txt` | queued |
| EFC-CET ETTh2 (smoke + 3 seeds) | 3090 job 6196 | `5be946e` | `efc_cet_E*.txt/json` | queued |
| laptop task 6: offline + ETTh2 H96 + Weather H96 / H192 | laptop | laptop `3aaef66` (code `0f960dc`) | `patchmem/logs/laptop/icl6_*` | waiting for the laptop |
