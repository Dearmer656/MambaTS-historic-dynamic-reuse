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
| `206ecfc` | `ridge` variant: closed-form ridge (lam 1) on the same in-window demos as id (solver ablation); `icl_kh.sbatch` takes the variant | iclridgeKH0 |
| `88e804a` | this registry | - |
| `3bf92a0` | **bug fix**: `r1_weather.sh` did not append `${EXTRA}`, so every Weather run with `EXTRA="--is_training 0"` TRAINED instead of test-only (ETTh2 script was fine). efc_cet: assert NaN-free stream input (Codex review; current slicing never hits NaN, no result change). Laptop branch merged at `e5eb743` | all Weather offline runs |
| `67a9a84` | 2x2 vs EFC-CET, feature (row E_j / EFC anti-diagonal D_o) x scope (window-local ICL / stream ridge): R1TeflICL `key='ced'` (`R1_ICL_KEY`, id / ridge only; key D_o[l] = E[o-l, lead l], value E[o], o = K..T-1-K); efc_cet row feature F_t = E_(t-H) (`--feats`), prints `[version] md5 + HEAD`. Codex: no real bug (3 points checked: value mask monotone in lead as in `demos()`, N < H never occurs, `--feats` runs go to their own dir) | iclidCedKH0, efc_row |

## Fixed choices (set before results)

- ICL step size: delta = eta * sum_j v_j <phi(k_j), phi(q)> / (|phi(q)| sum_j |phi(k_j)|); offline arms eta = 1 (not fitted).
- EFC-CET (paper leaves open): W shared across variables, NV = H validation origins, abar starts at 0, lam = 1, rho = 0.95, errors
  in data units; warm stream prefix 2880 steps.
- 2x2 (fixed before results): only the feature changes inside each scope; ICL cells id, eta 1, K = H; stream cells CET lam / NV / rho / alpha as above. Cells: row x local = ICL-0 (`E_iclIdKH0`), ced x local = `iclidCedKH0`, ced x stream = EFC-CET, row x stream = `efc_row`.
- Judged on test MSE, 3/3 seeds; iclperm (values shuffled across the window's demos) is the pairing control.

## Run batches

Cluster jobs run the repository as it is when the job STARTS (not when it was submitted); every log starts with `commit <hash>`,
which is the authoritative version. The "code" column is the latest commit the run needs; a later commit at start time only adds
unrelated code paths (checked when the batch is reported).

| batch | where | code | logs | status |
|---|---|---|---|---|
| smoke (tied / hyb / qk / tiedFz, offline id0 / hyb0), ETTh2 s3047 | 3090 jobs 6006, 6028 | `2704130` + working tree = `7f6c812` | `slurm_6006_iclsmk.txt`, `slurm_6028_iclsmk.txt` | done |
| offline id, K = H, ETTh2 s3047 | 3090 job 6048 | `7f6c812` + id-only K = H (working tree) | `E_iclIdKH0_s3047.txt` | done |
| offline id, K = H, ETTh2 s3048 / s3049 | 3090 jobs 6083, 6084 | same as 6048 | `E_iclIdKH0_s{3048,3049}.txt` | done |
| ~~offline id, K = H, Weather s3047-9~~ | 3090 jobs 6085-6087 | same as 6048 | `W_iclIdKH0_s*.txt` | **INVALID** (trained, see `3bf92a0`); cancelled 2026-10-25 on the user's order |
| offline id, K = H, Weather s3047-9 (rerun) | 3090 jobs 6619-6621 | `3bf92a0` | `W_iclidKH0_s*.txt` | queued |
| KH-trained smoke (tefl / tied / hyb / tiedFz, random H) | 3090 job 6132 | `88e804a` (log header) | `slurm_6132_khsmk.txt` | done, all 4 arms ran (1-epoch smoke, numbers not used) |
| ETTh2 H = 192 only, 6 arms x 3 seeds | 3090 array 6144 | `0f960dc` | `E_H192_{arm}_s*.txt` | queued |
| EFC-CET ETTh2 (smoke + 3 seeds) | 3090 job 6196 | `c819bb2` (log header; smoke and full run both started before `67a9a84`: full log has no `[version]` / `features` line) | `efc_cet_E*.txt/json` | queued |
| laptop task 6: offline + ETTh2 H96 + Weather H96 / H192 | laptop | laptop `e5eb743` (code `3bf92a0`); stage-1 W runs need >= `e5eb743` | `patchmem/logs/laptop/icl6_*` | waiting for the laptop |
| offline ridge, K = H, ETTh2 s3047-9 | 3090 jobs 6199-6201 | `206ecfc` | `E_iclridgeKH0_s*.txt` | s3047 done (6199), s3048-9 queued |
| ICL anti-diagonal key, K = H, ETTh2 / Weather s3047-9 | 3090 jobs 7305-7307 (E), 7308-7310 (W) | `67a9a84` | `{E,W}_iclidCedKH0_s*.txt` | queued |
| stream ridge, row feature, ETTh2 | 3090 job 7311 | `67a9a84` | `efc_row/`, `efc_cet_E_row*.txt` | queued |
