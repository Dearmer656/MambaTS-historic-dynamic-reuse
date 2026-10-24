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
| `38a0eb7` | `icl_kh.sbatch`: hyb loads the T0-MS checkpoint; `KH=0` keeps K = Hmax = 7 (TEFL's W1 input) | iclhybK7_0 |
| `a36fb02` | shrinkage (fixed before results, all reported): `R1_ICL_RLAM=s` ridge lam = s mean|k|^2; variant `hybridge` = ridge anchored at TEFL W2 in ReLU(W1) space (lam -> inf = TEFL); `R1_ICL_NSHRINK=c` eta n/(n+c); `icl_shrink.sbatch`. Codex: no bug (doc line fixed in the next commit) | iclridgeR*, iclhybN8, iclhybridgeR* |

## Fixed choices (set before results)

- ICL step size: delta = eta * sum_j v_j <phi(k_j), phi(q)> / (|phi(q)| sum_j |phi(k_j)|); offline arms eta = 1 (not fitted).
- EFC-CET (paper leaves open): W shared across variables, NV = H validation origins, abar starts at 0, lam = 1, rho = 0.95, errors
  in data units; warm stream prefix 2880 steps.
- 2x2 (fixed before results): only the feature changes inside each scope; ICL cells id, eta 1, K = H; stream cells CET lam / NV / rho / alpha as above. Cells: row x local = ICL-0 (`E_iclIdKH0`), ced x local = `iclidCedKH0`, ced x stream = EFC-CET, row x stream = `efc_row`.
- Shrinkage (2026-10-25): ridge lam scale s in {1, 10, 100, 1000} (A1, K = H); hyb eta n/(n+8) and hybridge s in {1, 10, 100, 1000} (TEFL T0-MS ckpt, K = 7); offline, eta 1; every value reported, none selected.
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
| offline id, K = H, Weather s3047-9 (rerun) | 3090 jobs 7362-7364 (6619-6621 failed: disk full) | `3bf92a0`+ | `W_iclidKH0_s*.txt` | queued |
| KH-trained smoke (tefl / tied / hyb / tiedFz, random H) | 3090 job 6132 | `88e804a` (log header) | `slurm_6132_khsmk.txt` | done, all 4 arms ran (1-epoch smoke, numbers not used) |
| ETTh2 H = 192 only, 6 arms x 3 seeds | 3090 array 6144 (a1 x3, tefl s3047-8, tied s3048 done); rerun array 7356 = tasks 5,6,8-17 (7356_5 / _6 hit disk quota at 0 s -> rerun 7414) | `0f960dc` (log headers 88e804a..9031b5c, no change to these arms) | `E_H192_{arm}_s*.txt` | partly done |
| EFC-CET ETTh2 (smoke + 3 seeds), DONE | 3090 job 6196 | `c819bb2` (log header; smoke and full run both started before `67a9a84`: full log has no `[version]` / `features` line) | `efc_cet_E*.txt/json` | queued |
| laptop task 6: offline + ETTh2 H96 + Weather H96 / H192 | laptop | laptop `e5eb743` (code `3bf92a0`); stage-1 W runs need >= `e5eb743` | `patchmem/logs/laptop/icl6_*` | waiting for the laptop |
| offline ridge, K = H, ETTh2 s3047-9 | 3090 jobs 6199-6201 | `206ecfc` | `E_iclridgeKH0_s*.txt` | DONE 3/3: s3047 6199, s3048-9 7357-7358 (6200 / 6201 lost to disk full; log headers 9031b5c / a2fc9e3, ridge path unchanged since 206ecfc). Result: +15..+95% vs A1, 3/3 seeds |
| ICL anti-diagonal key, K = H, ETTh2 / Weather s3047-9 | 3090 jobs 7365-7367 (E), 7368-7370 (W) (7305-7310 failed: disk full) | `67a9a84` | `{E,W}_iclidCedKH0_s*.txt` | queued |
| stream ridge, row feature, ETTh2 | 3090 job 7371 (7311 failed: disk full) | `67a9a84` | `efc_row/`, `efc_cet_E_row*.txt` | queued |
| EFC-CET Weather, ced + row | 3090 job 7372 | `67a9a84` | `efc_cet_W*.txt` | queued |

2026-10-25: /project/nlp-work5 full (quota exceeded): jobs started in that window died at 0 s or lost their logs; `patchmem/logs/plug_cache` (53G, plug_a0 caches, regenerable) deleted on the user's order.
| offline hyb0 (TEFL T0-MS ckpt + one ICL step on its residuals, eta 1, K = 7), ETTh2 s3047-9, 3x3 | 3090 jobs 7388-7390 | `38a0eb7` | `E_iclhybK7_0_s*.txt` | queued |
| shrinkage: ridge R1/10/100/1000 (A1), hybN8 + hybridge R1/10/100/1000 (TEFL), ETTh2 s3047-9, 3x3 | 3090 jobs 7460-7462 | `a36fb02` | `E_icl{ridgeR*KH0,hybN8K7_0,hybridgeR*K7_0}_s*.txt` | queued |
| laptop stage 2: ETTh2 H = 96 trained, a1 / tefl / tied / hyb / teflFz x 3 seeds, tiedFz s3049 (s3047-8 unfinished) | laptop pushes `19b7d44`, `956907f` | laptop `3c32134` (dirty: local completion-marker fix, ICL code = `0f960dc`) | `patchmem/logs/laptop/icl6_E_H96_*` (copied to the work branch) | done (tiedFz partial) |
| laptop stage 1: offline E id0 / idKH0 / tied0 s3047-8, hyb0 s3047 (K = 7, T0-MS ckpt) | same | same | `icl6_E_{id0,idKH0,tied0,hyb0}_s*` | partial; id / idKH / hyb0 duplicate cluster runs (idKH0 s3047 and hyb0 s3047 reproduce the cluster numbers exactly) |
| ~~laptop stage 1 W id0 s3047~~ | `956907f` | `3c32134` (before `3bf92a0`) | `icl6_W_id0_s3047.txt` | **INVALID** (r1_weather EXTRA bug: trained) |
| ~~laptop stages 5 / 6~~ -> moved to the server (user, 2026-10-25); all laptop task 6 runs stopped | - | - | - | cancelled |
| Weather H96 / H192 trained, a1 / tefl / hyb x 3 seeds (3 seeds in parallel on one elm73 6000 GPU, user's choice) | job 7481 | `c1e27ef` | `W_H{96,192}_{a1,tefl,hyb}_s*.txt` | running |

## Results so far (test MSE, ETTh2)

- H = 96 trained (laptop): hyb (TEFL + learned-eta ICL step) beats TEFL 3/3 seeds at every L: L1440 .2844 / .2868 / .2856 vs .2967 / .2968 / .2935
  (-4.1 / -3.4 / -2.7%), L720 -2.6 / -4.4 / -1.3%, L384 -3.9 / -0.7 / -0.1%. Inside hyb (L1440): ICL off +.0055 / +.0094 / +.0063, values shuffled
  +.0015 / -.0008 / +.0016 -> most of the ICL term survives shuffling (mean of the demo residuals), pairing ~0.5%. tied .359 / .342 / .350
  (beats A1 by 8-12%, loses to TEFL by 17-21%); teflFz +1.4 / +6.9 / +4.8% vs TEFL.
- H = 192 trained (cluster): TEFL .3718 / .3739 / .3689 vs A1 .4432 / .4231 / .4300 at L1440; tied s3048 .4114.
- Offline: ICL-0 (A1, K = H) L1440 H96 -7.4 / -7.3 / -7.3%; window-local ridge lam 1 +15..+95% (3/3); EFC-CET cold L1440 H96 -9.2 / -12.3 / -11.4%,
  warm -16.7 / -18.3 / -15.8%; hyb0 s3047 (TEFL ckpt, eta 1, K = 7) vs TEFL: L1440 -1.9 / -2.3 / -3.2% (H96 / 192 / 336), L720 (1 demo) +10..+13%.

