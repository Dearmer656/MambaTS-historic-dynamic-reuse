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
| `dde84fb` | audit on the training split: `R1_AUDIT_FLAGS=train,val,test`, train every `R1_AUDIT_TRAIN_STRIDE` (8)-th window; `icl_kh.sbatch` `LOGTAG` | iclidKH0_tvt |
| `66b4835` / `98a5bee` | lookback-limited stream `win_ced` / `win_row` (`efc_cet.py --win_only`, `efc_cet.sbatch D ced,row win`): at origin t only origins i in [t-L+H, t-H] (every target used lies in the lookback [t-L, t), ICL-0's information range); window sums add / subtract; alpha out of sample on the last NV = H in-window origins, NO EMA; alpha = 1 also reported; from the warm start. Fixed before results. Tests: brute-force recomputation, targets outside the window have no effect. Codex: no leakage / off-by-one; test `__main__` order fixed (98a5bee). Codex on dde84fb: no bug | efc_win |
| `d31f7d3` / `6ed28c0` | Traffic-32 (32 sensors, `make_traffic32.py`): `resid_cov.CFG['T']`, `efc_cet --data T`, `icl_kh.sbatch SEED T`, `traffic_hyb.sbatch` (copy of weather_hyb, 3090 array, H96 a1 / tefl / hyb); **`r1_traffic32.sh` now appends `${EXTRA}`** (it lacked it like the Weather bug, no offline T run existed before); offline hyb refused off ETTh2 (Codex) | T_* |
| `d83fd13` / `214641d` | `win_check.py` (main = win_row, user 2026-10-25): controls in the same window with the same out-of-sample alpha: bias (window mean error), persist (W = I), scalar (rho I), shuffle (W from (F_pi(i), E_i)), ridge_mean (per-variable centering + ridge); bases a1 / tefl (E R1TfmsR64h0s all H; W / T R1*TfmsH96s, H96 only) / a0 (ExpA0 direct per (L, H), E / W). Tests: stream_ctrl('ridge') == stream_win, every control window-only. Codex: no leak / slicing bug; A0 args now include the recorded run args | win_ck |

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
| offline id, K = H, Weather s3047-9 (rerun) | 3090 jobs 7362-7364 (6619-6621 failed: disk full) | `3bf92a0`+ | `W_iclidKH0_s*.txt` | DONE 3/3 (s3047 from 7522 tvt, same code path) |
| KH-trained smoke (tefl / tied / hyb / tiedFz, random H) | 3090 job 6132 | `88e804a` (log header) | `slurm_6132_khsmk.txt` | done, all 4 arms ran (1-epoch smoke, numbers not used) |
| ETTh2 H = 192 only, 6 arms x 3 seeds | 3090 array 6144 (a1 x3, tefl s3047-8, tied s3048 done); rerun array 7356 = tasks 5,6,8-17 (7356_5 / _6 hit disk quota at 0 s -> rerun 7414) | `0f960dc` (log headers 88e804a..9031b5c, no change to these arms) | `E_H192_{arm}_s*.txt` | DONE 6 arms x 3 seeds (7414 log header b5533ae) |
| EFC-CET ETTh2 (smoke + 3 seeds), DONE | 3090 job 6196 | `c819bb2` (log header; smoke and full run both started before `67a9a84`: full log has no `[version]` / `features` line) | `efc_cet_E*.txt/json` | queued |
| laptop task 6: offline + ETTh2 H96 + Weather H96 / H192 | laptop | laptop `e5eb743` (code `3bf92a0`); stage-1 W runs need >= `e5eb743` | `patchmem/logs/laptop/icl6_*` | waiting for the laptop |
| offline ridge, K = H, ETTh2 s3047-9 | 3090 jobs 6199-6201 | `206ecfc` | `E_iclridgeKH0_s*.txt` | DONE 3/3: s3047 6199, s3048-9 7357-7358 (6200 / 6201 lost to disk full; log headers 9031b5c / a2fc9e3, ridge path unchanged since 206ecfc). Result: +15..+95% vs A1, 3/3 seeds |
| ICL anti-diagonal key, K = H, ETTh2 / Weather s3047-9 | 3090 jobs 7365-7367 (E), 7368-7370 (W) (7305-7310 failed: disk full) | `67a9a84` | `{E,W}_iclidCedKH0_s*.txt` | DONE 3/3 |
| stream ridge, row feature, ETTh2 | 3090 job 7371 (7311 failed: disk full) | `67a9a84` | `efc_row/`, `efc_cet_E_row*.txt` | DONE 3/3 (log HEAD b5533ae, md5 cee9b26777) |
| EFC-CET Weather, ced + row | 3090 job 7372 | `67a9a84` | `efc_cet_W*.txt` | DONE 3/3 (log HEAD b5533ae, md5 cee9b26777) |

2026-10-25: /project/nlp-work5 full (quota exceeded): jobs started in that window died at 0 s or lost their logs; `patchmem/logs/plug_cache` (53G, plug_a0 caches, regenerable) deleted on the user's order.
| offline hyb0 (TEFL T0-MS ckpt + one ICL step on its residuals, eta 1, K = 7), ETTh2 s3047-9, 3x3 | 3090 jobs 7388-7390 | `38a0eb7` | `E_iclhybK7_0_s*.txt` | DONE 3/3 |
| shrinkage: ridge R1/10/100/1000 (A1), hybN8 + hybridge R1/10/100/1000 (TEFL), ETTh2 s3047-9, 3x3 | 3090 jobs 7460-7462 | `a36fb02` | `E_icl{ridgeR*KH0,hybN8K7_0,hybridgeR*K7_0}_s*.txt` | DONE 3/3 |
| laptop stage 2: ETTh2 H = 96 trained, a1 / tefl / tied / hyb / teflFz x 3 seeds, tiedFz s3049 (s3047-8 unfinished) | laptop pushes `19b7d44`, `956907f` | laptop `3c32134` (dirty: local completion-marker fix, ICL code = `0f960dc`) | `patchmem/logs/laptop/icl6_E_H96_*` (copied to the work branch) | done (tiedFz partial) |
| laptop stage 1: offline E id0 / idKH0 / tied0 s3047-8, hyb0 s3047 (K = 7, T0-MS ckpt) | same | same | `icl6_E_{id0,idKH0,tied0,hyb0}_s*` | partial; id / idKH / hyb0 duplicate cluster runs (idKH0 s3047 and hyb0 s3047 reproduce the cluster numbers exactly) |
| ~~laptop stage 1 W id0 s3047~~ | `956907f` | `3c32134` (before `3bf92a0`) | `icl6_W_id0_s3047.txt` | **INVALID** (r1_weather EXTRA bug: trained) |
| ~~laptop stages 5 / 6~~ -> moved to the server (user, 2026-10-25); all laptop task 6 runs stopped | - | - | - | cancelled |
| Weather H96 / H192 trained, a1 / tefl / hyb x 3 seeds (3 seeds in parallel on one elm73 6000 GPU, user's choice) | job 7481 | `c1e27ef` | `W_H{96,192}_{a1,tefl,hyb}_s*.txt` | running |
| ICL-0 (id, K = H) on A1, audits on train / val / test, grid 720/1440 x 96/192, 1 perm rep; Weather 7522-7524, ETTh2 7525-7527 | 3090 | `dde84fb` | `{W,E}_iclidKH0_tvt_s*.txt` | DONE 3/3 (log header b5533ae) |

## Results so far (test MSE, ETTh2)

- H = 96 trained (laptop): hyb (TEFL + learned-eta ICL step) beats TEFL 3/3 seeds at every L: L1440 .2844 / .2868 / .2856 vs .2967 / .2968 / .2935
  (-4.1 / -3.4 / -2.7%), L720 -2.6 / -4.4 / -1.3%, L384 -3.9 / -0.7 / -0.1%. Inside hyb (L1440): ICL off +.0055 / +.0094 / +.0063, values shuffled
  +.0015 / -.0008 / +.0016 -> most of the ICL term survives shuffling (mean of the demo residuals), pairing ~0.5%. tied .359 / .342 / .350
  (beats A1 by 8-12%, loses to TEFL by 17-21%); teflFz +1.4 / +6.9 / +4.8% vs TEFL.
- H = 192 trained (cluster): TEFL .3718 / .3739 / .3689 vs A1 .4432 / .4231 / .4300 at L1440; tied s3048 .4114.
- Offline: ICL-0 (A1, K = H) L1440 H96 -7.4 / -7.3 / -7.3%; window-local ridge lam 1 +15..+95% (3/3); EFC-CET cold L1440 H96 -9.2 / -12.3 / -11.4%,
  warm -16.7 / -18.3 / -15.8%; hyb0 s3047 (TEFL ckpt, eta 1, K = 7) vs TEFL: L1440 -1.9 / -2.3 / -3.2% (H96 / 192 / 336), L720 (1 demo) +10..+13%.

| lookback-limited stream ced / row, ETTh2 / Weather, 3 seeds | 3090 jobs 7728 (E), 7729 (W) | `98a5bee` | `efc_win/`, `efc_cet_{E,W}_win*.txt` | queued |
| Traffic-32 offline ICL-0 row / ced (A1 `R1Th0s*`), 3 seeds | 3090 jobs 7734-7739 | `6ed28c0` | `T_iclid{,Ced}KH0_s*.txt` | queued |
| Traffic-32 EFC stream ced / row (7740) + lookback-limited win (7741) | 3090 | `6ed28c0` | `efc_cet_T*.txt`, `efc_win/` | queued |
| Traffic-32 trained H96 a1 / tefl / hyb x 3 seeds | 3090 array 7742 | `6ed28c0` | `T_H96_{a1,tefl,hyb}_s*.txt` | queued |
| win_row vs controls: E a1 / tefl / a0, W a1 / tefl / a0, T a1 (T tefl after 7742; no Traffic A0 ckpt) | 3090 jobs 7896-7902 | `214641d` | `win_ck_{D}_{base}.txt`, `win_ck/` | queued |

**Firm-up criterion (fixed before results):** win_row beats every control on all 3 datasets, 3/3 seeds, and keeps a gain on A0 and TEFL; cells that fail are reported, none dropped.

## Results 2026-10-25 evening (test MSE unless noted; 3 seeds s3047 / 3048 / 3049; % vs the base forecast)

- train / val / test (ICL-0 on A1, K = H): the gain is the same size on the TRAINING split, so it is not an OOD correction.
  ETTh2 L1440/96 train -23.7 / -22.2 / -22.3%, val -23.4 / -25.9 / -22.1%, test -7.4 / -7.3 / -7.3%; Weather L1440/96 train -14.6 / -13.3 / -14.2%,
  val -17.3 / -15.4 / -16.8%, test -22.6 / -20.7 / -22.9%. Shuffle control ~0 to +3% on every split.
- 2x2 (L1440/96, cold stream): ETTh2 row x local -7.4 / -7.3 / -7.3%, ced x local -9.4 / -8.8 / -9.1%, ced x stream -9.2 / -12.3 / -11.4%,
  row x stream -19.0 / -18.9 / -18.2%; Weather row x local -22.6 / -20.7 / -22.9%, ced x local -23.4 / -21.2 / -23.6%, ced x stream
  -31.4 / -28.3 / -31.9%, row x stream -36.1 / -32.3 / -36.0%. Scope (stream) matters more than the feature; the row feature beats EFC's CED in
  the stream at every L / H, 3/3 seeds, both datasets. ced x local at L720/336: +19..+35% (1 demo).
- shrinkage (L1440/96): ridge s = 1 -15.7 / -13.7 / -16.3%, s = 10 -12.4 / -11.1 / -12.2%, s = 100 -3.0%, s = 1000 -0.4% (vs A1); s = 1 hurts
  at L720/192 (+15..+22%), s = 10 is within +-1.1% there. hybridge s = 10 vs TEFL -1.6 / -3.4 / -2.4% and ~0 at L720 (hyb0 eta 1: +10%);
  hybN8 -1.6 / -2.9 / -1.9%. All values reported, none selected.
- H192 trained, L1440: A1 .4432 / .4231 / .4300, TEFL .3718 / .3739 / .3646, hyb .3606 / .3712 / .3700 (-3.0 / -0.7 / +1.5% vs TEFL: not 3/3),
  tied .4049 / .4114 / .3953, tiedFz .4032 / .4167 / .4217, teflFz .3954 / .4100 / .3934.
- Weather H96 trained (job 7481, partial): L1440 A1 .3049 / .2705 / .3064, TEFL .1952 / .1939 / .1948; hyb still training.

### win_check results (2026-10-25; win_check.py md5 2fd5fbd6b9, efc_cet.py md5 c2cc5e45a3, HEAD 518da0b; jobs 7896-7900, 7902; 7901 W a0 L1440 still 2/3 seeds)
Firm-up criterion **NOT met**. Base sanity OK (E tefl L1440/96 .3010/.3058/.3179, E a1 .3991/.4087/.4054 match logged test MSE).
- a1 (R1): gain 3/3 in every active cell on E/W/T; beats all 5 controls 3/3 on W and T everywhere; E fails at L720/192, L1440/192 (ridge_mean ~1-2pt better) and L1440/336 (bias better).
- tefl: W H96 L720/1440 -4~-5% 3/3, beats all controls; E ~0 (L1440/96 +0.5~+1.3%), only L1440/336 -1~-2.5%.
- a0 (direct MambaTS): no gain (E/W +0~+3%), only E L1440/336 -1.4~-2.1%.
- T a1 base ~.88-.97, win_row -17~-43%; no a0, tefl pending 7742.
- Inactive cells (window too short: L-2H+1 < NV): L384/H192, L384/H336, L720/H336.
Summary script: patchmem/logs/server/win_ck/summ.py.

### win_attn batch: in-context attention on the error window (designed 2026-10-25 with the user, fixed before results)
`win_check.py --arms attn` (`sbatch patchmem/slurm/win_check.sbatch D BASE attn`, logs win_attn_{D}_{BASE}.txt, json win_ck/win_attn_*.json).
Same window [t-L+H, t-H], same out-of-sample alpha (clip [0,1], NV = H, no EMA) as win_row; keys L2-normalised, context = the same variable's window pairs.
- lin1: mean_j cos(q,k_j) e_j (ICL-0 analogue) | krlin: per-variable ridge on unit keys (lam 1)
- exp1: mean_j exp((cos-1)/s2) e_j (1 GD step in Gaussian-kernel space) | soft1: softmax(cos/s2) attention | krexp: Gaussian kernel ridge (lam 1)
- s2 = median pairwise squared distance of unit keys (median heuristic; <= 256 evenly spaced keys, user choice 1); keys row and ced (user choice 2); L2 normalisation (user choice 3).
- Criterion: kernel helps iff exp1 < lin1 and krexp < krlin; softmax normalisation helps iff soft1 < exp1; 3/3 seeds; watch A0 especially.
- Tests 10 pass (explicit dual / loop per arm, window-only). Codex: no leak / transpose bug; fixes: float64 before exp, added exp1 to separate kernel from normalisation.

### win_attn partial (2026-10-25 22:15; 8507 elm73 running: E 21/27 lines, T a1 13, W 3-6; 8566 T tefl DONE; 8565 win_ck T tefl DONE)
Summary script patchmem/logs/server/win_ck/summ_attn.py. So far:
- soft1 (softmax 1 step) never beats exp1 except some T a1 cells; 1-step kernels (exp1/soft1) ~ useless.
- krexp < krlin: T a1 all L>=720 cells (n1-2 seeds), T tefl L720/L1440 H96 3/3 (L1440/96 krexp_row -6.2/-6.2/-5.9% vs win_row -2.7/-2.6/-2.6%; abs .547 < trained hyb .554); E / W / a0 mostly no.
- a0 (E, W partial): every attention arm 0 or worse.
- per-variable linear ridge (krlin_row) < pooled win_row on E / W (sharing across variables helps); on T similar.
- win_ck T tefl: win_row -0.6 / -1.6 / -2.7% (L384/720/1440), beats all controls 3/3.

### Traffic-32 A0 training (2026-10-25, user "提")
`sbatch patchmem/slurm/a0_job.sbatch T 96 SEED L` for L in 384/720/1440, seeds 3047-3049 (9 jobs, 3090). No official 32-channel args, so the traffic32 R1/TEFL
architecture is used (dm32 el3 df512 nh16 dropout .2 bs8 lr 5e-4 patch 48; = Weather H96 A0 args); one direct model per L (like E / W A0).
Eval args: CFG['T']['a0']. Codex: ckpt name matches win_check.a0_errors glob; eval args reproduce training. Logs A0_T_L{L}_H96_s{s}.txt.
Then: win_check T a0 (ctrl and attn).
- Repacked 2026-10-25 (user: two tasks per 3090): 8647 (L384 s3047) runs alone; pending 8648-8655 cancelled -> a0_pair.sbatch 8658 (384: s3048,s3049), 8659 (720: s3047,s3048), 8660 (720 s3049, 1440 s3047), 8661 (1440: s3048,s3049). Same args / logs.

### gain_src batch: where does the gain come from (designed 2026-10-25, user "分析任务也请跑上 证明我们的收益来自于哪里"; fixed before results)
`patchmem/gain_src.py` (tests patchmem/test_gain_src.py; 13 pass with win_check / efc tests), packed runs via patchmem/slurm/gain_src_pack.sbatch.
Cells L 720/1440, H 96, 3 seeds; bases E a1/tefl/a0, W a1/tefl/a0, T a1/tefl (T a0 after its training).
Diagnostics: per-lead gain, per-day gain (spread vs episodes), per-variable gain, in-sample oracle removable share (row / ced / bias; upper
bound, cheating by design), oracle W structure (fresh 10% feature rows / diagonal / top-1 singular share), soft1 and krexp weight mass on
day / week lags vs uniform.
Unit-test finding: an exactly 6-periodic error gives soft1 day ratio 1.54 (max 6.1, effective neighbours 89%), krexp 3.65: the
median-heuristic softmax is near uniform, kernel ridge sharpens selection.
Codex: fixed min context (H + 1 as stream_ctrl), zero guards; day buckets not calendar-aligned (documented); env-order remark not a bug (exp_r1 reads env at model build).
- 2026-10-25 23:00: elm72 (6000) fails in mamba triton layer_norm ('invalid argument to exchangeDevice'): A0 jobs 8663/8664 and gain_src 8672/8673 all failed before computing. Resubmitted: A0 8675/8676 (a100-80, 4 per card), gain_src 8677/8678 (a6000, 4 per card); packing sbatches now --exclude=elm72. First Traffic A0: L384 s3047 test MSE .4476 (3090 job 8647).

### win_res batch: EFC + in-context attention on its residuals (designed 2026-10-25 with the user, fixed before results)
`win_check.py --arms res` (packed via patchmem/slurm/win_pack.sbatch res "D:B ..."; logs win_res_{D}_{B}.txt, json win_ck/win_res_*.json).
- Stage 1 (global error map, user 1: both): win_row (lookback-limited) | efc_ced (EFC full stream, paper feature) | efc_row (full stream, row), run from first - L.
- Stage 2 (user 3: attention setting): softmax attention over [t-L+H, t-H], keys = unit row features of E, values = stage-1 residuals,
  weights softmax(cos / (s2 / 4)) (user 2: 1/4 of the median heuristic; att1 = unscaled reference), no slope term; step beta out of
  sample clip [0,1] (NV = H). lam_l of the weighted-ridge form does not enter the attention form (user 4 moot).
- Criterion: g+att < g, 3/3 seeds, per cell; watch tefl and a0. Sanity: efc_row stage 1 must equal efc_cet "row warm" on the same base.
- Tests 14 pass (vals = E identity, scale, window-only for E and residuals). Codex: R causal, first0 coverage OK, scale 1 / vals None unchanged;
  its "query leak" remark is a misread (row_feat F[t] = E[t-H], targets y[t-H..t-1]; the window-only test perturbs E[t] and passes).

### 2026-10-25 23:20 partial results (summ_gsrc.py; gain_src E all / W a1 a0 / W tefl 2 seeds / T 1 seed)
- Traffic A0 (H96) test MSE: L384 .4476/.4479/.4484, L720 .4397/.4397/.4390 (L1440 training): far below TEFL (L720 .569) and TEFL+krexp.
- Gain source (L1440/96, win_row): A1 gain is broad (E 84-87% / W 90-92% of days improved, all variables) and front-loaded on E / W
  (lead Q1 -40 / -55% vs Q4 -12 / -25%); W oracle map near rank 1 on the freshest errors (fresh10 .68-.76, top1 .73-.80) = carry of the
  latest observed error; Traffic not front-loaded (Q1 -35, Q4 -40), map high rank (top1 .14) + periodic retrieval (krexp day x3.8, week x9.8).
- Oracle removable share (row, in-sample): A1 E .29 / W .38-.41 / T .31; TEFL E .04-.07 / W .09 / T .04; A0 E .06-.08 / W .02. Bias ~0.
- soft1 weights ~uniform (day x1.02-1.17); krexp day x1.3-2.0 (E / W), x2.5-3.8 (T).

### hyb main-method batch (2026-10-26 00:40; user: keep hyb as the main method, "做吧")
- Existing evidence (no new runs): ETTh2 H96 trained hyb (laptop stage 2, `patchmem/logs/laptop/icl6_E_H96_*`): TEFL .2967/.2968/.2935 -> hyb
  .2844/.2868/.2856 (-4.1/-3.4/-2.7%, 3/3). Inference audits in the training logs (L1440/96): icloff W +5.9..6.9%, T +10.8..11.4%,
  E +1.9..3.3%; iclperm W +5.3..5.9%, T +1.0%, E ~0 -> pairing matters on W only; on T / E the ICL term acts mostly as a residual mean.
- New training-time controls in `r1_tefl_icl.py` (fixed before results): hybmean (query-free weights |phi(k_j)| / sum, i.e. a recent TEFL-residual
  mean, same scale) and hybperm (residuals permuted among valid demos at every forward, train + eval; its iclperm audit is uninformative).
  Tests: test_tefl_icl.py 13 + others = 27 pass. Codex: no bug (iclperm on hybperm uninformative, documented).
- Runs (patchmem/slurm/train_pack.sbatch, 3 per card): E/W/T H96 hybmean / hybperm x 3 seeds; Traffic H192 a1 / tefl / hyb x 3 seeds.
- Criterion: hyb < hybmean and hyb < hybperm (3/3) => the gain needs query-dependent in-context pairing; otherwise report it as a
  residual-mean correction.

### 2026-10-26 results: hyb controls (W/T), H192, gain vs #in-window examples
- Training-time controls H96 L1440: W hyb .1831/.1829/.1840 vs hybmean .1960/.1942/.1948 and hybperm —/.1939/.1943 (s3047 running): W gain needs the paired map (criterion likely passes). T hybmean .5507/.5484/.5505 < hyb .5558/.5535/.5549 (3/3), hybperm .5563/.5572/.5562: T gain = recent-residual mean (criterion fails). E fails (hybperm 2/3).
- H192 L1440 hyb vs TEFL: E −3.0/−0.7/+1.5%, W −0.25/−0.41/−0.82%; T TEFL still training (hyb .5219/.5211/.5196).
- Gain vs #examples (n = L/48 − 2H/48; same trained model tested at L384/720/1440), H96 3-seed mean: E −1.6/−2.8/−3.4%, W −2.7/−5.6/−5.8%, T −1.2/−4.7/−4.5% (n = 4/11/26). H192 W: ≈0 / −0.6 / −0.5% (n = 0/7/22): the H192 drop is not explained by n; hypothesis n/K (13 vs 5.5), untested. E at n = 0 (hyb ≡ TEFL structurally) still varies ±3%: ETTh2 run-to-run noise.
- Diagnostic in flight: job 8991 (R1_ICL_WDUMP, commit 38da1f4), hyb demo weights train vs test, W/T x 3 seeds.
- Job 8991 done (R1_ICL_WDUMP, ckpt-eval test MSE matches training logs: T s3047 .555772, W .183081). `patchmem/icl_wdump_summ.py`:
  W (patch = 8 h): top lags 3, 6, 9 patches (= 1, 2, 3 days) on train AND test, ~2.2x the other lags; effective demos .71-.73 (train) vs .74-.75 (test);
  recent-3 mass .138-.139 -> .155-.157 (uniform .115); total weight .36-.37 both; JS(train, test) .0021-.0022 vs test-half noise .0015-.0020.
  T (patch = 2 days): near uniform (eff .96-.97), top lags 7, 14, 21 (= 2, 4, 6 weeks) at ~1.3x; JS(train, test) .0001 = noise.
  => hyb does NOT move its weight to different time points on test: same lag profile (daily on W, weak 2-week on T), only a slight
  shift to the newest errors on W. Its test-time adaptation comes from the window's own residual values, not from re-selecting lags.

### PAT-313 results (job 9041 on elm73, commit fb41113; offline `patchmem/icl_case.py`, summary patchmem/logs/server/icl_case/icl_case_summary.json)
Trained hyb, H96 L1440 test, 3 seeds; "kept" = share of the ICL gain kept (1 - dMSE(mode) / dMSE(icloff)); normal MSE matches the logs.
1 constant Delta (test mean): W -0.03..+0.03, T +0.04..+0.06 -> the gain is entirely case-specific (case-specific energy W .89-.92, T .96-.97).
2 context swap, kept vs d: W 10 min .99 | 6 h .53-.58 | 1 day .79 | 1 week .45-.47 | 30 days .21-.26;
  T 1 h .69-.71 | 6 h -.18..-.22 (worse than no ICL) | 1 day 1.00-1.01 | 1 week .87-.88 | 30 days .67-.69. Phase-aligned context
  (same time of day) works, phase-shifted context fails; usefulness decays over weeks.
3/4 W: energy const .49-.54, linear .13-.16, daily .23-.25, rest .08-.12; in-model kept const .69-.70, +linear .81, +daily 1.03-1.05;
  common over variables kept .10-.16. Clusters: level offsets growing over the leads (~60%), mixed linear/daily shapes (~35%).
  T: const .03, linear .01-.02, daily .46-.47, rest .48-.50; kept const .03-.05, +linear .05-.06, +daily .56-.58, common .03-.06;
  every cluster is a per-variable daily-cycle error profile (daily energy .70-.83) at different phases.
5 R2 of the top-3 shape scores on time of day + day of week: W <= .014, T .006-.087 (not a clock function). Alignment
  cos(Delta, -err_icloff): W mean .31-.33 (71-73% > 0), T .37 (89% > 0).
Pre-stated expectations: 1 W < 50% yes, "more on T" no; 2 W falls with d mostly but 1 day > 6 h (phase), T not "most at small d"
(6 h negative); 4 W daily part yes, "T const/common" no (T is daily per variable); 5 yes.

### PAT-313 point 6 results (job 9075, commit 8dc33d2): select-then-adapt, inference only on the trained hyb, H96 L1440 test
Gain kept vs hyb's own ICL term (>1 = better than hyb), and absolute test MSE:
- W: iclsel 1.22/1.23/1.38 (MSE .1804/.1805/.1797, −1.5/−1.3/−2.3% vs hyb); iclsel1 (one GD step per set) 1.24/1.47/1.51 (.1801/.1779/.1781,
  −1.6/−2.8/−3.2%); selfix_all 1.14/1.11/1.17; topk8 1.01–1.06; phase .28–.65; recent4 −1.75..−3.32 (overfits).
- T: iclsel 1.52/1.51/1.53 (.5247/.5211/.5225, −5.6/−5.8/−5.8%); iclsel1 1.68/1.65/1.66 (.5148/.5127/.5143, −7.4/−7.4/−7.3%); phase alone 1.41–1.42;
  selfix_all 1.01–1.09; topk8 .91–.96; recent4 .23–.31.
- Mean set weights near uniform (W all .25 recent4 .23 phase .26–.27 topk8 .25; T phase .28–.31, recent4 .21): the soft
  combination is close to an equal mix on average; per-window variation not yet measured.
- Criterion (kept > 1 on W, 3/3): PASSED (also on T). Open: is it per-case selection or the ensemble of restricted fits? Needs a
  fixed-uniform combination control (not pre-registered).
- Follow-up (job 9086, commit f9ba398, `patchmem/icl_sel_summ.py`): validated selection vs uniform 1/4 mix.
  W: sel1 vs avg1 +0.21/−0.26/−0.20%, sel vs avg −0.16/−0.15/−0.11% -> criterion (both 3/3) NOT met: W gain ≈ ensemble of restricted fits.
  T: sel1 vs avg1 −0.84/−0.83/−0.88%, sel vs avg −0.37% x3 -> selection adds ~0.85% on top of the ensemble (avg1 alone −6.6% vs hyb).
  Weights: mean max W .35, T .32 (uniform .25); KL W .085–.094, T .027–.031. Gain over uniform by KL quartile: T rises monotonically
  3/3 (+.0003 -> +.010); W mixed (top quartile −.0013/+.0015/+.0012). argmax w = best fixed set: W .28–.29, T .44 (chance .25).
  Oracle per-(window, variable) pick among the 4 one-step sets: W .148, T .459 (vs sel1 .178–.180 / .513–.515; optimistic: min of 4).
- Plan 1 Hedge (offline, `patchmem/icl_hedge.py`; (beta, gamma) chosen on test half 1, reported on half 2; all arms under the same
  linear mix approximation, gap vs real errors 1.2-3.2% in norm). Half 2 MSE vs uniform mix:
  W: hedge −0.14/−0.20/−0.20% (chose beta 1, gamma .9: near uniform), hedge x valid −0.44/−0.78/−0.68%, iclsel1 +0.31/−0.13/−0.09%,
     per-variable hindsight −0.9..−1.1%.
  T: hedge −0.96/−0.89/−1.04% (beta 10, gamma .997; phase weight .40-.41, recent4 .15), hedge x valid −0.8..−0.9%, iclsel1 −0.41..−0.45%,
     per-variable fixed hindsight +0.4..+0.7% (worse than hedge: the best set changes over time).
  Criterion (hedge beats avg1 and sel1 on half 2, 3/3, W and T): met, but the W margin is tiny; in-model confirmation pending.
- Plan 1 in-model check (job 9523, commit f5d3b54, `patchmem/icl_selw_summ.py`; real errors, causal Hedge weights fed per window):
  half 2 vs uniform mix: W hedge −0.13/−0.18/−0.21%, hedge x valid −0.44/−0.79/−0.71%, iclsel1 +0.30/−0.13/−0.07%;
  T hedge −1.03/−0.98/−1.16%, hedge x valid −0.86..−1.04%, iclsel1 −0.40..−0.43%. Matches the offline approximation. Criterion met.
  All test windows (half 1 used to pick beta, gamma): W hx .1792/.1770/.1772 vs hyb .1831/.1829/.1840; T hedge .5101/.5079/.5080 vs hyb
  .5558/.5535/.5549 (TEFL .5828/.5816/.5782).
- Traffic H192 done (jobs 8793-8795): L1440 A1 .8577/.8578/.8533, TEFL .5182/.5202/.5213, hyb .5219/.5211/.5196 (+0.70/+0.17/−0.33%).
  hyb vs TEFL by L (3-seed mean): L384 (0 examples) +0.77%, L720 (7) +0.30%, L1440 (22) +0.18%: no gain at H192 on T either.

### PAT-313 H192 select-then-adapt (jobs 9588 elm73 + 9592 elm43, commit d2504bd), MSE change vs hyb, L1440 test
- W: iclsel1 −0.91/−0.98/−0.79%, avg1 −0.87/−0.94/−0.74%, fixed phase (2/3/4 days) −1.10/−1.19/−1.01%, topk8 −0.98/−1.02/−0.85%.
- T: iclsel1 −1.52/−1.64/−1.35%, avg1 −1.47/−1.58/−1.31%, fixed phase (2/4 weeks) −3.12/−3.36/−2.89% (vs TEFL ≈ −2.4/−3.2/−3.2%), topk8 ≈ −0.9%.
- E: iclsel1 +0.40/+0.43/+0.40%, avg1 ≈ +0.4%, topk8 −0.63/−0.70/−1.49%, phase +1.6..+2.2%, recent4 +2.5..+2.7%.
- Converged ridge (iclsel / iclselavg) overfits at H192: W +8.3..+9.6%, E +17..+20%, T +1..+1.8% (22 demos, 192-dim rows; lam = mean |phi|^2 too weak).
- Set weights ~uniform (.23–.27). Criterion (iclsel1 or avg1 beats hyb 3/3): W yes, T yes, E no (E only topk8, not the pre-registered arm).

### PAT-313 two-layer test-time update (job 9615 elm66, commit 5269d9c, `patchmem/icl_2l_summ.py`), H192 hyb s3047, L1440 test
One normalised GD step on W1 (a1 / sum |k_j|^2, ReLU mask) then hyb's W2 step on the updated features; a1 picked on half 1.
- Half-2 MSE vs hyb: E icl2l +0.09% (a1 .1), icl2lw1 +1.10%; W icl2l +0.01% (a1 .1), icl2lw1 +0.93%; T icl2l −0.07% (a1 .3), icl2lw1 +0.52%.
- All windows: larger a1 diverges (a1 10: E 8.26, W .855, T .611). Optimal a1 -> 0: the per-window W1 step adds nothing at H192
  (22 demos vs a 64 x 192 W1); the hyb gain is the W2 step. Single seed, test-time only (no training of eta1).
- Layer-2 NLMS normalisation Z2 = sum |h'_j|^2 (job 9633 elm73, commit 55ff4fc), test time on the hyb (trained with the old norm),
  H192 s3047, a2 (x learned eta) and a1 chosen on half 1, half-2 MSE vs hyb: E a1 0 a2 1 +0.37%; W a1 0 a2 3 −0.86%; T a1 0 a2 3 −0.40%
  (a1 0.1 no better). a2 = 3 is the grid edge on W / T: the old eta is too small for Z2 there -> train the variant hybz.
- Z2 test-time check, all 3 seeds (jobs 9633, 9648, 9649 elm73), a1 = 0, a2 chosen on half 1, half-2 MSE vs hyb:
  W −0.86/−0.82/−0.40% (a2 = 3 x3, grid edge); T −0.40/−0.52/−0.30% (a2 = 3 x3); E +0.37/+0.66/+3.00% (a2 1/1/3). Stable 3/3 on W and T,
  3/3 worse on E. a1 = 0.1 never better. Next: train hybz (eta learned under Z2).
- Two-layer test-time update, 3 seeds (jobs 9615, 9673, 9674), H192, half-2 MSE vs W2 only (hyb), a1 chosen on half 1:
  W1+W2: E +0.09/+0.27/−0.15%, W +0.01/−0.04/+0.00%, T −0.07/−0.06/−0.04%; W1 only: E +1.10/+1.57/+8.16%, W +0.93/+0.94/+0.88%,
  T +0.52/+0.63/+0.45%; none (TEFL part): E +1.42/+1.66/+2.00%, W +1.02/+1.01/+0.93%, T +0.82/+0.94/+0.62%. W1+W2 ≈ W2 only.

### PAT-313 staircase diagnostic (job 9683, commit 85c905c, `patchmem/stair_diag.py`; TEFL ckpts s3047, L1440 test)
Removable share of the current error row, in-sample | fit on test half 1 -> scored on half 2:
- H192 E: row .070|−.086, stair .163|−.286, row+stair .242|−.414 (nothing transfers on ETTh2: the test period shifts)
- H192 W: row .043|+.018, stair .083|+.016, row+stair .108|+.011;  H192 T: row .049|+.014, stair .073|+.015, row+stair .101|+.017
- H96 W: row .098|+.076, row+stair .118|+.076;  H96 T: row .043|+.015, row+stair .082|+.035
The fresh staircase doubles the in-sample share at H192, but a fixed global map on it does not transfer across time (except T H96):
the extra information is period-specific -> use it inside the per-window in-context step, not as a trained global branch.
- Two-layer test-time update at H96 (job 9694 elm73; 9687 had landed on elm72 and failed), W / T x 3 seeds, a1 chosen on half 1:
  full-test means W none .1949, W1 .1886, W2 .1833, W1+W2 .1810; T none .6162, W1 .5875, W2 .5547, W1+W2 .5511.
  Half 2 vs W2 only: W1+W2 W −1.19/−1.17/−1.38%, T −0.54/−0.65/−0.61% (3/3); W1 only W +2.84/+2.38/+2.06%, T +4.10/+4.71/+3.99%.
  => At H96 both layers help (W1+W2 best 3/3); at H192 the W1 step adds nothing. Slides: method = W1+W2, layer ablation H96 + H192.

### PAT-314 periodicity vs ICL controls (job 10538, `patchmem/eval_itransformer_tefl_phase_controls.py`, uncommitted)
Fixed in advance: iTransformer-TEFL paper ckpts (TEFL_paper_L96_E192), Traffic N2880 P168 (14 demos, ages 336..2520), seeds 3047/2024/2025.
Same demos and TEFL residuals r_j for every arm; only the combination differs: icl_w2 (method, W2-only step), icl_w2_shufK
(similarity weights deranged among same-phase demos, 3 perms), phase_mean, phase_normw (icl_w2 with cos=1), phase_persist
(latest demo), ridge / ridge_icpt (kernel ridge on h=ReLU(W1 k), anchored at TEFL). Scales/lambda chosen on val, test once.
Decision rule: ICL mapping supported only if icl_w2 < phase_mean, phase_normw and shufK on test 3/3 seeds.
Result (job 10538, elm43, 14 min). Test MSE s3047/2024/2025 (TEFL .4201/.4208/.4201):
  icl_w2 .3961/.3959/.3964 | phase_mean .3959/.3956/.3962 | phase_normw .3975/.3971/.3978 | shufK .4006/.4005/.4009
  phase_persist .4157/.4158/.4156 | ridge .4062/.4065/.4065 | ridge_icpt (lam100 a.5) .3944/.3943/.3947
  Paired per-case win rate of icl_w2: vs phase_mean .488/.489/.481, vs ridge_icpt .394/.406/.393.
=> Decision rule FAILS: phase_mean <= icl_w2 3/3. The Traffic gain is a periodic error template (mean of 14 weekly residuals);
   similarity weighting adds nothing over uniform; a key-dependent part on top of the mean (ridge_icpt) gives only ~0.35% more, 3/3.
Codex review: no result-changing bug (icl_w2 picked a=1.0 = reference; normw differs only when h_q=0; n=1 derangement impossible).
Known diag bug: test neff_frac blows up where sum of scores is 0 (dead h_q); val value .71 is valid.
ETT/Weather controls (jobs 10568-10571, val-selected cells, 3 seeds s2024/2025/3047), test gain vs TEFL %:
  E N1440 P24 (41 demos): icl_w2 1.37/1.38/1.30 | phase_mean 1.03/0.98/0.95 | normw .80/.79/.76 | shufK .44/.43/.33 | ridge_icpt 1.54/1.43/1.29
  M1 N2880 P672 (3 demos): icl_w2 1.57/1.59/1.62 | phase_mean 1.60/1.66/1.65 | shufK 1.49/1.11/1.54 | ridge_icpt -.35/-.15/-.12
  M2 N2880 P96 (26 demos): icl_w2 1.25/1.37/1.39 | phase_mean 2.27/2.25/2.26 | normw .90/1.06/1.01 | ridge_icpt 2.30/2.40/2.42
  W N2880 P144 (17 demos): icl_w2 .20/.18/.05 | phase_mean 1.00/1.10/1.12 | shufK .40/.42/.48 | ridge_icpt 1.29/1.56/1.49
=> Decision rule passes only on ETTh2 (icl_w2 > phase_mean by ~0.35pp, > shufK by ~0.95pp, 3/3). Fails on M1, M2, W, T.
   Uniform mean of demo residuals beats icl_w2 clearly on M2 (+1pp) and W (+0.9pp). Mean + ridge key part is best on E/M2/W/T.
Standalone-backbone controls (--backbone base, srun job 10615 elm82, same cells, 3 seeds; W1 random, W2 zero so r_j = backbone error).
3-seed mean test MSE: base | base+mean | base+icl_w2 || TEFL | TEFL+mean | TEFL+icl_w2
  E  .3794 | .3763 | .3714 || .3724 | .3687 | .3672
  M1 .3798 | .3676 | .3683 || .3584 | .3525 | .3527
  M2 .2535 | .2426 | .2445 || .2353 | .2300 | .2321
  W  .2262 | .2224 | .2219 || .2108 | .2085 | .2105
  T  .4123 | .3961 | .3938 || .4203 | .3959 | .3961
=> base+mean beats TEFL only on Traffic (TEFL joint training hurts f there). On ETT/Weather TEFL keeps a clear lead over any
   training-free correction of the standalone backbone. On base, icl_w2 > mean on E (3/3, ~1.3pp) and T (3/3, ~0.6pp).

### PAT-314 average + centered softmax ICL, day vs week (srun job 10694 elm82, commit ca91629, `eval_itransformer_tefl_avg_icl.py`)
Fixed in advance: n in {4,8,16} demos per phase (day/week, same n on every dataset), a2=1, tau_dot=1 (/sqrt64), tau_cos=0.1, avg scale a=1;
avg_valA (a on val) as reference only. Test gain vs TEFL % (3-seed mean):
  avg a=1 is harmful almost everywhere (e.g. E day_n4 -70.9, T day_n16 -6.7); only T week_n8/n16 +1.5/+3.0, M2 week_n16 +0.1.
  softmax ICL on top (a2=1) is worse than avg a=1 in every cell (dot -8..-76, cos -19..-94).
  avg_valA: E day .33 (n16) / week 1.09 (n8); M1 day .94 / week 2.39 (n8); M2 day .45 / week 1.50 (n16);
            W day .87 / week .56 (n16); T day .13 / week 6.11 (n8). a_val identical across seeds in every cell.
=> Unshrunk corrections (a=1, a2=1) add mostly noise: step size must be shrunk. Weekly phase >= daily on E, M1, M2, T.
Codex review: no bug for default n (std NaN only when n=1).

### PAT-314 two-layer case-wise GD on TEFL head: bias -> W2 -> W1 (srun job 10727 elm82, commit 95dc8a4, `eval_itransformer_tefl_twolayer.py`)
Per case: b = a_b*mean r_j; W2 step on v_j = r_j - b (eta = a2/(|h_q| sum|h_j|)); W1 NLMS step on post-W2 residual (/sum_j |k_j|^2, x a1).
Steps chosen per family on val (grids a_b 7 x a2 5 x a1 5); test scored once. Same day/week n in {4,8,16} cells as above.
Best-cell test gain vs TEFL % (3-seed mean; all 3 seeds same direction):
  E  week_n8 : avg 1.09 | avg+W2 1.38 | avg+W2+W1 1.38   (avg+W2 > avg also in week_n4 .35->.69, day_n16 .33->.55; week_n16 .88->.72)
  M1 week_n8 : avg 2.39 | avg+W2 2.53 | avg+W2+W1 2.68   (avg+W2 > avg in all 6 cells; W1 adds 0..+0.15, s3047 week_n4 3.29 outlier)
  M2 week_n16: avg 1.50 | avg+W2 1.64 | avg+W2+W1 1.64   (week_n8 avg+W2 < avg; W1 diverges on week_n4/n8)
  W  day_n16 : avg .86  | avg+W2 .90  | W1 diverges      (week cells avg+W2 < avg)
  T  week_n8 : avg 6.11 | avg+W2 5.74 | W1 diverges      (W2 harmful in week_n4/n8; +.04 at week_n16)
W1 step diverges on test (1e3..1e36 x MSE) on T, W, M2 week_n4/n8 even at a1=0.03 chosen on val. Suspected cause (not verified):
per-channel /sum|k_j|^2 makes the q shift ~|q|/|k|; channels with near-zero demo errors blow up.
=> Gain remains mostly the same-phase residual average. W2 adds a stable +0.1..0.3pp on E/M1/M2 best cells, hurts T, ~0 on W.
   Note: M2 best here (1.64, n<=16) is below phase_mean with 26 day demos (2.26) above.
Codex review: no bugs.

### PAT-314 step-size rules / joint ICL / longer horizon (base backbone unless noted)
Step size (commit d5a58b9, `eval_itransformer_stepsize.py`, H192): cos W2 best on most cells; nlms much worse on TEFL;
  const W2 only wins on base ETTh2 (2.19..3.15% vs mean .80..86). W2->W1 order without clip diverges (eta^2 ~ 1/|h_q|^2).
Joint one step from random W1/W2 (commit e9a9434, g - g0): joint ~= W2-only; W1-only .03..35%. W1 step adds nothing.
H336/720 (commit ae15826, same P and demo count as H192, pure ICL clip_w1 0.1), test gain vs base %, 3 seeds:
  H336: E icl_w2 1.78..2.59 vs mean .66..88; M2 3.73..4.96 vs 3.08..4.36; W 1.17..1.37 vs .43..84; T pure W1W2 4.35..4.47 vs mean 3.10..3.26; M1 mean best (3 demos)
  H720: E all ~.1..2; M2 icl 2.42..2.99 vs mean .83..1.47; W icl .45..61 vs mean <0; M1 ridge_icpt best
  H720 T: phase_mean 1.51/1.65/1.58 | icl_w2 1.88/2.00/2.02 | shufK 1.20/1.31/1.31 | ridge 2.99/3.08/3.08 | pure W2-only 3.37/3.44/3.41 | pure W1+W2 2.05/2.17/2.14
=> Phase mean degrades with H; similarity-weighted ICL > mean and > shuffled K grows with H (except M1).

### PAT-315 error-path attention, stage 1 (commit 6c9124c, `eval_error_path.py`, jobs 10909-10911)
Traffic H192, base backbone, 14 same-phase demos (ages 336..2520), trained on train split, early stop on val, full test (2.86M cases).
Test gain vs base % (s3047/s2024/s2025, mean):
  A  uniform          4.38/4.52/4.47  4.46
  B  lag MLP          4.90/4.99/4.96  4.95
  Bp B + q.k          5.37/5.50/5.42  5.43   <- best
  End Bp + y endpoints 4.93/5.15/5.22 5.10
  Cmean diag path     5.23/5.27/5.27  5.26
  C  beta=2 (Householder P) 5.25/5.35/5.30 5.30
  C  learn beta       5.26/5.29/5.37  5.31   (beta q50 .80; |Pk|/|k| .62 -> .48 with age)
  Crev P^T            5.26/5.36/5.30  5.31
  Cerr A_r from errors 5.22/5.30/5.28 5.27
Decision rule (C > Bp, End, Cmean 3/3): FAILS. Every path variant < Bp 3/3 (-0.05..-0.22pp). Path > End 3/3 and ~= Cmean (+.02..+.10).
Order diag (200k test cases, per-demo random factor order): model vs permuted differs by <= .03pp (C two), <= .01pp (learn).
Entropy/log n q50 ~.96 (near uniform) for all score variants.
=> Observed y path between demo and query adds no error-predictive info beyond lag + error similarity on Traffic H192;
   order of Householder factors is irrelevant ("no path info", not just "no non-commutative benefit").

### PAT-315 prefix-state similarity (commit e7e6147, jobs 11169-11171)
Each side's own past day: h_{r,m} = A_{r-1}...A_{r-24} h0_m (4 learned probes, beta=2, u from y window via MLP).
Spre = Bp + sum_m h_t^T h_s / sqrt(d); Sperm = same with the 24 factors in a random order per batch (query and demos shared).
Same cell as stage 1 (Traffic H192, 14 same-phase demos). Test gain vs base % (s3047/s2024/s2025, mean):
  Bp    5.37/5.50/5.42  5.43
  Spre  5.49/5.69/5.55  5.58   > Bp 3/3 (+.12/+.19/+.13)
  Sperm 5.49/5.49/5.58  5.52   > Bp 2/3; Spre > Sperm 2/3 (+.00/+.20/-.03)
Order diag on trained Spre (200k test cases, random per-batch order at eval): 5.29->5.21, 5.68->5.52, 5.63->5.52 (3/3 drop .08-.16pp).
State cos q50 ~.8 (Spre) / .67-.80 (Sperm). Spre s2024 hit the 10-epoch cap.
=> Decision rule (Spre > Bp and > Sperm 3/3): FAILS on Sperm. First variant over Bp 3/3: each side's accumulated past day
   adds ~+0.15pp beyond lag + error similarity. Trained Spre relies on order, but an order-free state trained as such reaches ~the same.

### PAT-315 out-of-sample training (--train_split val, commit d8d5262, jobs 11188-11190)
Module trained on first 75% of val origins (1173), early stop on last 25% (200, after H gap); test unchanged.
77.7% of training demo values still lie in the train period (only targets strictly out-of-sample).
Test gain vs base % (s3047/s2024/s2025, mean) | train-split result:
  A      3.37/3.53/3.52  3.47 | 4.46
  Bp     4.37/4.52/4.33  4.41 | 5.43
  Spre   4.53/4.79/4.47  4.60 | 5.58
  Sperm  4.49/4.68/4.50  4.56 | 5.52
All variants ~1pp worse than training on train (0/3), including A (only the output scale is learned):
learned scale ~.72-.76 (val) vs ~.53-.54 (train). The val period favours a larger correction than test does.
Ranking unchanged: Bp > A 3/3 (+.8..1.0), Spre > Bp 3/3 (+.14/+.27/+.17), Spre > Sperm 2/3 (+.04/+.11/-.04).
Order diag on Spre: 4.28->4.21, 4.80->4.60, 4.46->4.48.
=> In-sample targets do not suppress the gain; training on train is better (more data, scale closer to test).

### PAT-315 Smean order-free control (commit 0459d62)
Smean = Bp + sum_m h_t^T h_s / sqrt(d), h_m = g_m * l2norm(mean over the same 24 steps of MLP(y window)) (m=4, d=16).
Test gain vs base % (s3047/s2024/s2025, mean): Bp 5.43 | Spre 5.58 | Sperm 5.52 | Smean 5.67/5.74/5.71 5.70
Smean > Bp 3/3 (+.30/+.24/+.29); Spre > Smean 0/3 (-.18/-.05/-.15); Sperm > Smean 0/3. State cos q50 ~.96.
=> The past-day gain comes from a plain commutative summary of the recent y windows; Householder ordered accumulation
   is worse than the mean. No evidence for an ordered/path structure on Traffic H192.

### PAT-315 training-free raw similarity (commit 6734b60, `eval_raw_sim.py`, jobs 11313-11315)
e_hat = s * sum_j softmax(tau_e sim(q, k_j) + tau_y sim(y past 24 steps at t, at s_j)) e_j, no learned map;
tau in {0,.5,...,64}, s in .05..2.0 chosen on full val, test once. Test gain vs base % (s2024/s2025/s3047):
  mean (tau=0)  3.04/3.19/3.15 (s=.8)
  cos e        3.62/3.77/3.72 (tau_e=2)
  cos y        3.68/3.83/3.79 (tau_y=2)
  cos e+y      3.96/4.11/4.06 (tau_e=1, tau_y=2)  > mean 3/3 by ~+.9pp
  l2 e / y / e+y  -.05..15 / 1.57..1.71 / 1.66..1.75 (val picked the smallest tau=.5: grid too coarse for L2, not conclusive)
Val picks s=.8 for every family, while the train-split modules learn s~.53 (A: 4.46 with the same uniform weights),
so absolute numbers across protocols are confounded by the scale; compare within protocol:
  training-free: cos e+y - mean = +0.9pp;  learned (train split): Smean - A = +1.25pp, Bp - A = +1.0pp.
=> Raw cosine on errors and on past-day y each helps, and they add; learned maps add a further ~0.3pp over raw.

### PAT-315 raw similarity, matched scale protocol (commit after 571c20f, jobs 11324-11329)
tau grid adds .03/.06/.125/.25. Main: tau and s selected on 1M fixed train cases (as the learned modules fit s on train).
Test gain vs base % (s3047/s2024/s2025, mean):
  raw, select on train : mean 4.37/4.50/4.46 4.45 (s=.55) | cos e 5.06 | cos y 5.15 | cos e+y 5.34/5.46/5.42 5.41 (tau 2,2; s .55)
                         l2 e 1.56 | l2 y 1.79 | l2 e+y 1.40 (train picks tau .5-1; train errors are smaller -> L2 tau mis-scaled)
  raw, fixed s=.7      : mean 3.91 (= PAT-314 phase_mean exactly) | cos e 4.52 | cos y 4.55 | cos e+y 4.84
  raw, select on val   : mean 3.13 (s .8) | cos e+y 4.04 | l2 y 4.15 (tau .03, s .65) | l2 e+y 3.65
  learned (train)      : A 4.46 | Bp 5.43 | Spre 5.58 | Sperm 5.52 | Smean 5.70 | C 5.30
=> With the same scale protocol, raw-mean == learned A (4.45 vs 4.46). Raw cos e+y (5.41) == learned Bp (5.43);
   only Smean's learned past-day map adds +.29pp over raw cos e+y (3/3). Cosine is robust across selection splits, L2 is not.

### PAT-315 mechanism: demo-age bands (`eval_age_bands.py`, 3 seeds)
Same 1353 late-test origins x 862 channels for every band; 4 same-phase demos per band; all demo keys/values out-of-sample
(target window after the train split). tau fixed (train-selected): e 2, y 4, e+y 2/2. 3-seed mean:
  optimal-scale gain % (mean / cos e / cos y / cos e+y) and s_opt:
    w02-05 1.38/1.47/1.49/1.56 s .23 | w06-09 .47/.57/.48/.57 s .14 | w10-13 .55/.62/.57/.63 s .14
    w14-17 .47/.50/.47/.51 s .13     | w18-21 .24/.29/.26/.30 s .09
  gain at s=.55 is negative in every band (-0.8 .. -6.5): on these late-test cases the correction must be shrunk to s~.1-.25.
  cos(e_j, target): mean .074 (w2-5) -> .046 (w18-21); by sim_y decile low->high .05->.15 (w2-5), .03->.11 (w18-21);
                    by sim_e decile .04->.13 (w2-5), .01->.13 (w18-21).
=> Two parts: (1) the unconditional same-phase error template decays with demo age (mean gain 1.38 -> .24, s_opt .23 -> .09):
   time-local drift, not a fixed map. (2) "more similar y / past error -> more similar error" holds at every age with
   about the same slope (top-decile cos ~.11-.16): a weak but age-stable state -> error relation.
   Per-pair error similarity is small (cos <= .16); the late-test subset has much less correctable error than the full test.

### PAT-315 FoX-family retrieval (commit eb562f6, jobs 11380-11382)
Raw cos logits tau_e cos(q,k_j) + tau_y cos(past-day y); test gain % (s3047/s2024/s2025, mean):
  F0 5.35/5.48/5.43 5.42 | Fage 5.54/5.69/5.65 5.63 | Fage_sink 5.55/5.69/5.65 5.63 (sink weight ~.005)
  FoX 1.82/2.56/2.15 2.18 | FoX_sink 3.17/3.50/3.17 3.28 | (Smean 5.70, Bp 5.43)
Fage > F0 3/3 (+.19..22): age decay helps. FoX < Fage 3/3 by 3.1-3.7pp although its best val MSE equals Fage's
(s3047 .3252 vs .3245): test-only collapse. Lead gain at leads 100-150 goes negative. Diagnostics from the first 20 test
batches look normal (log f ~ -4e-4/step, logD smooth -.15 .. -1.0, entropy as Fage) -> suspected gate blow-up on later
test y windows (data-dependent forget summed over up to 2520 steps), not yet verified.

### PAT-315 observed-y key length (training-free cos, select on train; `eval_raw_ylen.py`, jobs 11434-11436)
Test gain % (s3047/s2024/s2025, mean; s = .55 everywhere):
  mean 4.45 | e+y24 5.41 | e+y96 5.62/5.72/5.71 5.68 (tau_e 1, tau_y 4) | e+y192 5.58 | e+y24+y192 5.58
  e+y96c 5.42 | e+y192c 5.33 (window mean removed) | y96 alone 5.55 | y192 alone 5.45
e+y96 > e+y24 3/3 (+.26..29); e+y192 > e+y24 3/3 (+.16..18); centred versions lose the gain.
=> The backbone's current input window (y[t-96:t]) is the best y key; its level matters, not only its shape.
   Training-free e+y96 (5.68) matches the learned Smean (5.70) and beats Fage (5.63).

### PAT-315 one-layer W1 ICL, concatenated key (commit after 517dcd4, `eval_raw_ylen.py` rawylen2)
x_o = [y[o-H-L:o-H], y[o-H:o], E_{o-H}] (480 dims), one cos on the concatenation, softmax, value E_{s_j}; training-free,
tau/s on train. Test gain % (s3047/s2024/s2025, mean):
  cat3 5.28/5.40/5.37 5.35 | cat3n (segments l2-normalised) 5.35/5.48/5.44 5.42 | e+y96 (separate cos, tau_e 1, tau_y 4) 5.68
cat3 < e+y96 3/3 (-.33..34), cat3n < e+y96 3/3 (-.25..27), cat3n > cat3 3/3, cat3 < e+y24 3/3.
=> A single cosine on the concatenation is worse than separate per-segment cosines: the best key weights y96 4x the
   error, and the old forecast's input y[o-H-L:o-H] plus the first half of y[o-H:o] dilute it (cf. y192 < y96).

### PAT-315 multi-step kernel GD of the one-layer ICL (`eval_raw_multistep.py`, jobs 11455-11457)
Key e + y96 (tau_e 1, tau_y 4); f_{k+1} = f_k + eta sum_j a(x, x_j)(e_j - f_k(x_j)); K, eta, s on train; test once.
Test gain % (s3047/s2024/s2025, mean):
  K1 5.63/5.73/5.72 5.69 | self K2 5.75 | self K3 5.71/5.80/5.80 5.77 (val pick) | self K5 5.77 | self K10 5.76
  loo K2 5.76 (val pick) | loo K3 5.74 | loo K5 5.75 | loo K10 5.74
Every K > 1 beats K1 3/3 (+.05..+.09pp); gain saturates at K~2-3; self ~ loo.
=> Multi-step (stacked) in-case GD helps a little and consistently; a few steps are enough.

### PAT-315 PF same-clock forget gate (job 11520/11521, L96 H192)
Test gain % (s3047/s2024/s2025, mean): F0 5.42 | Fage 5.63 | PFm 5.54/5.68/5.62 5.61 | PFc 5.58/5.71/5.67 5.65 | *_sink = no sink.
=> Forgetting only over the same-clock points ~ Fage; the sink is unused.

### PAT-315 lookback fairness: L = 720 backbone (`run_itransformer_base.py --seq_len 720`, job 11453; ICL job 11519)
Backbone test MSE: L96 0.412 | L720 0.3630/0.3657/0.3633 (-11.7%).
ICL on L720 (test gain % over L720, s3047/s2024/s2025): multistep K1 .42/.35/.36 (K>1 no extra; Kval = K1);
  ylen: mean .24/.25/.30 | e+y96 .33/.34/.41 | cat3 .33/.35/.41 | best cat3n <= .44.
=> Pre-set criterion (ICL > 2% on L720) FAILS 3/3. The L96 gain (5.8%) mostly substitutes for the weekly history the
   short-input backbone cannot see.

### PAT-315 upper-bound diagnostic: fresh / cross-channel errors (`eval_fresh_diag.py`, ridge per lead, fit train, lam on val)
Test gain % (s3047/s2024/s2025), segments h1-24 / h25-96 / h97-192 for s3047:
  L96 : ret 5.63/5.74/5.73 | fresh .10/.08/.08 | fresh+ret 5.81/5.90/5.89 | fresh+cross .08/.07/.07 | all 5.80/5.89/5.88
  L720: ret .46/.40/.42 (1.06/.68/.19) | fresh .02/.02/.04 | fresh+ret .49/.41/.46 | fresh+cross .01/.01/.02 | all .48/.40/.45
=> The freshest observed errors (1-step errors of the last 24 h, errors on y_{t-1} from older origins) and all channels'
   recent 1-step errors (PCA 16 x 4 lags) carry ~no linear signal for the next H on Traffic. Remaining gain on L720 <0.5%.

### PAT-315 age scan: demos at every hourly age (`eval_age_scan.py`, 50k test cases, 14-demo sets, gain_opt = best-scale)
gain_opt % (s3047/s2024/s2025):
  L96 : phase 5.52/5.54/5.67 | top-14 by logit over all ages 4.49/4.42/4.40 | top-14 excl same clock 2.79/2.62/2.66 |
        random .09/.08/.05 | oracle top-14 by cos(e_j,target) 39.1/40.2/39.9
  L720: phase .34/.40/.31 | top-14 .15/.17/.11 | excl same clock .03/.05/.01 | random 0 | oracle 39.1/39.8/39.1
Per-age mean cos(e_j, target): same clock .11 (L96) / .04 (L720); same hour other day .01 / .00; other ages 0.
=> Only same-clock demos carry error information; full-history top-k picks look-alike states with unrelated errors and
   is worse than the fixed phase set. The oracle is mostly selection-on-target noise (max cos of 2400 random 192-d
   vectors is ~.3), not an attainable bound.

## PAT-315 Observation-conditioned ICL (C group), 2026-10-28
Motivation: residual ICL value = same-clock backbone errors = periodic error only (FreshCast M_p, Prop 4.3); L720 gain <= 0.4%,
dense error pools worse (L720 s3047 H192: phase 0.41 / daily ~0.1 / hourly ~0.2%). Residual-ICL jobs stopped.
`patchmem/eval_obs_icl.py` (test `test_obs_icl.py`, 4 pass; Codex: 3 fixes applied - C2 ridge floor, 2x2 eta ridge, P chosen on val per b rule)
- demo s (age >= H, ages <= 2880-96-H, same for every L): k = (y[s-96:s]-mu_s)/sigma_s, v = (y[s:s+H]-mu_s)/sigma_s, mu/sigma of y[s-P:s], P in {24,168}, sigma >= 0.1
- level (v^=0) | B softmax(beta k.k_t/96) v, beta in {0,..,16} | C1 one W2 GD step = mean_j (k_j.k_t/96) v_j | C2 closed-form LS W2, lam in {0.01..10}
- r = mu_t + eta sigma_t v^, y^ = f + b (r - f); eta + beta/lam on train (200k), P + b (global / per lead) on val (200k), test once (400k)
- pools phase / daily / hourly; H192, L720 and L96, 3 seeds. Criterion (fixed): C beats B on 3/3 seeds, mean >= 0.3 pt.

### obsicl v1 results (coupled b: y^ = f + b(mu_t - f) + b eta sigma_t v^), H192, test gain %, seeds 3047/2024/2025
- L720 phase: level 0.65/0.61/0.71 | B 0.33/0.35/0.42 | C1 0.56/0.61/0.78 | C2 0.35/0.44/0.63 | Bmean -0.18/-0.06/-0.08
- L720 daily: B 0.27/0.24/0.40 | C1 0.64/0.58/0.72 | C2 0.18/0.07/0.25 | Bmean 0.60/0.52/0.59
- L96 phase: level 0.19/0.11/0.19 | B 5.73/5.56/5.43 | C1 2.81/2.69/2.62 | C2 4.80/4.67/4.62 ; daily B ~1.6, C2 ~1.6, C1 ~0.34; hourly B ~1.1-1.3
- Reading: at L720 a level refresh alone (r = mean of last 168 h, b ~0.05) gives 0.65% > residual ICL 0.38%; nothing beats level;
  the shared b ties the level and shape terms (shape wants b ~0.3, level ~0.05). C1 > B is only C1's smaller shape term.

## obsicl v2 (2026-10-28): fresh tiled candidates + decoupled weights
FreshCast seasonal analogs use reference points 1..8 periods back with the following PERIOD tiled to H (age >= 1 day), our
demos had age >= H (8 days). v2: tiled pool s = t - p j, p in {24,168}, j = 1..8, cont = y[s + h mod p]; readouts shape /
value / both; naive24 / naive168 (tile of the last day / week); y^ = f + a(mu_t - f) + c sigma_t v^, (a, c) LS.
Readout + hyper on train, P + (a, c) on val (global / per lead). Pools tiled, phase, daily. Stats saved (*_obsicl2_s*_stats.pt).
Tests 4 pass; Codex: no bugs. Criterion unchanged: B (tiled) clear gain at L720; C beats B 3/3 seeds, mean >= 0.3 pt.

### obsicl v2 results, H192, test gain %, seeds 2024/2025/3047 (global (a, c); per-lead weights are <= global everywhere)
- L720 tiled: level 0.61/0.71/0.65 | naive24 0.62/0.75/0.66 | naive168 0.11/0.30/0.18 | B(value) 0.77/0.80/0.70 |
  C1 0.84/0.97/0.85 | C2 0.84/0.94/0.78 | Bmean 0.83/0.90/0.82
- L720 phase: B(value, beta 4-8) 1.68/1.70/1.65 | C2:value 1.60/1.70/1.58 | C1 1.15/1.32/1.20 | Bmean 1.35/1.53/1.39 |
  B:shape ~0.2 | B:both ~0.2 ; a ~0.21, c ~0.17
- L720 daily: B 0.73/0.86/0.78 | C1 0.70/0.86/0.74 | C2 0.81/0.93/-0.05 | Bmean 0.65/0.83/0.67
- L96 phase: B 7.70/7.56/7.91 | C2:value 6.69/6.67/6.79 | C1 3.87/3.85/3.97 | Bmean 6.31/6.07/6.46
- L96 tiled: B 6.55/6.53/6.79 | naive168 2.0/2.1/2.1 ; L96 daily: B 2.85/3.06/3.09, C2 3.19/3.31/3.32
- Reading: decoupled (a, c) + value readout of same-clock past weeks lifts L720 from 0.38 (residual ICL) / 0.37 (v1 B) to 1.68%,
  L96 from 5.7 to 7.7%. Fresh tiled candidates are worse than the phase pool (freshness not the driver here). Shape readout hurts.
  C criterion fails: C2 ~ B (-0.08/0.00/-0.07 at L720 phase), C1 well below; retrieval B is best. The working form is a
  linear forecast combination y^ = f + a(mu_t - f) + c(r_B - mu_t), r_B = softmax-weighted raw same-clock past continuations.

## PAT-315 obs ICL v2 sweep (2026-10-29)
- Traffic H96/336/720 x L96/L720, pools phase,tiled (jobs 11782-11787; H192 done above).
- Electricity (code EL): dataset/electricity/electricity.csv from HF thuml/Time-Series-Library (26304 x 321,
  sha256 7e45845d...a497ad). iTransformer official ECL recipe: e_layers 3, d_model 512, d_ff 512, batch 16, lr 5e-4,
  patience 3, 10 epochs. CELLS EL = Traffic window (history 2880+L-96, period 168). `slurm/obs_train_pack.sh EL L H`:
  train 3 seeds then eval_obs_icl phase,tiled. L96/L720 x H96/192/336/720 (jobs 11798-11805).

### v2 Traffic H sweep results (phase pool, test gain %, s2024/s2025/s3047)
| H | L | level | B | C1 | C2 |
|---|---|---|---|---|---|
| 96 | 96 | 0.31/0.33/0.32 | 7.43/7.20/7.24 | 4.34/4.34/4.27 | 6.32/6.29/6.13 |
| 192 | 96 | 0.11/0.19/0.19 | 7.70/7.56/7.91 | 3.87/3.85/3.97 | 3.71/3.62/4.25 |
| 336 | 96 | 0.02/0.01/-0.03 | 7.68/7.76/7.73 | 3.68/3.75/3.69 | 3.15/2.48/2.61 |
| 720 | 96 | -0.63/-0.67/-0.80 | 5.23/4.95/4.93 | 2.49/2.52/2.38 | 4.02/3.96/3.79 |
| 96 | 720 | 1.14/1.14/0.99 | 1.39/0.98/0.86 | 1.72/1.60/1.46 | 0.39/0.00/-0.29 |
| 192 | 720 | 0.61/0.71/0.65 | 1.68/1.70/1.65 | 1.15/1.32/1.20 | 0.50/0.47/0.64 |
| 336 | 720 | 0.26/0.58/0.43 | 1.66/1.64/1.70 | 0.94/1.19/1.11 | -0.12/-0.15/-0.50 |
| 720 | 720 | -0.41/-0.30/-0.43 | -0.39/-0.62/-0.18 | -0.39/-0.34/-0.23 | -2.39/-0.45/-0.18 |
Tiled pool better than phase only at H96 L720. H720 L720: all arms negative incl. level (val-fit (a,c) does not transfer; phase pool only ages 720..2064).

## FreshCast reproduction (eval_freshcast.py, commit 49caf2b + median/den fixes)
Same frozen iTransformer checkpoints, full test set (all origins x channels), 3 seeds; B = our obs-ICL phase-pool B
with its obsicl2-selected config re-evaluated on the same full test.
Test MSE averaged over H in {96,192,336,720}; EL L720 H720 gains FC 6.36/5.73/6.40, B 2.71/2.52/3.40 (s2024/2025/3047):
| data | L | base | FreshCast (ours) | FreshCast (paper) | B |
|---|---|---|---|---|---|
| Traffic | 96 | 0.4217 | 0.3823 (-9.3%) | 0.424 -> 0.384 | 0.3926 (-6.9%) |
| Traffic | 720 | 0.3790 | 0.3716 (-1.9%) | 0.387 -> 0.376 | 0.3753 (-1.0%) |
| Electricity | 96 | 0.1754 | 0.1522 (-13.2%) | 0.175 -> 0.153 | 0.1592 (-9.3%) |
| Electricity | 720 | 0.1625 | 0.1506 (-7.3%) | 0.163 -> 0.151 | 0.1572 (-3.3%) |
Reproduction matches the paper. FreshCast > B on 3/3 seeds in 14/16 cells; Traffic L720 H336 FC wins 2/3 (s3047 FC 1.30 < B 1.65); Traffic L720 H192 B wins 2/3 (avg B 1.66% vs FC 1.46%).
Untrained rule is weak (negative on Traffic L720 and H720). RKR weight: ~half on seasonal value readouts, ~half on shape analogs.

## Case-wise ICL calibration on top of FreshCast (eval_fc_icl.py, commit 3df13b1) — NEGATIVE
w_t = clip((sum_demo e.d + kappa w^ D0)/(sum_demo d.d + kappa D0), 0, 1), demos = last N fully observed origins
(s <= t-H), forms scalar / 8 lead buckets, N in {24,168,672}, kappa in {0,1,4,16,64}, chosen on full val.
Test MSE change vs FreshCast (same run, s2024/2025/3047): worse on 3/3 seeds in 15/16 cells, never better on 3/3;
range -0.03 .. -2.8% (largest: EL L720 H720 -2.8%, T/EL L96 H336 -1..-2.5%). On val the best grid point is
within +-1% of the global w^ (often worse), so local delayed calibration carries no usable signal.
Grid lacked kappa = inf (= FreshCast) as a fallback; does not change the conclusion.
Consistent with the TTA delay audit (2610.12232): feedback from labels >= H old gives <= 0.5% outside ETTm1.

## Predictable-error decomposition on top of FreshCast (eval_decomp.py) — v1 INVALID, v2 NEGATIVE
y^ = f + w^ d + a_b m_state + b_b m_x (a_b, b_b: 8 lead buckets, val least squares; no ratio mixing).
Term 1 (state-explainable): one w per channel shared by all leads, phi[t,h] = [1, d, rv-f, rs-f, level, log scale,
sin/cos of the TARGET hour of day / week]; w0 = ridge on the FreshCast calibration origins; then one GD step on periodic
same-clock demos (ages 24j / 168j >= H) and one on the N most recent demos (s <= t-H), step eta~ / sum|phi|^2
(= two linear-attention layers). Families S_0 / S_per / S_rec / S_2, (eta~1, eta~2, N) chosen on full val.
Term 2 (history increment), value rho = u - m_state, no q/k: per channel a 168-slot state indexed by the target
hour of week, c' = (I - beta u u^T) c (u learned relative to the current hour from the LQ window + d mu + d log sd),
written slots <- lam c' + (1-lam) rho_{t-H} / sig^gamma, read sig_t^gamma c[(t+h) mod 168].
Arms S_2+P (full), S_2+P_I (A = I: lam-EMA phase template), S_2+TEFL (MLP on 7 lagged rho). Trained on the 2016
train-tail origins (truncated BPTT 168). Criteria: S_2 > FC, S_per, S_rec and S_2+P > S_2, S_2+P_I, S_2+TEFL, 3/3.
Unit tests patchmem/test_decomp.py (GD == direct loops, causality of term 1 and term 2, write/read phase alignment,
time equivariance of the Householder direction, sigma gain). Runs: T / EL x L {96, 720} x 3 seeds at H192.
v1 (commit 40bd1a2) INVALID: level / scale z-scored by the 12-week train-tail channel std (EL channels with std 0,
|z| up to 3.5e6 on val/test; Traffic q99 23.5) -> m_state blew up, all pooled coefficients ~0 on EL. Traffic v1 vs FC
(%, 3 runs): S_2 +0.11/+0.22/+0.02, S_2+P +0.00/+0.12/+0.02, S_2+P_I -0.21/0.00/-0.09, S_2+TEFL +0.25/+0.36/+0.26.
v2: z clipped to +-5, std floor 0.05, level / scale stats over all train windows; phase = one-hot target hour (24) +
day (7) (NP 37); rho = u - a_b m_state; carried state indexed by (lead bucket, target hour of week), 8 x 168, one
Householder over the whole state (cross-bucket mixing); fit2 singular fallback keeps the better single term.
Runs 2 tasks per 3090 (3 per GPU OOMed in v1). v1 jsons renamed *_v1.json.
v2 result (commit 2784b6a), test MSE change vs FreshCast (%), seeds 2024/2025/3047:
| data L | S_0 | S_2 | S_2+P | S_2+P_I | S_2+TEFL |
|---|---|---|---|---|---|
| T 96 | -0.61/-0.49/-0.48 | -0.40/-0.57/-0.41 | -0.26/-0.46/-0.25 | -0.17/-0.49/-0.26 | -0.18/-0.30/-0.13 |
| T 720 | -0.71/-0.67/-0.51 | -0.66/-0.68/-0.32 | -0.66/-0.75/-0.36 | -1.07/-1.09/-0.77 | -0.55/-0.57/-0.21 |
| EL 96 | -1.70/-1.94/-1.40 | -1.30/-1.56/-1.07 | -1.28/-1.53/-1.06 | -1.24/-1.53/-1.02 | -1.00/-1.33/-0.78 |
| EL 720 | -0.97/-1.45/-1.27 | -0.19/-0.72/-0.44 | -0.20/-0.72/-0.44 | -0.24/-0.77/-0.48 | -0.03/-0.55/-0.24 |
Every arm worse than FC on 12/12. The val-fitted linear state map (37 features incl. one-hot hour/day) does not
transfer val -> test (S_0 worst); v2 is worse than v1 on Traffic. Term 2: S_2+P > S_2 3/3 on T L96 only; P > P_I 3/3
on T L720. User clarification afterwards: "state" was meant as the current INPUT x (does the backbone make a fixed
error for a given input pattern), not statistics -> term 1 to be redesigned as input-keyed error retrieval.

## Standalone decomposition (eval_decomp2.py) — NEGATIVE vs FreshCast
User: state = the backbone INPUT x (does the backbone make a fixed error for a given input), method standalone on
the backbone (FreshCast only as a baseline, read from its json). y^ = f + a_b m_state + b_b m_x.
S: key = the backbone's input window (length L, instance-normalised), same-channel demos s <= t-H, top-k cos, linear
attention m = (1/k) sum cos e_s, k in {16, 64, 256} on val. X (term 2), (lead bucket x target hour of week) carried
state: Pd = observation-diagonal writes (the newest observed target's errors at every lead) + Householder; Pd_I = A = I;
Pf = v2 whole-matured-vector writes (freshness ablation); TEFL = MLP on the 7 latest diagonals. Each X alone (rho = e)
and stacked (rho = e - a~ m_state, a~ from the train tail). Reported: S, X, S(+)X (joint val LS), S->X.
Criteria (fixed): S > base 3/3; Pd > Pd_I, TEFL (alone) 3/3; S(+)Pd and S->Pd > S 3/3; Pd > Pf 3/3; vs FreshCast.
Tests patchmem/test_decomp2.py (top-k == brute force, diagonal, diag writes vs a dict reference, diag causality,
full mode == v2 step, input normalisation).

Result (H192, 12/12 runs, test MSE gain % vs base; ranges over seeds 3047/2024/2025; k = 16 chosen on val in 12/12,
the grid's lower edge):

| setting | S | Pd | Pd_I | Pf | TEFL | S(+)Pd | S(+)Pd_I | S->Pd | best arm vs FC |
|---|---|---|---|---|---|---|---|---|---|
| T L96 | +4.1~4.2 | +2.6~2.8 | +0.5~0.8 | +2.6~2.8 | +0.7 | +5.0~5.2 | +4.0~4.2 | +4.9~5.0 | -4.1 ~ -5.2 |
| T L720 | +0.7~0.8 | +0.1 | -0.4~-0.6 | +0.2 | 0.0~0.1 | +0.7~0.8 | -0.1~+0.2 | +0.7~0.8 | -0.3 ~ -0.8 |
| EL L96 | +4.2~4.3 | +2.3~3.3 | +2.7~3.8 | +0.9 | +0.6~0.8 | +5.3~5.7 | +5.3~5.7 | +5.3~5.6 | -9.2 ~ -11.0 |
| EL L720 | +0.8~0.9 | +0.2 | +0.1 | 0.0 | +0.1 | +0.9 | +0.8 | +0.9 | -7.7 ~ -8.5 |

Criteria: S > base 12/12 PASS. S(+)Pd > S 11/12 (T L720 s3047 +0.78 vs +0.79) FAIL; S->Pd > S 11/12 (same seed) FAIL.
Pd > TEFL PASS. Pd > Pd_I FAIL on EL L96 (Pd_I better 3/3; stacked S(+)Pd ~ S(+)Pd_I): the transition helps on Traffic
only. Pd > Pf FAIL (holds on EL; T L96 2/3, T L720 Pf better). vs FreshCast: worse 12/12 (best arm gets about half of
FC's gain). Takeaway: the input-window error template (S) is a real, stable gain over base and term 2 adds ~1% on top at
L96, but the standalone method is not competitive with retrieving y continuations (FC).


## Input-keyed error retrieval stacked on FreshCast (eval_fc_stack.py, commit 7a826dc) — PASS on EL L96 only
y^ = y^_FC + a_b m, y^_FC = f + w^ (r - f) reproduced in-stream (asserted == freshcast json test MSE).
m = (1/k) sum_{top-k cos(x~_t, x~_s), s <= t-H} cos * v_s, key = backbone input window (L, instance-norm), same channel.
Arms: S_fc v = e_FC = y - y^_FC (primary); S_e v = e = y - f. a_b (8 lead buckets) val LS, k in {4, 8, 16, 64} on val.
Criterion (fixed): S_fc beats FreshCast on 3/3 seeds, mean gain >= 0.3%. T / EL x L {96, 720} x 3 seeds, H192,
jobs 12465-12468 (one 3090 per (data, L), 2 tasks in parallel). Codex: only fix = test MSE for the chosen k only.
Result (12/12, FC reproduced in-stream == json in all runs), test MSE change vs FreshCast %, s2024/2025/3047 (k chosen):
| setting | FC gain vs base | S_fc | S_e |
|---|---|---|---|
| T L96 | +9.0/+10.0/+9.3 | +0.19/+0.06/+0.15 (k16) | -0.19/-0.28/-0.18 (k16) |
| T L720 | +1.5/+1.7/+1.2 | -0.04/-0.07/-0.04 (k64) | -0.14/-0.15/-0.10 (k64) |
| EL L96 | +13.6/+14.2/+14.7 | +0.48/+0.43/+0.42 (k8) | +0.61/+0.57/+0.48 (k8) |
| EL L720 | +8.1/+8.5/+8.6 | -0.19/-0.17/-0.26 (k64) | +0.32/+0.29/+0.22 (k16) |
Criterion (S_fc 3/3, mean >= 0.3%): PASS EL L96 (mean +0.44); T L96 3/3 but mean +0.13 FAIL; L720 FAIL.
S_e (secondary): EL L96 +0.55, EL L720 +0.28 3/3; Traffic negative. Traffic val gains are +0.6..+0.8% (S_fc) but
test +0.06..+0.19%: the val-fitted a_b does not transfer on Traffic. Gains on top of FC are <= ~0.6% everywhere.


### fc_stack v2 (commit 3df190c + json-check warning): joint global (w1, w2) in [0,1]^2 on all val origins
y^ = f + w1 (r - f) + w2 m. 12/12 (V100 reruns: FC from RKR retraining differs from the 3090 json by ~0.1%, every arm is
compared to its own run's FC). Test MSE change vs FC %, s2024/2025/3047:
| setting | FC_refit | S_fc | J_fc | S_e | J_e | J_fc w1 / w2 |
|---|---|---|---|---|---|---|
| T L96 | -0.02/0.00/-0.16 | +0.12/+0.06/+0.05 | +0.26/+0.17/+0.05 | -0.18/-0.28/-0.29 | +0.24/+0.02/-0.05 | 0.65 / 0.3 |
| T L720 | +0.06/+0.02/+0.19 | -0.04/-0.07/-0.05 | +0.02/-0.04/+0.15 | -0.14/-0.15/-0.12 | +0.09/+0.04/+0.28 | 0.48 / 0.35 |
| EL L96 | +0.01/-0.01/-0.17 | +0.48/+0.43/+0.42 | +0.41/+0.31/+0.20 | +0.61/+0.56/+0.48 | +0.48/+0.44/+0.31 | 0.66 / 0.10 |
| EL L720 | +0.10/-0.14/-0.21 | -0.19/-0.17/-0.26 | +0.22/-0.01/-0.11 | +0.32/+0.29/+0.22 | +0.36/+0.14/+0.03 | 0.53 / 0.02 |
Criterion (J_fc 3/3, mean >= 0.3%): FAIL everywhere (EL L96 mean +0.31 but s3047 +0.20; T L96 mean +0.16).
Global scalars help on Traffic vs per-bucket a_b (val -> test transfer), hurt on EL. FC_refit ~0: gains come from m.

## Why B beats FreshCast on Traffic L720: 2x2 {candidates} x {weights} (eval_fc_vs_b.py, commit 74a1708)
z = (l = mu_t - f, s_fc = r_FC - mu_t, s_B = sd_t v^_B); all weights global, fitted on ALL val origins, test once.
FC = w (l + s_fc) clipped | FC_ac = a l + c s_fc | B_w = w (l + s_B) | B = a l + c s_B | FCB = a l + c1 s_fc + c2 s_B.
Test gain % vs base, s2024/2025/3047 (24/24 runs on V100; FC_orig / B_orig reproduce the freshcast json within 0.2):
| L720 | FC | FC_ac | B_w | B | FCB |
|---|---|---|---|---|---|
| T H96 | 2.56/2.52/2.23 | 2.81/2.78/2.39 | 0.58/0.59/0.38 | 1.24/1.22/0.88 | 2.12/2.05/1.74 |
| T H192 | 1.55/1.74/1.21 | 1.69/1.85/1.36 | 1.35/1.37/1.26 | 1.68/1.69/1.61 | 1.76/1.84/1.40 |
| T H336 | 2.40/2.45/1.61 | 2.43/2.51/1.67 | 1.63/1.37/1.47 | 1.70/1.65/1.66 | 2.44/2.51/1.66 |
| T H720 | 1.70/2.21/2.12 | 1.64/2.11/2.00 | -0.25/-0.35/0.11 | -0.49/-0.52/-0.14 | 1.24/1.44/1.74 |
| EL H96 | 7.08/6.84/6.93 | 7.02/6.77/6.83 | 2.64/2.40/3.38 | 2.98/2.69/3.56 | 7.07/6.76/6.94 |
| EL H192 | 8.16/8.39/8.44 | 8.06/8.28/8.32 | 3.97/4.02/3.61 | 4.16/4.20/3.71 | 8.23/8.38/8.36 |
| EL H336 | 7.25/8.36/7.70 | 7.07/8.25/7.55 | 2.72/3.72/2.69 | 2.85/3.94/2.95 | 7.15/8.28/7.61 |
| EL H720 | 6.38/5.92/6.85 | 6.11/5.74/6.87 | 2.49/2.20/2.77 | 2.70/2.52/3.40 | 6.15/5.74/6.88 |
Criteria: weighting effect (FC_ac > FC and B > B_w, 3/3): PASS T H96/H192/H336 (FC_ac - FC only +0.03..+0.26), FAIL T H720
and all EL (FC_ac < FC). Candidate effect (B > FC_ac and B_w > FC, 3/3): FAIL in every cell (1 of 24 runs: T H192 s3047).
FC_ac fits a ~= c (FC's tied weight is nearly optimal for its own candidates); B fits a 0.13-0.32, c 0.09-0.28.
B beats FC on 47-57% of Traffic channels at H192, 11-27% elsewhere on Traffic, 2-10% on EL.
FCB (nests FC_ac) is worse than FC_ac on test at T H96 / H720 (-0.5..-0.8): adding s_B does not transfer val -> test.
Reading: B's win at T L720 H192 is not a better candidate set or weighting; FC dips at H192 (1.2-1.7 vs 2.2-2.5 at
H96 / H336) and is seed-unstable (s3047), while B is flat (1.6-1.7 over H192 / H336). Nothing to extend.

## decomp2 Pd_err: PaTH carry driven by error features (eval_decomp2.py, commits c6d5286 / 8b057ef, job 12873 + laptop task 7)
Pd_err x = (per-bucket mean D[k]/sd_t, its change from k-1, dmu, dlogsd), train-tail standardized, clip +-5; Pd x = 96-step window.
Test gain % vs base, s3047/2024/2025, H192 (EL L720 s2025 from laptop 2be3ef9, reproduces the cluster's shared arms exactly):
| setting | FC | S | Pd | Pd_I | Pd_err | S(+)Pd | S(+)Pd_I | S(+)Pd_err | S->Pd_err |
|---|---|---|---|---|---|---|---|---|---|
| T L96 | 9.33/8.96/10.00 | 4.09/4.11/4.17 | 2.69/2.80/2.56 | 0.53/0.59/0.76 | 3.55/2.94/3.10 | 4.98/5.18/5.00 | 4.01/4.10/4.21 | 5.65/5.47/5.11 | 5.67/5.25/5.46 |
| T L720 | 1.19/1.47/1.72 | 0.79/0.66/0.75 | 0.07/0.12/0.14 | -0.42/-0.62/-0.62 | 0.24/0.20/0.17 | 0.78/0.69/0.80 | 0.19/-0.10/-0.06 | 0.86/0.72/0.79 | 0.88/0.72/0.79 |
| EL L96 | 14.70/13.62/14.23 | 4.25/4.20/4.29 | 2.63/3.26/2.31 | 2.70/3.76/2.95 | 3.19/3.30/3.31 | 5.36/5.67/5.29 | 5.26/5.70/5.39 | 5.63/5.63/5.73 | 5.40/5.44/5.62 |
| EL L720 | 8.63/8.07/8.51 | 0.81/0.89/0.90 | 0.16/0.17/0.17 | 0.08/0.11/0.13 | 0.15/0.10/0.25 | 0.86/0.94/0.93 | 0.76/0.85/0.87 | 0.76/0.81/0.91 | 0.76/0.80/0.91 |
Criterion (Pd_err > Pd and > Pd_I, 3/3): PASS T L96, T L720; FAIL EL L96 (Pd_I wins s2024), EL L720 (Pd wins s3047/2024).
Stacked (S(+)Pd_err > S(+)Pd and > S(+)Pd_I, 3/3): PASS T L96 only (+0.67/+0.29/+0.11 over S(+)Pd); T L720 s2025 -0.01,
EL L96 s2024 -0.04. Everything stays far below FC (best T L96 5.7 vs 9-10; EL L96 5.7 vs 14).

## What the iTransformer-TEFL map learns (diag_tefl_map.py, commit ef02e9f, V100 jobs 13199 / 13200)
g = W2 relu(W1 eps), eps = f-only error vector issued one horizon ago; TEFL_paper_L96_E192, H192, s3047/2024/2025.
Retained share of the TEFL test gain when g is replaced by a val-fitted mimic (alpha_h per lead), base / TEFL MSE
match tefl_metrics.json:
| data | TEFL gain % | diag | mean | phase | last | week | amp | lin | lin r1 | lin r2 | lin r4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh2 | 4.8-5.0 | .49-.51 | .84-.87 | .58-.59 | .48-.50 | .39-.40 | .02 | .99-1.00 | .73-.97 | .97-.98 | .98-.99 |
| ETTm1 | 9.3-9.8 | .32-.33 | .17-.18 | .09-.11 | .44-.47 | - | .03 | .93-.94 | .27-.33 | .71-.75 | .88-.89 |
| ETTm2 | 6.4-6.6 | .37-.39 | .48-.51 | .27-.28 | .41-.43 | - | .02-.03 | .90-.91 | .47-.52 | .76-.81 | .83-.85 |
| Weather | 7.3-7.5 | .17-.18 | .34 | .29-.30 | .36 | - | .13-.14 | .94 | .45-.46 | .66-.70 | .93-.94 |
| Traffic | 4.8-4.9 | .13 | .02 | .06-.07 | .02 | .36-.37 | .01-.02 | .90-.91 | .25 | .36-.37 | .53-.54 |
Criterion (a simple mimic >= 80% on 3/3): PASS only ETTh2 (mean = level shift). Common: g ~ linear in eps (R^2 .80-.95)
although every ReLU unit switches; |g| small (std ratio .06-.3 of eps); corr(g[h], eps[h]) -0.5..-0.8 at far leads
(pulls against the last horizon's error); input weight spread over the whole eps; gain on 99-100% of channels; amp ~ 0
(not a forecast-amplitude rescale). Rank: ETTh2 ~1 (output profile ~constant over leads, cos -0.97), ETTm / Weather ~4,
Traffic high (rank 4 = 53%), with 37% of A's energy on same-hour-of-day entries (chance 4%).

## Value of the backbone's own errors vs TEFL (eval_err_value.py 1e6b658, eval_err_value2.py 6da9b21; V100 srun 13231/13232, 13237/13238)
Post hoc on the frozen standalone iTransformer (L96 H192), linear predictor of e_t = E[t] (192 leads) from
eps = E[t-H] (TEFL's input), u = one-step error history (96), D = newest observation diagonal (192), bias.
Fit on train origins (targets inside train; Traffic last 6048), lam on val, test once; TEFL = official test MSE of the
jointly trained TEFL_paper_L96_E192 (same test windows). Stage 1 pooled (all channels share W); stage 2 per channel
W_c shrunk toward W_pool (lam2 on val). Test MSE improvement vs TEFL %, s3047/2024/2025:
| data | TEFL vs base | pooled eps | pooled all | per-ch eps+D | per-ch all | per-ch all vs base |
|---|---|---|---|---|---|---|
| ETTh2 | +1.8..1.9 | +5.7/+6.2/+5.9 | +5.2/+6.2/+5.6 | +6.6/+7.1/+6.8 | +6.0/+7.0/+6.4 | +7.7/+8.7/+8.2 |
| ETTm1 | +5.2..5.9 | -0.3/-0.3/+0.9 | +0.4/+0.7/+2.1 | +0.9/+0.5/+2.2 | +0.4/+0.3/+2.7 | +6.2/+6.2/+7.7 |
| ETTm2 | +7.1..7.4 | -0.9/+0.7/-0.9 | -0.0/+2.5/+2.5 | +1.0/+4.0/+4.4 | +0.8/+4.2/+4.1 | +7.8/+11.0/+11.2 |
| Weather | +6.2..7.3 | -1.1/-1.2/-1.6 | +1.6/-0.2/-0.4 | +6.8/+3.5/+3.3 | +6.8/+3.8/+3.8 | +13.2/+9.8/+10.8 |
| Traffic | -1.6..-2.2 | +5.8/+5.9/+5.4 | +6.0/+6.1/+5.5 | +5.8/+5.9/+5.4 | +6.0/+6.0/+5.5 | +4.0/+4.0/+4.0 |
Criterion (arm < TEFL on 3/3 seeds, per dataset): per-channel eps+D and per-channel all PASS on 5/5 datasets; pooled all
fails ETTm2 (s3047 -0.04) and Weather (2/3). Fresh D adds to eps on ETTm1/ETTm2/Weather; per-channel heads matter on
Weather / ETTm2 (lam2 0.1-1), not on ETTh2 / Traffic (lam2 10-100). Smallest margin ETTm1 (+0.3..+2.7).
Codex: no leakage (feature targets <= t-1, fit targets inside train, lam on val); test moments are computed before
selection but never enter it.
