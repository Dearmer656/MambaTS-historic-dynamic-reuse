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
