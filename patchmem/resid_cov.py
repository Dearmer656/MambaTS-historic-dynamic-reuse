"""PAT-307 residual covariance: how much of the FUTURE forecast residual is linearly explained by the HISTORY residuals, per split.
Models (seed 3047, H = 336 = 7 patches, P = 48): A1 = recurrent R1 without correction (one checkpoint, any L); A0 = official
MambaTS (block, flatten head), the checkpoint trained for (L, 336). L in {384, 720, 1440}. Data: ETTh2, Weather. Splits: train/val/test.
Origin t (absolute index in the split's data_x, the dataset's StandardScaler units, no RevIN), lookback x[t-L:t]:
  future residual   r_fut(t)[k]  = x[t+kP : t+(k+1)P] - yhat(t)[kP:(k+1)P]                            lead k = 0..6  (P values per variable)
  history residual  r_hist(t)[m] = the 1-patch-ahead error of patch [t-mP, t-(m-1)P), m = 1..8:
     A1: the in-window teacher-forced error x - read(h) of that patch (what the correction module reads), from the window's own pass
     A0: x - yhat(t-mP)[0:P], A0's forecast issued at the past origin t-mP (its first patch)
  joint m: r_hist(t)[1..m] concatenated (m*P values per variable; for A1 m <= L/P - 1).
Same origins for both models: t in [L + 8P, len(data_x) - H]. Per variable c, lead k, m: ridge regression r_fut[k] ~ r_hist[1..m]
(+ intercept, lambda = 0.01 * tr(X'X)/p on the centred fit set). Tables [m][k], pooled over variables (sum SSE / sum SST):
  R2adj  within a split: fit and score on the same split, adjusted R^2 with the ridge effective dof (the within-split covariance);
  R2cov  train -> val / test, slope only: both sides centred on the scored split's own means (does the covariance transfer; a mean
         shift between periods is not counted);
  MSEred train -> val / test, slope + intercept from train: 1 - SSE / sum(r_fut^2), the fraction of the uncorrected residual MSE a
         linear correction fitted on train removes (bias included).
(A 2-fold within-split cross-fit was dropped: the residual mean drifts between the halves of a split, which dominated it.)
Sanity: the A1 / A0 test MSE over the kept origins vs their logs (all origins) is printed.
usage: python patchmem/resid_cov.py --data E|W [--L 384,720,1440]"""
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args
from patchmem.plug_a0 import load_a0, cache_split
from data_provider.data_factory import data_provider

P, H, NP, MMAX, SEED = 48, 336, 7, 8, 3047
CFG = {
    'E': dict(common='--task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 '
                     '--dec_in 7 --c_out 7 --data ETTh2 --features M --label_len 48 --pred_len 336 --d_layers 2 --factor 1 --itr 1 '
                     '--n_heads 16 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA',
              r1='--model_id ETTh2_r1 --model MambaTS --seq_len 1440 --e_layers 5 --d_model 16 --dropout 0.3 --batch_size 8 --learning_rate 0.001',
              r1des='R1h0s3047', r1ck='long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_R1h0s3047_0',
              a0='--model MambaTS --e_layers 2 --d_model 32 --dropout 0.3 --batch_size 32 --learning_rate 0.0005',
              a0id='ETTh2_{L}_336', a0ck='long_term_forecast_ETTh2_{L}_336_MambaTS_ETTh2_ftM_sl{L}_ll48_pl336_dm32_nh16_el2_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0s3047_0',
              a0log='A0_ETTh2_L{L}_H336_s3047.txt', r1log='E_A1_s3047.txt'),
    'W': dict(common='--task_name long_term_forecast --is_training 0 --root_path ./dataset/weather --data_path weather.csv --enc_in 21 '
                     '--dec_in 21 --c_out 21 --data custom --features M --label_len 48 --pred_len 336 --d_layers 2 --factor 1 --itr 1 '
                     '--n_heads 16 --d_model 32 --d_ff 512 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA',
              r1='--model_id weather_r1 --model MambaTS --seq_len 1440 --e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005',
              r1des='R1Wh0s3047', r1ck='long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_R1Wh0s3047_0',
              a0='--model MambaTS --e_layers 2 --dropout 0.3 --batch_size 8 --learning_rate 0.001',
              a0id='weather_{L}_336', a0ck='long_term_forecast_weather_{L}_336_MambaTS_custom_ftM_sl{L}_ll48_pl336_dm32_nh16_el2_dl2_df512_fc1_ebtimeF_dtTrue_ExpA0s3047_0',
              a0log='A0_W_L{L}_H336_s3047.txt', r1log='W_A1_s3047.txt'),
    'T': dict(common='--task_name long_term_forecast --is_training 0 --root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 '
                     '--enc_in 32 --dec_in 32 --c_out 32 --data custom --features M --label_len 48 --pred_len 336 --d_layers 2 --factor 1 '
                     '--itr 1 --n_heads 16 --d_model 32 --d_ff 512 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA',   # R1 only (no A0)
              r1='--model_id traffic32_r1 --model MambaTS --seq_len 1440 --e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005',
              r1des='R1Th0s3047', r1ck='long_term_forecast_traffic32_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_R1Th0s3047_0',
              r1log='T_A1_s3047.txt'),
}


def split_x(a, L, flag):
    b = copy.copy(a); b.seq_len, b.pred_len = L, H
    return np.asarray(data_provider(b, flag)[0].data_x, dtype=np.float32)


@torch.no_grad()
def a1_resid(ex, X, L, origins, dev, bs=128, mmax=MMAX):
    """A1 future residual (N, H, C) and in-window history errors of the last MMAX patches (N, MMAX, P, C), m = 1 first."""
    m = ex.model; m.eval(); ex._fix_eval_order()
    T = L // P; mh = min(mmax, T - 1)
    Xt = torch.as_tensor(X)
    F, Hs = [], []
    for i0 in range(0, len(origins), bs):
        ts = origins[i0:i0 + bs]
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        out = m(bx, NP)
        y = torch.stack([Xt[t:t + H] for t in ts]).to(dev)
        F.append((y - out['pred']).cpu())
        eh = bx[:, P:].reshape(len(ts), T - 1, P, -1) - out['hist']                          # (b, T-1, P, C) patches 1..T-1
        Hs.append(eh[:, -mh:].flip(1).cpu())                                                # m = 1 (most recent) first
    return torch.cat(F).numpy(), torch.cat(Hs).numpy()


def a0_resid(Y, X, L, origins, mmax=MMAX):
    """A0 from the cached forecasts Y (index s - L for origin s): future residual and the past-origin first-patch errors."""
    o = np.asarray(origins)
    F = np.stack([X[t:t + H] for t in o]) - Y[o - L]
    Hs = np.stack([np.stack([X[t - mm * P:t - mm * P + P] - Y[t - mm * P - L][:P] for mm in range(1, mmax + 1)]) for t in o])
    return F, Hs


def fit_eval(Xf, Yf, Xe, Ye, mh, alpha=0.01):
    """Ridge of every lead's P targets on the m most recent history patches, all m at once (nested Gram blocks).
    Returns (mh, NP) arrays: SSE_slope (eval-centred: the fit set's slope only, X and Y centred with the EVAL set's means),
    SSE_full (the fit set's slope and intercept), SST (about the eval mean), SS0 (sum of squares about 0), dof (ridge effective params)."""
    mx, my = Xf.mean(0), Yf.mean(0)
    Xc, Yc = Xf - mx, Yf - my
    G, XY = Xc.T @ Xc, Xc.T @ Yc
    ex_, ey = Xe.mean(0), Ye.mean(0)
    sst = ((Ye - ey) ** 2).reshape(len(Ye), NP, P).sum((0, 2))
    ss0 = (Ye ** 2).reshape(len(Ye), NP, P).sum((0, 2))
    out = np.zeros((5, mh, NP))
    for mm in range(1, mh + 1):
        d = mm * P
        Gm = G[:d, :d]; lam = alpha * np.trace(Gm) / d
        B = np.linalg.solve(Gm + lam * np.eye(d), XY[:d])
        rs = (Ye - ey) - (Xe[:, :d] - ex_[:d]) @ B
        rf = (Ye - my) - (Xe[:, :d] - mx[:d]) @ B
        ev = np.linalg.eigvalsh(Gm)
        out[0, mm - 1] = (rs ** 2).reshape(len(Ye), NP, P).sum((0, 2))
        out[1, mm - 1] = (rf ** 2).reshape(len(Ye), NP, P).sum((0, 2))
        out[4, mm - 1] = (ev / (ev + lam)).sum()
    out[2] = sst; out[3] = ss0
    return out


def per_var(F, Hs, c):
    return Hs[:, :, :, c].reshape(len(Hs), -1).astype(np.float64), F[:, :, c].astype(np.float64)


def analyse(R):
    """R[split] = (F (N, H, C), Hs (N, mh, P, C)). Tables [m][k] pooled over variables (sum over variables before the ratio):
    R2adj[split]  in-sample ridge fit and score on the same split, adjusted for the ridge effective dof: the within-split covariance;
    R2cov[val|test] slope fitted on train, both sides centred on the scored split: does the covariance structure transfer;
    MSEred[val|test] slope + intercept fitted on train: the fraction of the uncorrected residual MSE a linear correction removes."""
    C = R['train'][0].shape[-1]; mh = R['train'][1].shape[1]
    out = {'R2adj': {}, 'R2cov': {}, 'MSEred': {}, 'mse': {}, 'var': {}}
    for sp, (F, Hs) in R.items():
        out['mse'][sp] = [float((F[:, k * P:(k + 1) * P] ** 2).mean()) for k in range(NP)]
        n = len(F)
        tv = np.zeros((C, 5, mh, NP))
        for c in range(C):
            Xd, Yd = per_var(F, Hs, c)
            tv[c] = fit_eval(Xd, Yd, Xd, Yd, mh)
        dof = tv[:, 4].mean(0)
        r2 = 1 - tv[:, 0].sum(0) / tv[:, 2].sum(0)
        out['R2adj'][sp] = (1 - (1 - r2) * (n - 1) / (n - dof - 1)).tolist()
        out['var'][f'R2adj_{sp}'] = (1 - tv[:, 0] / tv[:, 2]).tolist()
    Ft, Ht = R['train']
    for sp in ('val', 'test'):
        F, Hs = R[sp]
        tv = np.zeros((C, 5, mh, NP))
        for c in range(C):
            Xa, Ya = per_var(Ft, Ht, c); Xe, Ye = per_var(F, Hs, c)
            tv[c] = fit_eval(Xa, Ya, Xe, Ye, mh)
        out['R2cov'][sp] = (1 - tv[:, 0].sum(0) / tv[:, 2].sum(0)).tolist()
        out['MSEred'][sp] = (1 - tv[:, 1].sum(0) / tv[:, 3].sum(0)).tolist()
        out['var'][f'R2cov_{sp}'] = (1 - tv[:, 0] / tv[:, 2]).tolist()
        out['var'][f'MSEred_{sp}'] = (1 - tv[:, 1] / tv[:, 3]).tolist()
    return out


def show(tag, tab):
    t = np.asarray(tab)
    print(f'  {tag:22s} lead k: ' + ' '.join(f'{k + 1:>6d}' for k in range(NP)) + '    mean', flush=True)
    for mm in range(t.shape[0]):
        print(f'  {"":22s} m={mm + 1}:     ' + ' '.join(f'{v:+.3f}' for v in t[mm]) + f'  {t[mm].mean():+.3f}', flush=True)


def logged_test(fn, L, r1):
    """Test MSE at (L, 336) from a training log: '[R1 test] L=... H=336' (A1) or the last 'mse:' line (A0)."""
    path = f'patchmem/logs/server/{fn}'
    if not os.path.exists(path):
        return None
    lines = open(path).read().splitlines()
    if r1:
        hit = [l for l in lines if l.startswith('[R1 test]') and f'L={L:4d} H=336' in l]
        return float(hit[0].split('| mse ')[1].split()[0]) if hit else None
    hit = [l for l in lines if l.startswith('mse:')]
    return float(hit[-1].split('mse:')[1].split(',')[0]) if hit else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--L', default='384,720,1440')
    a_ = ap.parse_args()
    cf = CFG[a_.data]
    dev = 'cuda'
    res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    for L in [int(v) for v in a_.L.split(',')]:
        t0 = time.time()
        aa = official_args((cf['common'] + ' ' + cf['a0'] + f' --model_id {cf["a0id"].format(L=L)} --seq_len {L} --des ExpA0s{SEED} --seed {SEED}').split())
        a0, _, order = load_a0(aa, cf['a0ck'].format(L=L), SEED)
        R = {'A1': {}, 'A0': {}}
        for sp in ('train', 'val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + MMAX * P, len(X) - H + 1))
            Y = cache_split(a0, dev, X, L, H, order)[0].numpy()
            if sp == 'test':                                                            # sanity: every test origin, as in the logs
                allo = list(range(L, len(X) - H + 1))
                fa1 = a1_resid(ex, X, L, allo, dev)[0]
                fa0 = np.stack([X[t:t + H] for t in allo]) - Y[np.asarray(allo) - L]
                print(f'[{a_.data} L{L} test] ALL {len(allo)} origins: A1 {(fa1 ** 2).mean():.6f} A0 {(fa0 ** 2).mean():.6f} | '
                      f'first {MMAX * P} origins only: A1 {(fa1[:MMAX * P] ** 2).mean():.4f} A0 {(fa0[:MMAX * P] ** 2).mean():.4f}', flush=True)
            R['A1'][sp] = a1_resid(ex, X, L, origins, dev)
            R['A0'][sp] = a0_resid(Y, X, L, origins)
            print(f'[{a_.data} L{L} {sp}] {len(origins)} origins | test-style MSE over kept origins: A1 {(R["A1"][sp][0] ** 2).mean():.4f} '
                  f'A0 {(R["A0"][sp][0] ** 2).mean():.4f}', flush=True)
        a0log = 'A0_W_H336_s3047.txt' if a_.data == 'W' and L == 720 else cf['a0log'].format(L=L)
        print(f'[{a_.data} L{L}] logged test MSE (all test windows): A1 {logged_test(cf["r1log"], L, True)} A0 {logged_test(a0log, L, False)} '
              f'| residuals {time.time() - t0:.0f}s', flush=True)
        for mdl in ('A1', 'A0'):
            o = analyse(R[mdl])
            res[f'{mdl}_L{L}'] = o
            print(f'===== {a_.data} {mdl} L={L} | future-residual MSE per lead: ' +
                  ' | '.join(f'{sp} ' + ' '.join(f'{v:.3f}' for v in o['mse'][sp]) for sp in o['mse']), flush=True)
            for sp in ('train', 'val', 'test'):
                show(f'R2adj within {sp}', o['R2adj'][sp])
            for sp in ('val', 'test'):
                show(f'R2cov train->{sp}', o['R2cov'][sp])
            for sp in ('val', 'test'):
                show(f'MSEred train->{sp}', o['MSEred'][sp])
        del a0; torch.cuda.empty_cache()
        json.dump(res, open(f'patchmem/logs/server/resid_cov_{a_.data}_s{SEED}.json', 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
