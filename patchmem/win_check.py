"""PAT-307 task 6: firm up the lookback-limited row stream (win_row, patchmem/efc_cet.py stream_win) against controls and on stronger
base forecasters. Training-free, test split, official test origins, 3 seeds. All choices fixed 2026-10-25 before any result.

Main method win_row (unchanged, efc_cet.stream_win): feature F_t = E_(t-H)[1..H]; at origin t the ridge (lam 1, W H x H shared by the
variables) uses only origins i in [t-L+H, t-H] (targets inside the lookback); alpha = clip_[0,1] out-of-sample fit on the last NV = H
in-window origins with the map fitted on the other in-window origins (no EMA); correction alpha F_t W.
Controls (same window, same out-of-sample alpha, only the map from the window's pairs to the correction changes; stream_ctrl):
  bias        mean_i E_i per variable and lead (the "recent residual mean" baseline; no feature)
  persist     F_t (W = I: the error one horizon earlier carried over)
  scalar      rho F_t, rho = sum<F, E> / (lam + sum |F|^2) (one scalar decay, no cross-lead map)
  shuffle     W fitted on (F_pi(i), E_i), pi a fresh random permutation of the window's origins at every fit (pairing destroyed,
              statistics of F and E kept)
  ridge_mean  per-variable window means removed before the ridge, added back: mu_E + (F_t - mu_F) W (bias + transport)
Bases: a1 (plain R1, as efc_cet) | tefl (R1 T0-MS, trained with error feedback; ETTh2 random-H ckpt R1TfmsR64h0s, all H; Weather /
Traffic fixed-H96 ckpts R1{W,T}TfmsH96s, H = 96 only) | a0 (official MambaTS, one direct model per (L, H), ExpA0s ckpts; ETTh2 / Weather).
--arms attn (in-context attention check, fixed 2026-10-25): same window and alpha rule, but L2-normalised keys and a per-variable context;
  lin1 / krlin (linear kernel: 1 GD step / ridge) vs exp1 / soft1 / krexp (Gaussian kernel exp((cos - 1) / s2), s2 = median pairwise squared
  key distance: 1 GD step / softmax attention / kernel ridge), each with key row and ced. Kernel helps iff exp1 < lin1 and krexp < krlin;
  softmax normalisation helps iff soft1 < exp1 (3/3 seeds). Codex: s2 is a <=256-key approximation of the median (kept, documented).
usage: python patchmem/win_check.py --data E|W|T --base a1|tefl|a0 [--arms ctrl|attn] [--seeds ..] [--L ..] [--H ..] [--smoke]"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import patchmem.efc_cet as ec
dev = ec.dev
CTRLS = ('bias', 'persist', 'scalar', 'shuffle', 'ridge_mean')
ATTN = ('lin1', 'krlin', 'exp1', 'soft1', 'krexp')                                                # --arms attn (2026-10-25, fixed before results)
TEFL = {'E': ('R1TfmsR64h0s{s}', 7, None), 'W': ('R1WTfmsH96s{s}', 2, 96), 'T': ('R1TTfmsH96s{s}', 2, 96)}   # des, R1_HMAX, only H
A0ID = {'E': 'ETTh2', 'W': 'weather', 'T': 'traffic32'}


def ridge(Fm, Ym, Fq):
    """Fm, Ym (n, C, H), Fq (m, C, H) -> Fq W, W = (lam I + sum F^T F)^-1 sum F^T Y over all (origin, variable) rows."""
    H = Fm.shape[-1]
    S = torch.einsum('nch,ncg->hg', Fm, Fm).double(); T = torch.einsum('nch,ncg->hg', Fm, Ym).double()
    W = torch.linalg.solve(ec.LAM * torch.eye(H, device=Fm.device, dtype=torch.float64) + S, T).float()
    return Fq @ W


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _sigma2(K, nmax=256):
    """K (C, m, H) unit keys -> (C, 1, 1) median pairwise squared distance (median heuristic) over <= nmax evenly spaced keys."""
    C, m, _ = K.shape
    if m < 2:
        return torch.ones(C, 1, 1, device=K.device)
    sub = K[:, torch.linspace(0, m - 1, min(m, nmax), device=K.device).long()]
    d2 = (2 - 2 * sub @ sub.transpose(1, 2)).clamp_min(0)
    iu = torch.triu_indices(sub.shape[1], sub.shape[1], 1, device=K.device)
    return d2[:, iu[0], iu[1]].median(dim=1).values.clamp_min(1e-6).view(C, 1, 1)


def attn_predict(mode, Fm, Ym, Fq):
    """In-context maps on L2-normalised keys, context = the same variable's window pairs. Fm, Ym (m, C, H), Fq (q, C, H) -> (q, C, H).
    lin1  one normalised GD step (linear attention): mean_j cos(q, k_j) e_j
    krlin ridge on unit keys per variable (linear kernel, lam 1)
    exp1  one GD step in the Gaussian-kernel feature space: mean_j exp((cos(q, k_j) - 1) / s2) e_j (kernel vs lin1, no normalisation)
          s2 = median pairwise |k_i - k_j|^2 over <= 256 evenly spaced window keys (median heuristic, kernel exp(-d^2 / 2 s2))
    soft1 softmax attention: sum_j softmax_j(cos(q, k_j) / s2) e_j = exp1 normalised per query (normalisation vs exp1)
    krexp kernel ridge with the same Gaussian kernel exp((cos - 1) / s2), lam 1: k_q (K + lam I)^-1 E"""
    Kf, Yf, Kq = _unit(Fm).transpose(0, 1), Ym.transpose(0, 1), _unit(Fq).transpose(0, 1)       # (C, m, H), (C, m, H), (C, q, H)
    m, H = Kf.shape[1], Kf.shape[2]
    if mode == 'lin1':
        out = (Kq @ Kf.transpose(1, 2)) @ Yf / m
    elif mode == 'krlin':
        S = (Kf.transpose(1, 2) @ Kf).double(); T = (Kf.transpose(1, 2) @ Yf).double()
        out = Kq @ torch.linalg.solve(ec.LAM * torch.eye(H, device=Kf.device, dtype=torch.float64) + S, T).float()
    else:
        s2 = _sigma2(Kf)
        s2 = s2.double()
        A = (Kq @ Kf.transpose(1, 2)).double().sub(1) / s2                                         # (C, q, m), float64 before exp
        if mode == 'exp1':
            out = (torch.exp(A) @ Yf.double() / m).float()
        elif mode == 'soft1':
            out = (torch.softmax(A, dim=-1) @ Yf.double()).float()
        elif mode == 'krexp':
            Km = torch.exp((Kf @ Kf.transpose(1, 2)).double().sub(1) / s2) + ec.LAM * torch.eye(m, device=Kf.device, dtype=torch.float64)
            out = (torch.exp(A) @ torch.cholesky_solve(Yf.double(), torch.linalg.cholesky(Km))).float()
        else:
            raise ValueError(mode)
    return out.transpose(0, 1).contiguous()


def predict(mode, Fm, Ym, Fq, g):
    if mode in ATTN:
        return attn_predict(mode, Fm, Ym, Fq)
    if mode == 'ridge':
        return ridge(Fm, Ym, Fq)
    if mode == 'shuffle':
        return ridge(Fm[torch.randperm(Fm.shape[0], generator=g).to(Fm.device)], Ym, Fq)
    if mode == 'persist':
        return Fq.clone()
    if mode == 'scalar':
        return float((Fm * Ym).sum() / (ec.LAM + Fm.square().sum())) * Fq
    if mode == 'bias':
        return Ym.mean(0, keepdim=True).expand_as(Fq).clone()
    if mode == 'ridge_mean':
        mF, mY = Fm.mean(0, keepdim=True), Ym.mean(0, keepdim=True)
        return mY + ridge(Fm - mF, Ym - mY, Fq - mF)
    raise ValueError(mode)


@torch.no_grad()
def stream_ctrl(E, H, first, nv, feat, L, mode, seed=0):
    """Direct (per-origin) version of efc_cet.stream_win with the map chosen by mode; mode 'ridge' equals stream_win (tested)."""
    N, _, C = E.shape
    Eh = E[:, :H]
    assert not torch.isnan(Eh).any(), 'stream input must have full targets'
    D, valid = (ec.ced if feat == 'ced' else ec.row_feat)(E, H)
    F, Y = D.permute(0, 2, 1).float(), Eh.permute(0, 2, 1).float()
    g = torch.Generator().manual_seed(seed)
    corr = torch.zeros_like(Eh); active = torch.zeros(N, dtype=torch.bool, device=E.device)
    vcpu = valid.cpu()
    for t in range(first, N):
        lo, hi = max(0, t - L + H), t - H
        if hi < lo or not bool(vcpu[t]):
            continue
        idx = torch.arange(lo, hi + 1)[vcpu[lo:hi + 1]].to(E.device)
        if len(idx) < nv + 1:
            continue
        fi, vi = idx[:-nv], idx[-nv:]
        Q = predict(mode, F[fi], Y[fi], F[vi], g)
        al = float(((Q * Y[vi]).sum() / (Q.square().sum() + 1e-8)).clamp(0.0, 1.0))
        corr[t] = (al * predict(mode, F[idx], Y[idx], F[t:t + 1], g)[0]).T
        active[t] = True
    return corr, active


@torch.no_grad()
def a0_errors(data, L, H, seed, X, n0):
    """Errors (n0, H, C) of the official direct MambaTS (ExpA0) at origins L .. L+n0-1; NaN where the target is beyond X; None if no ckpt."""
    from patchmem.diag_rollout import official_args
    from patchmem.plug_a0 import load_a0, cache_split
    from patchmem.resid_cov import CFG
    pat = f'checkpoints/long_term_forecast_{A0ID[data]}_{L}_{H}_MambaTS_*_sl{L}_ll48_pl{H}_*_ExpA0s{seed}_0'
    hit = glob.glob(pat)
    if len(hit) != 1:
        return None, f'no unique A0 ckpt ({len(hit)}): {pat}'
    st = os.path.basename(hit[0])
    kv = dict(dm=re.search(r'_dm(\d+)_', st).group(1), el=re.search(r'_el(\d+)_', st).group(1), df=re.search(r'_df(\d+)_', st).group(1))
    common = re.sub(r'--(pred_len|d_model|d_ff) \S+', '', CFG[data]['common'])
    aa = official_args((common + ' ' + CFG[data].get('a0', '') + f' --model MambaTS --model_id {A0ID[data]}_{L}_{H} --seq_len {L} --pred_len {H} --d_model {kv["dm"]} '
                        f'--e_layers {kv["el"]} --d_ff {kv["df"]} --des ExpA0s{seed} --seed {seed}').split())
    m, d_, order = load_a0(aa, st, seed)
    Y = cache_split(m, d_, X, L, H, order)[0][:n0]                                       # (n0, H, C), origin L + k
    Xt = torch.as_tensor(X)
    T = len(X)
    y = torch.full_like(Y, float('nan'))
    for k in range(n0):
        t = L + k; j = min(H, T - t)
        y[k, :j] = Xt[t:t + j]
    del m; torch.cuda.empty_cache()
    return (y - Y).to(dev), st


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W', 'T'])
    ap.add_argument('--base', required=True, choices=['a1', 'tefl', 'a0'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='384,720,1440')
    ap.add_argument('--H', default='96,192,336'); ap.add_argument('--out', default='patchmem/logs/server/win_ck')
    ap.add_argument('--smoke', action='store_true', help='first seed, L 720, H 96, first 300 test origins')
    ap.add_argument('--arms', default='ctrl', choices=['ctrl', 'attn'])
    a_ = ap.parse_args()
    if a_.smoke:
        a_.seeds, a_.L, a_.H = a_.seeds.split(',')[0], '720', '96'
    Hs = [int(v) for v in a_.H.split(',')]
    if a_.base == 'tefl' and TEFL[a_.data][2]:
        Hs = [h for h in Hs if h == TEFL[a_.data][2]]
    cf = CFG[a_.data]
    for k in [k for k in os.environ if k.startswith(('R1_TEFL', 'R1_ICL', 'R1_OBS', 'R1_MEM', 'R1_PATHRES', 'R1_HMAX'))]:
        os.environ.pop(k)
    if a_.base == 'tefl':
        os.environ['R1_TEFL'] = 'ms'; os.environ['R1_HMAX'] = str(TEFL[a_.data][1])
    from patchmem.exp_r1 import Exp_R1
    os.makedirs(a_.out, exist_ok=True)
    import hashlib, subprocess
    md5 = hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:10]; md5e = hashlib.md5(open(ec.__file__, 'rb').read()).hexdigest()[:10]
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print(f'[version] win_check.py md5 {md5} | efc_cet.py md5 {md5e} | HEAD {head} | data {a_.data} base {a_.base}', flush=True)
    print(f'win_row: lam {ec.LAM} NV = H, window [t-L+H, t-H], alpha out of sample without EMA | controls {",".join(CTRLS)} | '
          f'stream starts {ec.WARM} steps before the first test origin', flush=True)
    res = {}
    for seed in [int(s) for s in a_.seeds.split(',')]:
        ex = None
        if a_.base != 'a0':
            des = ec.A1[a_.data].format(s=seed) if a_.base == 'a1' else TEFL[a_.data][0].format(s=seed)
            ck = cf['r1ck'].replace(cf['r1des'], des)
            ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
            ex = Exp_R1(ar)
            ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
            ex.model.eval(); ex._fix_eval_order()
            print(f'[ckpt] {ck} | {type(ex.model).__name__}', flush=True)
        else:
            ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des x --seed {seed}').split())
        for L in [int(v) for v in a_.L.split(',')]:
            X = split_x(ar, L + ec.WARM, 'test')
            n0 = len(X) - L - min(Hs) + 1
            E_all = ec.rollout_errors(ex.model, X, L, n0) if ex is not None else None
            for H in Hs:
                t0 = time.time()
                if E_all is not None:
                    E, src = E_all, 'rollout'
                else:
                    E, src = a0_errors(a_.data, L, H, seed, X, n0)
                    if E is None:
                        print(f'  s{seed} L={L:4d} H={H:3d} | skip: {src}', flush=True); continue
                first, last = ec.WARM, len(X) - H - L
                if a_.smoke:
                    last = first + 299
                sl = slice(first, last + 1)
                y_err = E[sl, :H]
                assert not torch.isnan(y_err).any(), 'official windows must have full targets'
                base = y_err.square().mean().item()
                Es = E[:last + 1]
                row = dict(n=last - first + 1, base=base)
                c, act = ec.stream(Es, H, first, H, True, 'row', L)
                row['win_row'] = (y_err - c[sl]).square().mean().item(); row['win_row_active'] = act[sl].float().mean().item()
                if a_.arms == 'attn':
                    arms = [f'{md}_{ft}' for ft in ('row', 'ced') for md in ATTN]
                    for ft in ('row', 'ced'):
                        for md in ATTN:
                            c, act = stream_ctrl(Es, H, first, H, ft, L, md, seed)
                            row[f'{md}_{ft}'] = (y_err - c[sl]).square().mean().item()
                    res[f's{seed}|{L}|{H}'] = row
                    print(f'  s{seed} L={L:4d} H={H:3d} | base {base:.4f} ({src if src != "rollout" else a_.base}) | win_row '
                          f'{(row["win_row"] / base - 1) * 100:+.1f}% | ' + ' | '.join(f'{k} {(row[k] / base - 1) * 100:+.1f}%' for k in arms)
                          + f' | n {row["n"]} | {time.time() - t0:.0f}s', flush=True)
                    continue
                c, _ = ec.stream(Es, H, first, H, False, 'row', L)
                row['win_row_a1'] = (y_err - c[sl]).square().mean().item()
                for md in CTRLS:
                    c, act = stream_ctrl(Es, H, first, H, 'row', L, md, seed)
                    row[md] = (y_err - c[sl]).square().mean().item()
                res[f's{seed}|{L}|{H}'] = row
                print(f'  s{seed} L={L:4d} H={H:3d} | base {base:.4f} ({src if src != "rollout" else a_.base}) | win_row {row["win_row"]:.4f} '
                      f'({(row["win_row"] / base - 1) * 100:+.1f}%, active {row["win_row_active"]:.2f}) | alpha=1 '
                      f'{(row["win_row_a1"] / base - 1) * 100:+.1f}% | ' +
                      ' | '.join(f'{md} {(row[md] / base - 1) * 100:+.1f}%' for md in CTRLS) + f' | n {row["n"]} | {time.time() - t0:.0f}s', flush=True)
            tag = 'win_ck' if a_.arms == 'ctrl' else 'win_attn'
            json.dump(res, open(f'{a_.out}/{tag}_{a_.data}_{a_.base}{"_smoke" if a_.smoke else ""}.json', 'w'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
