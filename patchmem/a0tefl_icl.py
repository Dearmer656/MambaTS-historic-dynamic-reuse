"""PAT-313: in-context (per-case) update of a trained A0-TEFL's W1 / W2 (patchmem/exp_a0tefl.py, TEFL original setting) from demonstrations
taken from the lookback history; full grid over the total lookback length N and the demo stride D (training-free).
Errors (TEFL's own definition, the jointly trained f): E[o] = x[o:o+H] - f(x[o-L:o]) for every origin o (one forward per origin, cached).
Test case with forecast origin t (same test windows as the official test): TEFL = f(x[t-L:t]) + W2 ReLU(W1 E[t-H]).
Demos s_k = t - H - k*D (k = 0..n-1, the newest first): key E[s_k - H], value E[s_k] (observed: s_k + H <= t), inside the lookback of
length N: s_k - H - L >= t - N, i.e. n = floor((N - L - 2H) / D) + 1. Update = the two-layer step of r1_tefl_icl.two_layer, per variable:
  W1' = W1 + a1 * sum_j g_j k_j^T / sum_j |k_j|^2,  g_j = (W2^T r_j) * 1[W1 k_j > 0],  r_j = v_j - W2 ReLU(W1 k_j)
  out = W2 h'_q + a2 * sum_j <h'_j, h'_q> (v_j - W2 h'_j) / (|h'_q| sum_j |h'_j|),  h' = ReLU(W1' .)
a1 = 0: W2 only; a2 = 0: W1 only. (a1, a2) chosen per (N, D) and family on the first half of the test windows (MSE), reported on the second
half; TEFL and A0-part (f alone) on the same halves. Sanity: TEFL over all windows must equal the official test MSE of the checkpoint.
usage: python patchmem/a0tefl_icl.py --data E --H 192 [--seed 3047 --L 384] -> patchmem/logs/server/a0tefl_icl/<D>_H<H>_s<seed>.pt"""
import argparse, glob, itertools, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args
from data_provider.data_factory import data_provider
from utils.tools import set_seed

DS = {'E': ('ETTh2', '--root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --data ETTh2 --n_heads 16 --dropout 0.3'),
      'W': ('weather', '--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --data custom --n_heads 16 --d_model 32 --d_ff 512'),
      'T': ('traffic32', '--root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 --enc_in 32 --dec_in 32 --c_out 32 --data custom --n_heads 16 --d_model 32 --d_ff 512 --e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005')}
HX = {('E', 96): '--e_layers 5 --d_model 16 --batch_size 8 --learning_rate 0.001', ('E', 192): '--e_layers 3 --d_model 16 --batch_size 16 --learning_rate 0.001',
      ('W', 96): '--e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005', ('W', 192): '--e_layers 2 --dropout 0.3 --batch_size 16 --learning_rate 0.001',
      ('T', 96): '', ('T', 192): ''}
N_GRID = [960, 1152, 1440, 1920, 2400, 2880, 3840]                                       # plus the minimum L + 2H (one demo)
D_GRID = [48, 96, 144, 192, 288, 384]
A1 = [0.0, 0.03, 0.1, 0.3, 1.0]
A2 = [0.0, 0.1, 0.3, 1.0, 3.0]


def two_layer(q, K, V, W1, W2, a1, a2):
    """q (B, C, H), K / V (B, C, n, H); W1 (r, H), W2 (H, r). Returns the corrected TEFL output (B, C, H)."""
    pre_k = K @ W1.T
    r_ = V - torch.relu(pre_k) @ W2.T
    g = (r_ @ W2) * (pre_k > 0).to(K.dtype)
    Z1 = K.square().sum((-1, -2)).clamp_min(1e-12)[..., None, None]
    pre_k2 = pre_k + a1 * (K @ K.transpose(-1, -2)) @ g / Z1
    pre_q2 = q @ W1.T + a1 * ((q[:, :, None, :] @ K.transpose(-1, -2)) @ g).squeeze(2) / Z1[..., 0]
    hk, hq = torch.relu(pre_k2), torch.relu(pre_q2)
    out = hq @ W2.T
    if a2:
        r2 = V - hk @ W2.T
        s_ = (hk * hq[:, :, None]).sum(-1)
        den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
        out = out + a2 * (s_[..., None] * r2).sum(-2) / den[..., None]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--H', type=int, required=True)
    ap.add_argument('--seed', type=int, default=3047); ap.add_argument('--L', type=int, default=384); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--chunk', type=int, default=512)
    a_ = ap.parse_args()
    D, H, L, s = a_.data, a_.H, a_.L, a_.seed
    name, dargs = DS[D]
    Ns = sorted({L + 2 * H} | {n for n in N_GRID if n > L + 2 * H})
    Nmax = Ns[-1]
    args = official_args((f'--task_name long_term_forecast --is_training 0 --model MambaTS --features M --seq_len {L} --label_len 48 --pred_len {H} '
                          f'--d_layers 2 --factor 1 --itr 1 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --seed {s} '
                          f'--model_id {name}_{L}_{H} {dargs} {HX[(D, H)]} --des A0Tefls{s}').split())
    from patchmem.exp_a0tefl import Exp_A0Tefl
    ex = Exp_A0Tefl(args)                                                                 # builds A0Tefl(f with L), sets args.seq_len = L + H
    m = ex.model
    ck = glob.glob(f'checkpoints/long_term_forecast_{name}_{L}_{H}_MambaTS_*_sl{L + H}_*_A0Tefls{s}_0/checkpoint.pth')
    assert len(ck) == 1, ck
    m.load_state_dict(torch.load(ck[0], map_location=ex.device)); m.eval()
    dev = ex.device
    args.seq_len = Nmax                                                                   # the same test windows, longer history in front
    ds, _ = data_provider(args, 'test')
    X = torch.tensor(ds.data_x, dtype=torch.float32)                                      # (T_all, C) scaled
    n_test = len(ds)
    T_all, C = X.shape
    t_all = torch.arange(n_test) + Nmax                                                   # forecast origins of the test windows
    assert int(t_all[-1]) + H == T_all, (int(t_all[-1]), H, T_all)
    set_seed(s)                                                                           # as run.py before the official test (VPT order)
    o_all = torch.arange(L, T_all - H + 1)                                                # every origin with an observed target
    F = torch.empty(len(o_all), H, C)
    with torch.no_grad():
        for i in range(0, len(o_all), a_.bs):
            o = o_all[i:i + a_.bs]
            F[i:i + len(o)] = m.base(X[o[:, None] + torch.arange(-L, 0)[None]].to(dev), None, None, None).cpu()
    E = X[o_all[:, None] + torch.arange(H)[None]] - F                                     # (O, H, C), E[o - L]
    Eo = lambda o: E[o - L]
    W1, W2 = m.W1.weight.detach().cpu().double(), m.W2.weight.detach().cpu().double()
    tgt = Eo(t_all).double().permute(0, 2, 1)                                             # (B, C, H) = f's error at t = what TEFL predicts
    q = Eo(t_all - H).double().permute(0, 2, 1)
    tefl = torch.relu(q @ W1.T) @ W2.T
    se = lambda pred: (tgt - pred).square().mean((1, 2))                                  # per-window MSE of f + correction
    B = n_test; h1, h2 = slice(0, B // 2), slice(B // 2, B)
    base_se, tefl_se = se(torch.zeros_like(tgt)), se(tefl)
    print(f'[a0tefl_icl] {D} H{H} s{s} L{L} | windows {B} | f alone all {base_se.mean():.6f} | TEFL all {tefl_se.mean():.6f} (official test of the '
          f'checkpoint must match) | half2: f {base_se[h2].mean():.4f} TEFL {tefl_se[h2].mean():.4f}', flush=True)
    res = dict(f=base_se, tefl=tefl_se, grid={})
    for N, Dd in itertools.product(Ns, D_GRID):
        n = (N - L - 2 * H) // Dd + 1
        if n < 1 or (N == L + 2 * H and Dd != D_GRID[0]):
            continue
        pairs = [(a1, a2) for a1, a2 in itertools.product(A1, A2) if a1 or a2]
        cell = {k: torch.empty(B, dtype=torch.float64) for k in pairs}
        W1d, W2d = W1.to(dev), W2.to(dev)
        for i in range(0, B, a_.chunk):                                                   # windows in chunks (memory)
            tb = t_all[i:i + a_.chunk]
            sk = tb[:, None] - H - Dd * torch.arange(n)[None]                             # (b, n), newest first
            K = Eo(sk - H).double().permute(0, 3, 1, 2).to(dev)                           # (b, C, n, H)
            V = Eo(sk).double().permute(0, 3, 1, 2).to(dev)
            qb, yb = q[i:i + a_.chunk].to(dev), tgt[i:i + a_.chunk].to(dev)
            for k in pairs:
                cell[k][i:i + len(tb)] = (yb - two_layer(qb, K, V, W1d, W2d, *k)).square().mean((1, 2)).cpu()
        row = dict(n=n, cell=cell)
        for fam, ok in (('W1+W2', lambda k: k[0] > 0 and k[1] > 0), ('W2', lambda k: k[0] == 0), ('W1', lambda k: k[1] == 0)):
            ks = [k for k in cell if ok(k)]
            best = min(ks, key=lambda k: cell[k][h1].mean())
            row[fam] = (best, float(cell[best][h2].mean()))
        res['grid'][(N, Dd)] = row
        print(f'[a0tefl_icl] {D} H{H} N {N:4d} D {Dd:3d} n {n:3d} | half2 vs TEFL: ' + ' | '.join(
            f'{fam} {100 * (row[fam][1] / float(tefl_se[h2].mean()) - 1):+.2f}% (a {row[fam][0]})' for fam in ('W1+W2', 'W2', 'W1')), flush=True)
    os.makedirs('patchmem/logs/server/a0tefl_icl', exist_ok=True)
    torch.save(res, f'patchmem/logs/server/a0tefl_icl/{D}_H{H}_s{s}.pt')


if __name__ == '__main__':
    main()
