"""PAT-315: is there periodic shock information beyond the 96 shock lags of base_sh?  (No training: isolates the data
from the carry's optimisation -- truncated BPTT of one day and Adam on theta could not learn daily / weekly lags.)

Fixed rotary state of the shock series (the s1_R carry with theta, rho frozen at their init), per channel and mode m < 64:
  z_j = rho_m e^(i theta_m) z_(j-1) + (1 - rho_m) s_j,   theta_m = 2 pi m / (PW q) (weekly harmonics),
  rho_m = exp(-1 / tau_m), tau log-spaced 4 h .. 2 weeks      (s_j = mean of the observation diagonal of time j)
Feature at origin t: z_(t-1) (Re, Im; 128), observed by t - 1.  Same ridge machinery as diag_shock_ridge (iTransformer L96
H192, pooled channels, fit on train, lam on val, test once), target E[t, :].  All arms on the same origins (state warm-up
of 3 x 2 weeks + H).  Arms: sh (96 shock lags = base_sh), rot, sh+rot, and sh+tail = sh + z_(t-97) (rotary state of the
shocks older than the 96 lags only; interpretation arm, not part of the criterion).
Criterion (fixed before the runs): sh+rot beats sh by >= 0.3 pp test gain, 3/3 seeds, >= 4/6 datasets.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, L
from patchmem.eval_sr import D_MODES, PW, Q

LS = 96
ND = 2 * D_MODES
BL = {'sh': (0, LS), 'rot': (LS, LS + ND), 'tail': (LS + ND, LS + 2 * ND)}
ARMS = {'sh': ('sh',), 'rot': ('rot',), 'sh+rot': ('sh', 'rot'), 'sh+tail': ('sh', 'tail')}
NF = LS + 2 * ND + 1


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    X, M, n_tr, n_va = ev.load_series(exp, code)
    o0 = L
    E = ev.stream_errors(exp, X, M, o0, args.bs)                                     # n H C
    n, C = E.shape[0], E.shape[2]
    ssum = torch.zeros(n + H - 1, C)
    cnt = torch.zeros(n + H - 1)
    for l in range(H):
        ssum[l:l + n] += E[:, l]
        cnt[l:l + n] += 1
    shock = ssum / cnt[:, None]                                                      # index tau - o0
    del ssum

    q = Q[code]
    th = 2 * math.pi * torch.arange(D_MODES, dtype=torch.float64) / (PW * q)
    tau = q * torch.logspace(math.log10(4), math.log10(2 * PW), D_MODES, dtype=torch.float64)
    rho = torch.exp(-1 / tau)
    lam_c = (rho * torch.exp(1j * th)).to(torch.complex128).to(dev)
    gain_in = (1 - rho).to(dev)
    Z = torch.zeros(n, C, ND)                                                        # Z[k] = z_(k-1): features of origin k
    z = torch.zeros(C, D_MODES, dtype=torch.complex128, device=dev)
    sd = shock.to(dev).double()
    for j in range(n - 1):
        z = lam_c * z + gain_in * sd[j][:, None]
        Z[j + 1] = torch.cat([z.real, z.imag], -1).float().cpu()
    del sd

    def feats(t):
        k = t - o0
        jl = torch.arange(LS)
        sh = shock[k[:, None] - 1 - jl].permute(0, 2, 1)                             # B C LS
        Xf = torch.cat([sh, Z[k], Z[k - LS], torch.ones(len(k), C, 1)], -1).to(dev).double()
        return Xf, E[k].permute(0, 2, 1).to(dev).double()

    def moments(ts):
        xx = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
        xy = torch.zeros(NF, H, dtype=torch.float64, device=dev)
        yy = torch.zeros(H, dtype=torch.float64, device=dev)
        m = 0
        for i in range(0, len(ts), args.chunk):
            Xf, Y = feats(ts[i:i + args.chunk])
            Xf, Y = Xf.reshape(-1, NF), Y.reshape(-1, H)
            xx += Xf.T @ Xf
            xy += Xf.T @ Y
            yy += Y.square().sum(0)
            m += len(Y)
        return dict(xx=xx, xy=xy, yy=yy, n=m)

    first = o0 + max(LS, 3 * 2 * PW * q) + H                                         # full lags + 3 x the slowest tau
    t_tr = torch.arange(first, n_tr - H + 1)
    t_va = torch.arange(n_tr, n_tr + n_va - H + 1)
    t_te = torch.arange(n_tr + n_va, len(X) - H + 1)
    assert len(t_tr) > 0
    mo = {k: moments(ts) for k, ts in (('train', t_tr), ('val', t_va), ('test', t_te))}
    base = float(mo['test']['yy'].sum()) / (mo['test']['n'] * H)
    assert abs(base - official) < 1e-4, (base, official)
    res = {}
    for arm, bl in ARMS.items():
        c = torch.cat([torch.arange(*BL[b]) for b in bl] + [torch.tensor([NF - 1])]).to(dev)
        val = {lam: ev.mse(mo['val'], c, ev.solve(mo['train'], c, lam)) for lam in ev.LAMS}
        lam = min(val, key=val.get)
        te = ev.mse(mo['test'], c, ev.solve(mo['train'], c, lam))
        res[arm] = dict(lam=lam, val_mse=val[lam], test_mse=te, gain=100 * (base - te) / base)
        print(f'[test] {arm:7s} lam {lam:g} gain {res[arm]["gain"]:+.2f}%', flush=True)
    delta = res['sh+rot']['gain'] - res['sh']['gain']
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f'{code}_rotridge_s{seed}.json'
    p.write_text(json.dumps(dict(dataset=code, seed=seed, H=H, base=base, n_train=len(t_tr), q=q, arms=res,
                                 delta_pp=delta, pass_=delta >= 0.3), indent=2) + '\n')
    print(f'[done] {p} delta {delta:+.2f} pp', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=tuple(Q))
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_rot_diag')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
