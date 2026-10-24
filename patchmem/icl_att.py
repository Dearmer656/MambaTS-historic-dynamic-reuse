"""PAT-307 offline probe: lag-specific softmax (kernel) attention memories over in-window errors + validated convex fusion
(the 'Multi-Lag In-Context Error Correction via Softmax Attention' proposal; no model training; frozen A1 checkpoints, seeds
3047/3048/3049). Residuals as resid_cov.py: chronological in-window 1-step errors e_0..e_{J-1} (J = L/P - 1), target = rollout
residual of lead h (position J-1+h).
Memory M_k (per window, per variable): keys rep(e_i), values the RAW error patch e_{i+k}, i = 0..J-1-k. A_k(q) = sum_i
softmax_i(s(q, rep(e_i))) e_{i+k}. Query for lead h through lag k (k >= h): rep(e_{J-1+h-k}) -- an observed error, never a
predicted one. rep: 'raw' = the 48-step patch standardised by the window's own (scalar) mean / std of that variable, or 'phi' =
(patch mean, LS slope) standardised per feature over the window's keys. s: 'dot' = q.k / (sqrt(d) tau) (the proposal's
exponential kernel), 'rbf' = -|q - k|^2 / (2 tau^2 d). tau in TAUS. (rep, s, tau) chosen on val per lag.
Estimators (lead 1): att1 = A_1(e_{J-1}); att2 = A_2(e_{J-2}); att_fuse = (1 - beta) att1 + beta att2, beta in BETAS chosen on
val (the proposal's scheme A, no double counting). Lead 2: att2 = A_2(e_{J-1}) (no fusion: only lag 2 among {1, 2} is valid).
Same with in-window ridge (icl_lag build/solve, linear, phi features, lam in LAMS chosen on val): rdg1, rdg2, rdg_fuse.
Baselines: mean (all J errors), recent8. Shuffle (sh_): key rows (attention) / demo inputs (ridge) permuted inside each window and
variable, values untouched (pairing broken, bias kept); NSH draws averaged; at the real estimator's config and beta.
All numbers are absolute test MSEred = 1 - SSE / sum r^2.
Pre-registered (ETTh2, A1, L = 1440, lead 1): multi-lag complementarity is supported iff att_fuse - att1 >= 0.01 on test for ALL
three seeds. L = 720 and Weather are secondary.
usage: python patchmem/icl_att.py --data E|W [--seeds 3047,3048,3049] [--L 1440,720] [--stride 1]"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.icl_lag import feats, build, solve, P
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)
BETAS = tuple(i / 10 for i in range(11))
LAMS = (0.3, 1.0, 3.0, 10.0)
NSH = 3


def reps(E):
    """E (B, C, J, P) chronological errors -> {'raw': (B, C, J, P), 'phi': (B, C, J, 2)} (phi not yet standardised)."""
    mu, sd = E.mean((2, 3), keepdim=True), E.std((2, 3), keepdim=True).clamp_min(1e-6)
    return {'raw': (E - mu) / sd, 'phi': feats(E)}


def attend(R, E, k, qpos, rep, kern, tau, perm=None):
    """Lag-k memory. R rep (B, C, J, d), E (B, C, J, P). Keys R[i], values E[i+k], i = 0..J-1-k; query R[qpos]. -> (B, C, P)."""
    J = E.shape[2]
    Kx, V, q = R[:, :, :J - k], E[:, :, k:], R[:, :, qpos]
    if rep == 'phi':
        mu, sd = Kx.mean(2, keepdim=True), Kx.std(2, keepdim=True).clamp_min(1e-6)
        Kx, q = (Kx - mu) / sd, (q - mu[:, :, 0]) / sd[:, :, 0]
    if perm is not None:
        Kx = torch.gather(Kx, 2, perm[:, :, :Kx.shape[2], None].expand_as(Kx))
    d = Kx.shape[-1]
    if kern == 'dot':
        s = (Kx @ q[..., None])[..., 0] / (math.sqrt(d) * tau)
    else:
        s = -((Kx - q[:, :, None]) ** 2).sum(-1) / (2 * tau * tau * d)
    return (torch.softmax(s, -1)[..., None] * V).sum(2)


def perm_for(keys, n):
    """Restriction of shared random keys (B, C, J) to the first n slots -> a permutation of 0..n-1, padded to J."""
    p = torch.argsort(keys[:, :, :n], -1)
    return torch.cat([p, torch.arange(n, keys.shape[2], device=keys.device).expand(*p.shape[:2], -1)], -1)


def ridge_lag(Ech, Fp, h, g, lam, keys=None):
    b = build(Ech, Fp, h, (g,), keys=keys)
    return solve(*b, lam)[0]


def draws(F, Hs, gen):
    """Yields, for the real run and NSH nulls, {(name, cfg): delta (N, C, P)}; then targets (r1, r2) and the baselines via .send-free
    attributes: returns (generator, r1, r2, base)."""
    Ft = torch.as_tensor(F, dtype=torch.float64, device=dev)
    Ech = torch.as_tensor(np.ascontiguousarray(Hs[:, ::-1]), dtype=torch.float64, device=dev)     # (N, J, P, C) chronological
    E = Ech.permute(0, 3, 1, 2)                                                                     # (N, C, J, P)
    N, C, J, _ = E.shape
    Fp = feats(Ech.permute(0, 1, 3, 2))
    R = reps(E)
    r1, r2 = Ft[:, :P].transpose(1, 2), Ft[:, P:2 * P].transpose(1, 2)
    base = {'mean': E.mean(2), 'recent8': E[:, :, -8:].mean(2)}
    def gen_draws():
        for keys in [None] + [torch.rand(N, C, J, generator=gen, device=dev, dtype=torch.float64) for _ in range(NSH)]:
            o = {}
            pa = lambda k: None if keys is None else perm_for(keys, J - k)
            for rep in ('raw', 'phi'):
                for kern in ('dot', 'rbf'):
                    for tau in TAUS:
                        cfg = f'{rep}_{kern}_t{tau}'
                        o[('att1', cfg)] = attend(R[rep], E, 1, J - 1, rep, kern, tau, pa(1))
                        o[('att2', cfg)] = attend(R[rep], E, 2, J - 2, rep, kern, tau, pa(2))
                        o[('att2_lead2', cfg)] = attend(R[rep], E, 2, J - 1, rep, kern, tau, pa(2))
            for lam in LAMS:
                o[('rdg1', f'l{lam}')] = ridge_lag(Ech, Fp, 1, 1, lam, keys)
                o[('rdg2', f'l{lam}')] = ridge_lag(Ech, Fp, 1, 2, lam, keys)
                o[('rdg2_lead2', f'l{lam}')] = ridge_lag(Ech, Fp, 2, 2, lam, keys)
            yield o
    return gen_draws(), r1, r2, base


def score(F, Hs, gen):
    """SSE tables: per draw, {(name, cfg): sse}; baselines {name: (sse1, sse2)}; ss (lead1, lead2)."""
    it, r1, r2, base = draws(F, Hs, gen)
    ss = ((r1 ** 2).sum().item(), (r2 ** 2).sum().item())
    S = []
    for o in it:
        s = {}
        for (nm, cfg), d in o.items():
            r = r2 if nm.endswith('lead2') else r1
            s[(nm, cfg)] = ((r - d) ** 2).sum().item()
        for fam in ('att', 'rdg'):
            c1s = [cfg for (nm, cfg) in o if nm == f'{fam}1']
            c2s = [cfg for (nm, cfg) in o if nm == f'{fam}2']
            for c1 in c1s:
                d1 = o[(f'{fam}1', c1)]
                for c2 in c2s:
                    diff = o[(f'{fam}2', c2)] - d1
                    e0 = r1 - d1
                    # |e0 - b diff|^2 = |e0|^2 - 2 b <e0, diff> + b^2 |diff|^2
                    a0, a1_, a2 = (e0 ** 2).sum().item(), (e0 * diff).sum().item(), (diff ** 2).sum().item()
                    for b in BETAS:
                        s[(f'{fam}_fuse', f'{c1}|{c2}|b{b}')] = a0 - 2 * b * a1_ + b * b * a2
        S.append(s)
        del o
    B = {nm: (((r1 - d) ** 2).sum().item(), ((r2 - d) ** 2).sum().item()) for nm, d in base.items()}
    return S, B, ss


NAMES = ('att1', 'att2', 'att_fuse', 'rdg1', 'rdg2', 'rdg_fuse', 'att2_lead2', 'rdg2_lead2')


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, H, split_x, a1_resid
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='1440,720'); ap.add_argument('--stride', type=int, default=1)
    a_ = ap.parse_args()
    cf = CFG[a_.data]; res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    for seed in [int(s) for s in a_.seeds.split(',')]:
        des, ck = cf['r1des'].replace('3047', str(seed)), cf['r1ck'].replace('s3047_0', f's{seed}_0')
        ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
        ex = Exp_R1(ar)
        ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
        gen = torch.Generator(device=dev).manual_seed(seed)
        for L in [int(v) for v in a_.L.split(',')]:
            J = L // P - 1
            sc = {}
            for sp in ('val', 'test'):
                X = split_x(ar, L, sp)
                origins = list(range(L + J * P, len(X) - H + 1))[::a_.stride]
                F, Hs = a1_resid(ex, X, L, origins, dev, mmax=J)
                sc[sp] = score(F, Hs, gen)
            (Sv, Bv, ssv), (St, Bt, sst) = sc['val'], sc['test']
            m = lambda sse, ss: 1 - sse / ss
            rows = {}
            for nm in NAMES:
                li = 1 if nm.endswith('lead2') else 0
                cands = [key for key in Sv[0] if key[0] == nm]
                best = max(cands, key=lambda key: m(Sv[0][key], ssv[li]))
                rows[nm] = dict(cfg=best[1], val=m(Sv[0][best], ssv[li]), test=m(St[0][best], sst[li]),
                                sh_test=float(np.mean([m(s[best], sst[li]) for s in St[1:]])))
            for nm in ('mean', 'recent8'):
                rows[nm] = dict(test_lead1=m(Bt[nm][0], sst[0]), test_lead2=m(Bt[nm][1], sst[1]))
            res[f's{seed}_L{L}'] = rows
            print(f'===== {a_.data} A1 s{seed} L={L} (J={J}) | absolute MSEred, config / beta chosen on val', flush=True)
            for nm in NAMES:
                rw = rows[nm]
                print(f'  {"lead2 " if nm.endswith("lead2") else "lead1 "}{nm:11s} test {rw["test"]:+.4f} sh {rw["sh_test"]:+.4f} val {rw["val"]:+.4f} | {rw["cfg"]}', flush=True)
            print(f'  lead1 mean {rows["mean"]["test_lead1"]:+.4f} recent8 {rows["recent8"]["test_lead1"]:+.4f} | lead2 mean {rows["mean"]["test_lead2"]:+.4f} recent8 {rows["recent8"]["test_lead2"]:+.4f}', flush=True)
            print(f'  criterion: att_fuse - att1 = {rows["att_fuse"]["test"] - rows["att1"]["test"]:+.4f} (rdg_fuse - rdg1 = {rows["rdg_fuse"]["test"] - rows["rdg1"]["test"]:+.4f})', flush=True)
            json.dump(res, open(f'patchmem/logs/server/icl_att_{a_.data}.json', 'w'))
        del ex; torch.cuda.empty_cache()
    print('DONE')


if __name__ == '__main__':
    main()
