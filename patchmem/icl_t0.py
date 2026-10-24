"""PAT-307 offline probe: does in-window ICL still help ON TOP OF the best trained arm, T0r (ETTh2: R1 + TEFL native, Hc = 2) /
TF0 (Weather: Hc = 0)? No training; frozen checkpoints, seeds 3047/3048/3049.
T0r's forecast at origin T: lead k = backbone rollout + sigma * g(eps)[k], eps = the window's last Hc RevIN-normalised one-step
backbone errors (never corrected), g = W2 ReLU W1 (patchmem/r1_tefl.py). Its own residual r_k = y - pred.
In-window demonstrations of T0r's lead-1 residual: for an in-window origin = patch p, T0r would predict patch p as
yh_{p-1} + sigma * g(eps_p)[0], eps_p = the errors of the Hc patches before p (Hc = 0: all patches 1..p-1, right-aligned,
zero-padded, as in R1Tefl.error). So rho_p = e_p - sigma * g(eps_p)[0] (data / scaler units; e_p = x_p - hist_p) is EXACTLY what T0r
leaves at lead 1 from origin p (the same teacher-forced backbone read; only RevIN uses the whole window's statistics, as for A1).
The g for origin T is checked against the model's own eps (ret_eps) on every batch.
Probes per lead k on the chronological series rho (and, 'xe', inputs from the backbone errors e at the same positions):
  rdg      lag-k in-window ridge (icl_lag build/solve, phi features, linear), lam in LAMS; adj2 = G {k, k+1}
  att      lag-k softmax attention (icl_att.attend, query = the most recent patch), rep / kernel / tau grid
  recent8, mean   in-window bias of rho
  best     the best of all the above on val (per lead)
Shuffle (sh_): demo inputs / keys permuted (icl_lag v3 null), NSH draws averaged, at the real config. Absolute test MSEred
relative to T0r's own forecast (1 - SSE / sum r^2).
Pre-registered (ETTh2, L = 1440, lead 1): ICL adds to T0r iff best >= 0.05 AND best - sh_best >= 0.03 on test for ALL three seeds.
Leads 2-7, L = 720 and Weather are secondary.
usage: python patchmem/icl_t0.py --data E|W [--seeds 3047,3048,3049] [--L 1440,720] [--stride 1]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.icl_lag import feats, build, solve, P, NP
from patchmem.icl_att import attend, reps, perm_for, TAUS
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
LAMS = (0.3, 1.0, 3.0, 10.0)
NSH = 3
ARM = {'E': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64'), des='R1TfnH2R64h0s{s}'),
       'W': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64'), des='R1WTfnH0R64h0s{s}')}


def g_lead1(m, en_prev):
    """en_prev (b, n, P, C) the normalised errors of the patches before the origin (chronological). -> sigma-free lead-1 g (b, P, C)."""
    b, n, _, C = en_prev.shape
    e = en_prev.permute(0, 1, 3, 2)                                                     # (b, n, C, P)
    if m.hc:
        e = e[:, -m.hc:]
    else:
        e = torch.cat([e.new_zeros(b, m.tmax - 1 - n, C, P), e], 1)
    eps = e.permute(0, 2, 1, 3).reshape(b, C, -1)
    return m.W2(torch.relu(m.W1(eps))).reshape(b, C, m.hmax, P)[:, :, 0].transpose(1, 2), eps


@torch.no_grad()
def t0_resid(ex, X, L, origins, bs=128):
    """-> F (N, H, C) T0r rollout residual, rho (N, J', P, C) chronological in-window T0r lead-1 residuals, e (N, J', P, C) backbone
    errors at the same patches."""
    m = ex.model; m.eval(); ex._fix_eval_order()
    T = L // P
    p0 = m.hc + 1 if m.hc else 2                                                        # first patch with a full eps
    Xt = torch.as_tensor(X)
    Fs, Rs, Es = [], [], []
    for i0 in range(0, len(origins), bs):
        ts = origins[i0:i0 + bs]
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        out = m(bx, NP, ret_eps=True)
        y = torch.stack([Xt[t:t + NP * P] for t in ts]).to(dev)
        Fs.append((y - out['pred']).cpu())
        eh = bx[:, P:].reshape(len(ts), T - 1, P, -1) - out['hist']                    # patches 1..T-1, data units
        sd = m.revin(bx)[1][:, 0]                                                       # (b, C)
        en = eh / sd[:, None, None]
        _, epsT = g_lead1(m, en)                                                        # origin T: all patches 1..T-1 before it
        assert torch.allclose(epsT, out['eps'], atol=1e-4), (epsT - out['eps']).abs().max()
        rho = []
        for p in range(p0, T):                                                          # in-window origin = patch p (index p-1 in eh)
            gl, _ = g_lead1(m, en[:, :p - 1])
            rho.append(eh[:, p - 1] - gl * sd[:, None])
        Rs.append(torch.stack(rho, 1).cpu()); Es.append(eh[:, p0 - 1:].cpu())
    return torch.cat(Fs).numpy(), torch.cat(Rs).numpy(), torch.cat(Es).numpy()


def probe(F, Rho, Eb, gen):
    """-> {(name, cfg): [sse per lead]} for the real run and NSH nulls, baselines, ss per lead."""
    Ft = torch.as_tensor(F, dtype=torch.float64, device=dev)
    Ech = torch.as_tensor(Rho, dtype=torch.float64, device=dev)                         # (N, J, P, C) chronological
    Eb_ = torch.as_tensor(Eb, dtype=torch.float64, device=dev)
    N, J, _, C = Ech.shape
    Fp = {'rho': feats(Ech.permute(0, 1, 3, 2)), 'xe': feats(Eb_.permute(0, 1, 3, 2))}
    E = Ech.permute(0, 3, 1, 2)                                                         # (N, C, J, P)
    R = reps(E)
    rs = [Ft[:, k * P:(k + 1) * P].transpose(1, 2) for k in range(NP)]
    ss = [(r ** 2).sum().item() for r in rs]
    out = []
    for keys in [None] + [torch.rand(N, C, J, generator=gen, device=dev, dtype=torch.float64) for _ in range(NSH)]:
        s = {}
        for k in range(1, NP + 1):
            r = rs[k - 1]
            add = lambda key, d: s.setdefault(key, [np.nan] * NP).__setitem__(k - 1, ((r - d) ** 2).sum().item())
            for src in ('rho', 'xe'):
                for lam in LAMS:
                    for nm, G in (('rdg', (k,)), ('adj2', (k, k + 1))):
                        b = build(Ech, Fp[src], k, G, keys=keys)
                        if b is not None:
                            add((f'{nm}_{src}', f'l{lam}'), solve(*b, lam)[0])
            for rep in ('raw', 'phi'):
                for kern in ('dot', 'rbf'):
                    for tau in TAUS:
                        add(('att', f'{rep}_{kern}_t{tau}'), attend(R[rep], E, k, J - 1, rep, kern, tau, None if keys is None else perm_for(keys, J - k)))
            add(('recent8', '-'), E[:, :, -8:].mean(2)); add(('mean', '-'), E.mean(2))
        out.append(s)
    return out, ss


FAMS = ('rdg_rho', 'adj2_rho', 'rdg_xe', 'adj2_xe', 'att', 'recent8', 'mean')


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, H, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='1440,720'); ap.add_argument('--stride', type=int, default=1)
    a_ = ap.parse_args()
    cf, arm = CFG[a_.data], ARM[a_.data]; res = {}
    os.environ.pop('R1_OBS', None); os.environ.update(arm['env'])
    from patchmem.exp_r1 import Exp_R1
    for seed in [int(s) for s in a_.seeds.split(',')]:
        des = arm['des'].format(s=seed)
        ck = cf['r1ck'].replace(cf['r1des'], des)
        assert os.path.exists(f'checkpoints/{ck}/checkpoint.pth'), ck
        ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
        ex = Exp_R1(ar)
        ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
        print(f'[ckpt] {ck} | model {type(ex.model).__name__} hc={ex.model.hc}', flush=True)
        gen = torch.Generator(device=dev).manual_seed(seed)
        for L in [int(v) for v in a_.L.split(',')]:
            J = L // P - 1
            sc = {}
            for sp in ('val', 'test'):
                X = split_x(ar, L, sp)
                origins = list(range(L + J * P, len(X) - H + 1))[::a_.stride]
                F, Rho, Eb = t0_resid(ex, X, L, origins)
                if sp == 'test':
                    print(f'  [sanity] T0r kept-origin test MSE per lead: ' + ' '.join(f'{(F[:, k * P:(k + 1) * P] ** 2).mean():.4f}' for k in range(NP)), flush=True)
                sc[sp] = probe(F, Rho, Eb, gen)
            (Sv, ssv), (St, sst) = sc['val'], sc['test']
            m = lambda v, ss: [1 - a / b for a, b in zip(v, ss)]
            rows = {}
            for k in range(NP):
                for nm in FAMS + ('best',):
                    keys = [key for key in Sv[0] if (key[0] == nm or nm == 'best') and not np.isnan(Sv[0][key][k])]
                    if not keys:
                        continue
                    b = max(keys, key=lambda key: 1 - Sv[0][key][k] / ssv[k])
                    rows.setdefault(nm, dict(test=[np.nan] * NP, sh=[np.nan] * NP, cfg=[None] * NP))
                    rows[nm]['test'][k] = 1 - St[0][b][k] / sst[k]
                    rows[nm]['sh'][k] = float(np.mean([1 - s[b][k] / sst[k] for s in St[1:]]))
                    rows[nm]['cfg'][k] = f'{b[0]}:{b[1]}'
            res[f's{seed}_L{L}'] = rows
            print(f'===== {a_.data} T0 s{seed} L={L} (J\'={Rho.shape[1]}) | absolute test MSEred relative to T0r own forecast | real / shuffle', flush=True)
            f = lambda v: ' '.join('   -  ' if np.isnan(x) else f'{x:+.3f}' for x in v)
            for nm, rw in rows.items():
                print(f'  {nm:9s} {f(rw["test"])} | sh {f(rw["sh"])}' + (f' | {rw["cfg"][:3]}' if nm == 'best' else ''), flush=True)
            b = rows['best']
            print(f'  criterion lead1: best {b["test"][0]:+.4f} (>= 0.05) | best - sh {b["test"][0] - b["sh"][0]:+.4f} (>= 0.03)', flush=True)
            json.dump(res, open(f'patchmem/logs/server/icl_t0_{a_.data}.json', 'w'))
        del ex; torch.cuda.empty_cache()
    print('DONE')


if __name__ == '__main__':
    main()
