"""PAT-307 offline probe: in-window ICL experts + a CROSS-WINDOW memory of which lag (hop) has been paying off (no model training;
A1 / A0 seed-3047 checkpoints, frozen). Follows patchmem/icl_lag.py (v3): fixed single lags g ~ 15-17 carry large pairing gains at
leads 4-7 on ETTh2 test, but which lag works drifts (val 17-22 vs test 15-17) and the per-window LOO is too noisy to find it.
Experts per lead k (each recomputed inside every window, i.e. the error -> correction map is learnt in context):
  zero; mean (all J in-window errors); recent{4,8,12} (mean of the n most recent errors);
  single g (G = {g}, g = k..J-NMIN) and pair g (G = {k, g}, g = k+1..J-NMIN): icl_lag.build/solve, linear, lam = 1.
Memory (per lead k): after the lead-k outcome of a past origin is REVEALED (origin + k*P steps, i.e. ceil(k*P / stride) windows
later), each expert's loss on it (MSE over the patch) enters a decayed sum Lg <- gamma * Lg + loss_g (gamma from a half-life in
windows); weights w_g ~ exp(-eta * (Lg / L_zero - min_g' Lg' / L_zero)) (eta = inf: follow the leader); share = per variable or
pooled over variables. The window then predicts delta_k = sum_g w_g delta_{k,g}. Test continues from the memory at the end of val
(val precedes test, all val outcomes are revealed before the first test origin); val starts from uniform weights.
(half-life, eta, share) chosen on val per lead. The future target of the current origin is never used.
Shuffle control (sh_): the same, with every expert's demo INPUTS permuted (icl_lag v3 null: outputs / bias / mean / recent kept),
NSH draws averaged. Context rows: the val-chosen fixed lag, and the test-hindsight best single lag (optimistic, not an estimator).
Pre-registered (ETTh2, A1, L = 1440, test, mean over leads 4-7):
  hedge - sh_hedge >= 0.02   AND   hedge - stack_lag(icl_lag v3: +0.072) >= 0.02.   Weather and L = 720 secondary.
usage: python patchmem/icl_hedge.py --data E|W [--L 1440,720] [--stride 1]"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import patchmem.icl_lag as IL
from patchmem.icl_lag import feats, build, solve, P, NP, SEED, NMIN

RECENT = (4, 8, 12)
HALF = (24, 72, 168, 500, math.inf)
ETAS = (1.0, 3.0, 10.0, 30.0, math.inf)
SHARE = (0, 1)
dev = IL.dev


def experts(Ech, Fp, k, keys):
    """-> names, pred (B, K, C, P) float32."""
    B, J, _, C = Ech.shape
    E = Ech.permute(0, 3, 1, 2)                                                         # (B, C, J, P)
    nm, pr = ['zero', 'mean'], [torch.zeros_like(E[:, :, 0]), E.mean(2)]
    for n in RECENT:
        nm.append(f'recent{n}'); pr.append(E[:, :, -n:].mean(2))
    for fam, Gs in (('single', [(g,) for g in range(k, J - NMIN + 1)]), ('pair', [(k, g) for g in range(k + 1, J - NMIN + 1)])):
        for G in Gs:
            b = build(Ech, Fp, k, G, keys=keys)
            if b is None:
                continue
            nm.append(f'{fam}{G[-1]}'); pr.append(solve(*b, 1.0)[0])
    return nm, torch.stack(pr, 1).float()


def expert_preds(F, Hs, k, gen, chunk, nsh):
    """-> names, r (N, C, P) f32, preds list over [real] + nsh nulls, each (N, K, C, P) f32."""
    N = len(F)
    rs, outs = [], [[] for _ in range(nsh + 1)]
    for i0 in range(0, N, chunk):
        Ft = torch.as_tensor(F[i0:i0 + chunk], dtype=torch.float64, device=dev)
        Ech = torch.as_tensor(np.ascontiguousarray(Hs[i0:i0 + chunk, ::-1]), dtype=torch.float64, device=dev)
        Fp = feats(Ech.permute(0, 1, 3, 2))
        B, J, _, C = Ech.shape
        rs.append(Ft[:, (k - 1) * P:k * P].transpose(1, 2).float())
        for d in range(nsh + 1):
            keys = None if d == 0 else torch.rand(B, C, J, generator=gen, device=dev, dtype=torch.float64)
            names, pr = experts(Ech, Fp, k, keys)
            outs[d].append(pr)
    return names, torch.cat(rs), [torch.cat(o) for o in outs]


CFGS = [(h, e, sh) for h in HALF for e in ETAS for sh in SHARE]


def hedge_all(loss, pr, r, delay, L0=None):
    """All (half-life, eta, share) configs at once. loss (N, K, C) per-window expert losses in time order (expert 0 = zero),
    pr (N, K, C, P), r (N, C, P). Window i may only use the outcomes of windows j <= i - delay.
    Returns SSE per config (len(CFGS),) and the memory (Hn, K, C) after flushing every outcome (for the next split)."""
    N, K, C = loss.shape
    gam = torch.tensor([1.0 if math.isinf(h) else 0.5 ** (1.0 / h) for h in HALF], dtype=torch.float64, device=loss.device)
    L = torch.zeros(len(HALF), K, C, dtype=torch.float64, device=loss.device) if L0 is None else L0.clone()
    fin = [e for e in ETAS if not math.isinf(e)]
    eta = torch.tensor(fin, dtype=torch.float64, device=loss.device)[None, :, None, None, None]
    sse = torch.zeros(len(HALF), len(ETAS), len(SHARE), dtype=torch.float64, device=loss.device)
    uni = torch.full((len(HALF), len(ETAS), len(SHARE), K, C), 1.0 / K, dtype=torch.float64, device=loss.device)
    for i in range(N):
        j = i - delay
        if j >= 0:
            L = gam[:, None, None] * L + loss[j]
        if bool((L[:, 0] > 0).all()):
            Ls = torch.stack([L, L.sum(-1, keepdim=True).expand_as(L)], 1)             # (Hn, S, K, C)
            rel = Ls / Ls[:, :, 0:1].clamp_min(1e-12)
            rel = rel - rel.min(2, keepdim=True).values
            Wf = torch.softmax(-eta * rel[:, None], 3)                                  # (Hn, Ef, S, K, C)
            Wi = torch.nn.functional.one_hot(rel.argmin(2), K).permute(0, 1, 3, 2).to(rel.dtype)[:, None]   # (Hn, 1, S, K, C)
            W = torch.cat([Wf, Wi], 1)
        else:
            W = uni
        d = torch.einsum('hesk c,kcp->hescp'.replace(' ', ''), W, pr[i].double())
        sse += ((r[i].double() - d) ** 2).sum((-1, -2))
    for j in range(max(N - delay, 0), N):
        L = gam[:, None, None] * L + loss[j]
    return sse.flatten(), L


def run(Rv, Rt, gen, chunk, nsh, stride):
    """Rv, Rt = (F, Hs) of val and test. Returns per lead dict of rows."""
    out = {}
    for k in range(1, NP + 1):
        t0 = time.time()
        delay = math.ceil(k * P / stride)
        names, rv, pv = expert_preds(*Rv, k, gen, chunk, nsh)
        _, rt, pt = expert_preds(*Rt, k, gen, chunk, nsh)
        lv = [((rv[:, None] - p) ** 2).mean(-1).double() for p in pv]                   # (N, K, C)
        lt = [((rt[:, None] - p) ** 2).mean(-1).double() for p in pt]
        ss_v, ss_t = (rv ** 2).sum().item(), (rt ** 2).sum().item()
        row = {}
        vals = []
        for d in range(nsh + 1):
            sv, Lend = hedge_all(lv[d], pv[d], rv, delay)
            st, _ = hedge_all(lt[d], pt[d], rt, delay, Lend)
            vals.append((1 - sv / ss_v, 1 - st / ss_t))
        res = {f'h{h}_e{e}_s{sh}': (vals[0][0][c].item(), vals[0][1][c].item(), float(np.mean([v[0][c].item() for v in vals[1:]])),
                                     float(np.mean([v[1][c].item() for v in vals[1:]]))) for c, (h, e, sh) in enumerate(CFGS)}
        best = max(res, key=lambda c: res[c][0])
        row['hedge'] = dict(cfg=best, val=res[best][0], test=res[best][1], sh_val=res[best][2], sh_test=res[best][3])
        # context: fixed experts (val-chosen, and test hindsight among single lags)
        fixed_v = {n: 1 - ((rv - pv[0][:, i]) ** 2).sum().item() / ss_v for i, n in enumerate(names)}
        fixed_t = {n: 1 - ((rt - pt[0][:, i]) ** 2).sum().item() / ss_t for i, n in enumerate(names)}
        fixed_ts = {n: float(np.mean([1 - ((rt - p[:, i]) ** 2).sum().item() / ss_t for p in pt[1:]])) for i, n in enumerate(names)}
        bv = max(fixed_v, key=fixed_v.get)
        bt = max((n for n in names if n.startswith('single')), key=fixed_t.get)
        row['fixed_valbest'] = dict(cfg=bv, test=fixed_t[bv], sh_test=fixed_ts[bv])
        row['single_hindsight'] = dict(cfg=bt, test=fixed_t[bt], sh_test=fixed_ts[bt])
        for n in ('mean', 'recent4', 'recent8', 'recent12'):
            row[n] = dict(test=fixed_t[n])
        row['all_cfgs'] = res
        out[k] = row
        print(f'  lead {k}: hedge [{best}] test {res[best][1]:+.3f} sh {res[best][3]:+.3f} (val {res[best][0]:+.3f} sh {res[best][2]:+.3f}) | '
              f'fixed val-best {bv} test {fixed_t[bv]:+.3f} sh {fixed_ts[bv]:+.3f} | hindsight {bt} {fixed_t[bt]:+.3f} sh {fixed_ts[bt]:+.3f} | '
              f'recent8 {fixed_t["recent8"]:+.3f} mean {fixed_t["mean"]:+.3f} | {len(names)} experts {time.time() - t0:.0f}s', flush=True)
    return out


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, H, split_x, a1_resid
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--L', default='1440,720'); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--nsh', type=int, default=3)
    a_ = ap.parse_args()
    cf = CFG[a_.data]; res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    gen = torch.Generator(device=dev).manual_seed(0)
    for L in [int(v) for v in a_.L.split(',')]:
        J = L // P - 1
        R = {}
        for sp in ('val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + J * P, len(X) - H + 1))[::a_.stride]
            R[sp] = a1_resid(ex, X, L, origins, dev, mmax=J)
        print(f'===== {a_.data} A1 L={L} (J={J}) val {len(R["val"][0])} / test {len(R["test"][0])} windows, stride {a_.stride} | config chosen on val per lead', flush=True)
        out = run(R['val'], R['test'], gen, a_.chunk, a_.nsh, a_.stride)
        h = np.array([out[k]['hedge']['test'] for k in range(1, NP + 1)]); s = np.array([out[k]['hedge']['sh_test'] for k in range(1, NP + 1)])
        print(f'  hedge test {" ".join(f"{x:+.3f}" for x in h)} | sh {" ".join(f"{x:+.3f}" for x in s)} | L4-7 hedge {h[3:].mean():+.3f} sh {s[3:].mean():+.3f} gain {h[3:].mean() - s[3:].mean():+.3f}', flush=True)
        res[f'A1_L{L}'] = {str(k): v for k, v in out.items()}
        json.dump(res, open(f'patchmem/logs/server/icl_hedge_{a_.data}_s{SEED}.json', 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
