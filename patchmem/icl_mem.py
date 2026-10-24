"""PAT-307 offline probe: TEFL's W2 as the slow weight of a self-attention 'dual' -- an implicit in-context update from the model's OWN
PAST FORECASTS (no training; frozen T0r (ETTh2: R1 + TEFL native Hc = 2) / TF0 (Weather: Hc = 0) checkpoints, seeds 3047/3048/3049).
At origin t the forecast is T0r's own pred_t = backbone rollout + W2 phi_t, phi = ReLU(W1 eps) (patchmem/r1_tefl.py). The added term
  delta_t = eta * sum_s a(phi_t, phi_s) v_s        (per variable; full horizon H = 336, all 7 leads at once)
over a memory of past origins s = t - d, d in {H, H + dstep, ..., M} (s + H <= t: every stored future is fully observed at t):
  key   phi_s = ReLU(W1 eps_s), eps_s = what T0r saw at s (its own standard lookback L, its own RevIN; the deployed forecast at s)
  value v_s   'own': y[s, s+H) - pred_s (T0r's own full-horizon error = v - W2 phi in GD terms, so attention over own errors is the
              implicit GD step from W2, no explicit residual model) | 'bb': the backbone-only error (W2 off, audit 'zero')
  a     'cos' softmax(cos / tau) | 'rbf' softmax(-|q-k|^2 / (tau mean_s |q-k|^2)) | 'lin' (phi_t . phi_s) / n (linear-attention dual)
        baselines 'mean' (1/n) and 'last' (the d = H demo only)
eta: closed-form least squares on val per config (one scalar shared by all leads), applied to test. Memory span M in MS (x H) and demo
spacing dstep in {P, H} explore the look-back length; model lookback L in {1440, 720}.
Shuffle (sh): per window and variable, the demo values are permuted against the keys (pairing broken, the demo-mean bias kept), NSH
draws averaged, at the real config and eta. All numbers: absolute test MSEred relative to T0r's own forecast, 1 - SSE / sum r^2,
over all 7 leads (= horizon MSE) and per lead.
Query origins = the official split origins (stride s): X is loaded with lookback L + Mmax so every query has the full memory.
Pre-registered (ETTh2, L = 1440): the attention config (cos / rbf / lin; any M, dstep, value) best on val must reach on test, ALL
three seeds: 7-lead MSEred >= 0.02, minus its shuffle >= 0.01, and above the best-on-val baseline (mean / last, any M). Weather,
L = 720 and the M curve are secondary.
usage: python patchmem/icl_mem.py --data E|W [--seeds 3047,3048,3049] [--L 1440,720] [--stride 1]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, H, NP = 48, 336, 7
MS = (1, 2, 3, 5, 8)                                                                    # memory span M = m * H
TAUS = {'cos': (0.03, 0.1, 0.3, 1.0), 'rbf': (0.1, 0.3, 1.0, 3.0)}
NSH = 3
ARM = {'E': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64'), des='R1TfnH2R64h0s{s}'),
       'W': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64'), des='R1WTfnH0R64h0s{s}')}


@torch.no_grad()
def forecasts(m, X, L, origins, bs=128):
    """-> own (N, H, C) T0r errors, bb (N, H, C) backbone-only errors, phi (N, C, r) keys; data (scaler) units."""
    Xt = torch.as_tensor(X)
    own, bb, phi = [], [], []
    for i0 in range(0, len(origins), bs):
        ts = origins[i0:i0 + bs]
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        y = torch.stack([Xt[t:t + H] for t in ts]).to(dev)
        m.set_audit(None)
        out = m(bx, NP, ret_eps=True)
        own.append((y - out['pred']).cpu()); phi.append(torch.relu(m.W1(out['eps'])).cpu())
        m.set_audit('zero')
        bb.append((y - m(bx, NP)['pred']).cpu())
    m.set_audit(None)
    return torch.cat(own), torch.cat(bb), torch.cat(phi)


def weights(q, K, kern, tau):
    """q (b, C, r), K (b, n, C, r) -> (b, C, n)."""
    K = K.transpose(1, 2)                                                               # (b, C, n, r)
    n = K.shape[2]
    if kern == 'mean':
        return K.new_full(K.shape[:3], 1.0 / n)
    if kern == 'last':
        w = K.new_zeros(K.shape[:3]); w[..., 0] = 1.0                                   # demo 0 = d = H (most recent complete)
        return w
    if kern == 'lin':
        return (K @ q[..., None])[..., 0] / n
    if kern == 'cos':
        s = (torch.nn.functional.normalize(K, dim=-1) @ torch.nn.functional.normalize(q, dim=-1)[..., None])[..., 0] / tau
    else:
        d2 = ((K - q[:, :, None]) ** 2).sum(-1)
        s = -d2 / (tau * d2.mean(-1, keepdim=True).clamp_min(1e-12))
    return torch.softmax(s, -1)


def configs():
    out = [('mean', 0), ('last', 0), ('lin', 0)]
    for kern, ts in TAUS.items():
        out += [(kern, t) for t in ts]
    return out


def score(own, bb, phi, qi, ds, gen, chunk=256):
    """qi query indices into the origin grid, ds demo offsets in grid steps (ds[0] = H / stride). -> {(src, kern, tau): stats}, stats
    = (draws, 3, NP): per draw (0 real, 1.. shuffles) <r, d>, <d, d>, <r, r> per lead."""
    cf = configs()
    st = {(src, k, t): torch.zeros(1 + NSH, 3, NP, dtype=torch.float64) for src in ('own', 'bb') for k, t in cf}
    ds_ = torch.as_tensor(ds)
    for c0 in range(0, len(qi), chunk):
        q = torch.as_tensor(qi[c0:c0 + chunk])
        di = q[:, None] - ds_[None, :]                                                  # (b, n)
        Q, K = phi[q].to(dev), phi[di].to(dev)
        r = own[q].to(dev).double()                                                     # (b, H, C) T0r's residual at t
        b, n, C = di.shape[0], di.shape[1], Q.shape[1]
        perms = [None] + [torch.rand(b, C, n, generator=gen).argsort(-1).to(dev) for _ in range(NSH)]
        W = {(k, t): weights(Q, K, k, t).double() for k, t in cf}
        for src, E in (('own', own), ('bb', bb)):
            V = E[di].to(dev).double().permute(0, 3, 1, 2)                              # (b, C, n, H)
            for di_, pm in enumerate(perms):
                Vp = V if pm is None else torch.gather(V, 2, pm[..., None].expand_as(V))
                for (k, t), w in W.items():
                    d = torch.einsum('bcn,bcnh->bhc', w, Vp).reshape(b, NP, P, C)
                    rr = r.reshape(b, NP, P, C)
                    st[(src, k, t)][di_, 0] += (rr * d).sum((0, 2, 3)).cpu()
                    st[(src, k, t)][di_, 1] += (d * d).sum((0, 2, 3)).cpu()
                    st[(src, k, t)][di_, 2] += (rr * rr).sum((0, 2, 3)).cpu()
    return st


def evaluate(sv, stt):
    """eta from val (real draw, all leads) -> dict(val, test, sh, lead (test per lead), eta)."""
    rd, dd = sv[0, 0].sum(), sv[0, 1].sum()
    eta = float(rd / dd) if dd > 0 else 0.0
    red = lambda s: 1 - (s[2] - 2 * eta * s[0] + eta ** 2 * s[1]).sum().item() / s[2].sum().item()
    lead = lambda s: (1 - (s[2] - 2 * eta * s[0] + eta ** 2 * s[1]) / s[2]).tolist()
    return dict(val=red(sv[0]), test=red(stt[0]), sh=float(np.mean([red(stt[i]) for i in range(1, stt.shape[0])])),
                lead=lead(stt[0]), sh_lead=np.mean([lead(stt[i]) for i in range(1, stt.shape[0])], 0).tolist(), eta=eta)


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='1440,720'); ap.add_argument('--stride', type=int, default=1)
    a_ = ap.parse_args()
    sd_ = a_.stride
    assert P % sd_ == 0 and H % sd_ == 0, sd_
    cf, arm = CFG[a_.data], ARM[a_.data]; res = {}
    os.environ.pop('R1_OBS', None); os.environ.update(arm['env'])
    from patchmem.exp_r1 import Exp_R1
    mext = max(MS) * H
    for seed in [int(s) for s in a_.seeds.split(',')]:
        des = arm['des'].format(s=seed)
        ck = cf['r1ck'].replace(cf['r1des'], des)
        assert os.path.exists(f'checkpoints/{ck}/checkpoint.pth'), ck
        ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
        ex = Exp_R1(ar)
        ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
        m = ex.model; m.eval(); ex._fix_eval_order()
        print(f'[ckpt] {ck} | model {type(m).__name__} hc={m.hc}', flush=True)
        gen = torch.Generator().manual_seed(seed)
        for L in [int(v) for v in a_.L.split(',')]:
            S = {}
            for sp in ('val', 'test'):
                X = split_x(ar, L + mext, sp)                                           # official split, lookback extended by mext
                origins = list(range(L, len(X) - H + 1, sd_))                           # grid: index i <-> origin L + i * stride
                own, bb, phi = forecasts(m, X, L, origins)
                qi = np.arange(mext // sd_, len(origins))                               # = the official origins of the split
                if sp == 'test':
                    print(f'  [sanity] L={L} T0r official test MSE {(own[qi] ** 2).mean():.4f} (backbone {(bb[qi] ** 2).mean():.4f}), '
                          f'{len(qi)} origins', flush=True)
                for mm in MS:
                    for dstep in (P, H):
                        ds = list(range(H // sd_, mm * H // sd_ + 1, dstep // sd_))
                        S[(sp, mm, dstep)] = score(own, bb, phi, qi, ds, gen)
            rows = {}
            for mm in MS:
                for dstep in (P, H):
                    for key in S[('val', mm, dstep)]:
                        rows[(mm, dstep) + key] = evaluate(S[('val', mm, dstep)][key], S[('test', mm, dstep)][key])
            att = {k: v for k, v in rows.items() if k[3] in ('cos', 'rbf', 'lin')}
            base = {k: v for k, v in rows.items() if k[3] in ('mean', 'last')}
            ba, bb_ = max(att, key=lambda k: att[k]['val']), max(base, key=lambda k: base[k]['val'])
            fmt = lambda k: f'M={k[0]}H dstep={k[1]} {k[2]} {k[3]} tau={k[4]}'
            print(f'===== {a_.data} T0 s{seed} L={L} | absolute test MSEred vs T0r own forecast (7 leads) | real / shuffle', flush=True)
            for src in ('own', 'bb'):
                for mm in MS:
                    for dstep in (P, H):
                        sub = lambda fam: max([k for k in rows if k[:3] == (mm, dstep, src) and k[3] in fam], key=lambda k: rows[k]['val'])
                        a, mb, la = sub(('cos', 'rbf', 'lin')), sub(('mean',)), sub(('last',))
                        print(f'  v={src:3s} M={mm}H dstep={dstep:3d} | att {rows[a]["test"]:+.4f} sh {rows[a]["sh"]:+.4f} ({a[3]} {a[4]}, eta {rows[a]["eta"]:+.2f}) '
                              f'| mean {rows[mb]["test"]:+.4f} | last {rows[la]["test"]:+.4f}', flush=True)
                bs_ = max([k for k in att if k[2] == src], key=lambda k: att[k]['val'])
                print(f'  best att v={src} [{fmt(bs_)}] test {att[bs_]["test"]:+.4f} sh {att[bs_]["sh"]:+.4f} | per lead {" ".join(f"{x:+.3f}" for x in att[bs_]["lead"])}', flush=True)
            A, B = att[ba], base[bb_]
            print(f'  best att  [{fmt(ba)}] test {A["test"]:+.4f} sh {A["sh"]:+.4f} val {A["val"]:+.4f} eta {A["eta"]:+.3f}', flush=True)
            print(f'    per lead {" ".join(f"{x:+.3f}" for x in A["lead"])} | sh {" ".join(f"{x:+.3f}" for x in A["sh_lead"])}', flush=True)
            print(f'  best base [{fmt(bb_)}] test {B["test"]:+.4f} val {B["val"]:+.4f} | per lead {" ".join(f"{x:+.3f}" for x in B["lead"])}', flush=True)
            print(f'  criterion: att {A["test"]:+.4f} (>= 0.02) | att - sh {A["test"] - A["sh"]:+.4f} (>= 0.01) | att - base {A["test"] - B["test"]:+.4f} (> 0)', flush=True)
            res[f's{seed}_L{L}'] = {'|'.join(map(str, k)): v for k, v in rows.items()}
            json.dump(res, open(f'patchmem/logs/server/icl_mem_{a_.data}.json', 'w'))
        del ex, m; torch.cuda.empty_cache()
    print('DONE')


if __name__ == '__main__':
    main()
