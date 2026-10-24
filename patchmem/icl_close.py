"""PAT-307 closing probe for the 'TEFL as a self-attention dual' line (frozen T0r (ETTh2) / TF0 (Weather), seeds 3047/3048/3049).
Memory = the model's OWN PAST FORECASTS (patchmem/icl_mem.py): for a query origin t, demos are past origins s = t - d, d in [Hh, M]
step P (s + Hh <= t). Values are always the RAW error over the forecast horizon (no projection), src 'own' (T0r's own error) or 'bb'
(backbone error, W2 off). The target is T0r's own residual at t. All numbers: absolute test MSEred vs T0r's own forecast.
Part A  bias memory (no keys): fixed-window means m_M (M in MS x Hh), exponential-decay means (scale lam x Hh, all demos up to 16 Hh),
        and a least-squares combination of the fixed-window means (coefficients fit on val). Hh in HS (the first Hh steps of the
        336-step rollout = the Hh-step forecast). One scalar eta / coefficient vector fit on val, applied to test.
Part B  H -> H fixed key (Hh = 336): key_s = src error of the forecast made at s - H (observed at s), query = src error at t - H;
        softmax attention (cos / rbf) or in-context kernel ridge (linear, unpenalised intercept = demo mean) from key to value.
        Shuffle: values (attention) / demo inputs (ridge) permuted against keys per window and variable. M in MB.
Part C  learned low-rank attention (Hh = 336): a = softmax(Wq x_t . Wk x_s / (sqrt(r) T)), delta = eta_k * sum_s a_s v_s, Wq, Wk
        (din -> r = 8, shared by variables), T, eta_1..7 trained (Adam, MSE on T0r's residual), backbone frozen. x = eps (what TEFL
        sees) or e_{s-H} (H -> H). Matched baseline: uniform weights with per-lead eta fit on the same training queries (= Wq = Wk = 0).
        Regimes: 'tr' trains on the train split, early stop / selection on val; 'vh' trains on the first half of val, early stop /
        selection on the second half. Shuffle at test: values permuted against keys.
Pre-registered (ETTh2, L = 1440, test, ALL three seeds):
  B  the val-best H -> H config (attention or ridge, any M / src) >= its matched mean + 0.01 AND >= its shuffle + 0.01
  C  regime 'tr': the val-best learned config >= its matched mean + 0.01 AND >= its shuffle + 0.01   ('vh' secondary)
  A  (Hh = 336) the val-best of {combination, exponential decay} >= the val-best fixed-window mean + 0.005
Weather, L = 720 and Hh = 96 / 192 are secondary.
usage: python patchmem/icl_close.py --data E|W [--seeds 3047,3048,3049] [--L 1440,720] [--stride 1] [--tr_sub 1] [--out DIR] [--smoke]"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.icl_lag import kridge
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, H, NP = 48, 336, 7
EXT = 17 * H                                                                            # lookback extension: 16 H of memory + e_{s-H} keys
MS = (1, 2, 3, 5, 8, 12, 16)
LAMD = (0.5, 1.0, 2.0, 4.0, 8.0)
HS = (96, 192, 336)
MB = (2, 8)
TAUS = {'cos': (0.03, 0.1, 0.3, 1.0), 'rbf': (0.1, 0.3, 1.0, 3.0)}
KRL = (0.3, 1.0, 3.0, 10.0)
NSH = 3
R, EPOCHS, PATIENCE, LR, BS = 8, 60, 5, 3e-3, 128
ARM = {'E': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64'), des='R1TfnH2R64h0s{s}'),
       'W': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64'), des='R1WTfnH0R64h0s{s}')}


@torch.no_grad()
def forecasts(m, X, L, origins, bs=128):
    """-> dict own / bb (N, H, C) errors (T0r / backbone only), eps (N, C, Din) TEFL inputs; cpu float32."""
    Xt = torch.as_tensor(X)
    o = {'own': [], 'bb': [], 'eps': []}
    for i0 in range(0, len(origins), bs):
        ts = origins[i0:i0 + bs]
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        y = torch.stack([Xt[t:t + H] for t in ts]).to(dev)
        m.set_audit(None)
        out = m(bx, NP, ret_eps=True)
        o['own'].append((y - out['pred']).cpu()); o['eps'].append(out['eps'].float().cpu())
        m.set_audit('zero')
        o['bb'].append((y - m(bx, NP)['pred']).cpu())
    m.set_audit(None)
    return {k: torch.cat(v) for k, v in o.items()}


def demo_idx(q, ds_steps, sd):
    """q (b,) query grid indices, ds_steps demo offsets in time steps -> (b, n) grid indices."""
    return q[:, None] - torch.as_tensor([d // sd for d in ds_steps], device=q.device)[None, :]


def red(ss, sse):
    return 1 - sse / ss


# ------------------------------------------------------------------ Part A: bias memory
def part_a(D, qi, sd, Hh, src, chunk=64):
    """-> (G (k, k), b (k,), ss) over all elements; regressors = fixed means (MS) then exp-decay means (LAMD)."""
    ds = list(range(Hh, 16 * Hh + 1, P))
    dd = torch.as_tensor(ds, dtype=torch.float64)
    Wm = [(dd <= M * Hh).double() for M in MS]
    Wm = [w / w.sum() for w in Wm]
    We = [torch.exp(-(dd - Hh) / (lam * Hh)) for lam in LAMD]
    We = [w / w.sum() for w in We]
    Wall = torch.stack(Wm + We).to(dev)                                                # (k, n)
    k = Wall.shape[0]
    G = torch.zeros(k, k, dtype=torch.float64, device=dev); b = torch.zeros(k, dtype=torch.float64, device=dev); ss = 0.0
    E = D[src]
    for c0 in range(0, len(qi), chunk):
        q = torch.as_tensor(qi[c0:c0 + chunk], device=dev)
        V = E[demo_idx(q, ds, sd)][:, :, :Hh].to(dev).double()                         # (b, n, Hh, C)
        r = D['own'][q][:, :Hh].to(dev).double()                                       # (b, Hh, C)
        Rg = torch.einsum('kn,bnhc->kbhc', Wall, V).reshape(k, -1)                     # (k, b*Hh*C)
        rv = r.reshape(-1)
        G += Rg @ Rg.T; b += Rg @ rv; ss += (rv * rv).sum().item()
    return G.cpu(), b.cpu(), ss


def eval_a(va, te):
    """va / te = (G, b, ss). -> rows {name: (val, test)} and the criterion pieces."""
    (Gv, bv, sv), (Gt, bt, st) = va, te
    sse = lambda c, G, b, s: s - 2 * (c @ b).item() + (c @ G @ c).item()
    rows = {}
    k = Gv.shape[0]
    for i in range(k):
        c = torch.zeros(k, dtype=torch.float64)
        c[i] = bv[i] / Gv[i, i] if Gv[i, i] > 0 else 0.0
        nm = f'mean{MS[i]}H' if i < len(MS) else f'exp{LAMD[i - len(MS)]}H'
        rows[nm] = (red(sv, sse(c, Gv, bv, sv)), red(st, sse(c, Gt, bt, st)))
    nmn = len(MS)
    A = Gv[:nmn, :nmn] + 1e-8 * torch.trace(Gv[:nmn, :nmn]) / nmn * torch.eye(nmn, dtype=torch.float64)
    c = torch.zeros(k, dtype=torch.float64); c[:nmn] = torch.linalg.solve(A, bv[:nmn])
    rows['comb'] = (red(sv, sse(c, Gv, bv, sv)), red(st, sse(c, Gt, bt, st)))
    fixed = max([n for n in rows if n.startswith('mean')], key=lambda n: rows[n][0])
    adapt = max([n for n in rows if n.startswith('exp') or n == 'comb'], key=lambda n: rows[n][0])
    return rows, fixed, adapt


# ------------------------------------------------------------------ Part B: H -> H fixed key
def part_b(D, qi, sd, M, src, gen, chunk=64):
    """-> {cfg: stats (draws, 3)}: per draw <r, d>, <d, d>, <r, r> (all elements, Hh = H). cfg = ('mean', 0) / (kern, tau) / ('rdg', lam)."""
    ds = list(range(H, M * H + 1, P))
    hs = H // sd
    E = D[src]
    cfgs = [('mean', 0)] + [(k, t) for k, ts in TAUS.items() for t in ts] + [('rdg', l) for l in KRL]
    st = {c: torch.zeros(1 + NSH, 3, dtype=torch.float64) for c in cfgs}
    for c0 in range(0, len(qi), chunk):
        q = torch.as_tensor(qi[c0:c0 + chunk], device=dev)
        di = demo_idx(q, ds, sd)
        V = E[di].to(dev).double().permute(0, 3, 1, 2)                                 # (b, C, n, H) values
        K = E[di - hs].to(dev).double().permute(0, 3, 1, 2)                            # (b, C, n, H) keys e_{s-H}
        Qy = E[q - hs].to(dev).double().permute(0, 2, 1)                               # (b, C, H) query e_{t-H}
        r = D['own'][q].to(dev).double().permute(0, 2, 1)                              # (b, C, H)
        b, C, n, _ = V.shape
        perms = [None] + [torch.rand(b, C, n, generator=gen).argsort(-1).to(dev) for _ in range(NSH)]
        Kn, Qn = nn.functional.normalize(K, dim=-1), nn.functional.normalize(Qy, dim=-1)
        cosv = (Kn @ Qn[..., None])[..., 0]
        d2 = ((K - Qy[:, :, None]) ** 2).sum(-1)
        d2n = d2 / d2.mean(-1, keepdim=True).clamp_min(1e-12)
        for di_, pm in enumerate(perms):
            g = (lambda T: T) if pm is None else (lambda T: torch.gather(T, 2, pm[..., None].expand_as(T)))
            Vp = g(V)
            outs = {('mean', 0): Vp.mean(2)}
            for t in TAUS['cos']:
                outs[('cos', t)] = (torch.softmax(cosv / t, -1)[..., None] * Vp).sum(2)
            for t in TAUS['rbf']:
                outs[('rbf', t)] = (torch.softmax(-d2n / t, -1)[..., None] * Vp).sum(2)
            Kp = g(K)                                                                  # ridge null: demo inputs permuted
            for l in KRL:
                pr, _ = kridge(Kp.reshape(b * C, n, H), V.reshape(b * C, n, H), Qy.reshape(b * C, 1, H), l)
                outs[('rdg', l)] = pr.reshape(b, C, H)
            for c, d in outs.items():
                st[c][di_, 0] += (r * d).sum().item(); st[c][di_, 1] += (d * d).sum().item(); st[c][di_, 2] += (r * r).sum().item()
    return st


def eval_b(sv, stt):
    eta = (sv[0, 0] / sv[0, 1]).item() if sv[0, 1] > 0 else 0.0
    f = lambda s: red(s[2].item(), (s[2] - 2 * eta * s[0] + eta ** 2 * s[1]).item())
    return dict(val=f(sv[0]), test=f(stt[0]), sh=float(np.mean([f(stt[i]) for i in range(1, stt.shape[0])])), eta=eta)


# ------------------------------------------------------------------ Part C: learned low-rank Wq / Wk
class LRAtt(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.Wq, self.Wk = nn.Linear(din, R, bias=False), nn.Linear(din, R, bias=False)
        nn.init.normal_(self.Wq.weight, std=1 / math.sqrt(din)); nn.init.normal_(self.Wk.weight, std=1 / math.sqrt(din))   # off the Wq = Wk = 0 saddle; eta = 0 keeps delta = 0 at init
        self.logt = nn.Parameter(torch.zeros(()))
        self.eta = nn.Parameter(torch.zeros(NP))

    def forward(self, xq, xk, V, perm=None):
        """xq (b, C, din), xk (b, C, n, din), V (b, C, n, H) -> (b, C, H)."""
        s = (self.Wk(xk) @ self.Wq(xq)[..., None])[..., 0] / (math.sqrt(R) * torch.exp(self.logt))
        a = torch.softmax(s, -1)
        if perm is not None:
            V = torch.gather(V, 2, perm[..., None].expand_as(V))
        d = (a[..., None] * V).sum(2)
        return (d.unflatten(-1, (NP, P)) * self.eta[:, None]).flatten(-2)


def batch_c(D, q, ds, sd, src, inp):
    hs = H // sd
    di = demo_idx(q, ds, sd)
    V = D[src][di].permute(0, 3, 1, 2)                                                  # (b, C, n, H)
    if inp == 'eps':
        xk, xq = D['eps'][di].permute(0, 2, 1, 3), D['eps'][q]                         # (b, C, n, din), (b, C, din)
    else:
        xk, xq = D[src][di - hs].permute(0, 3, 1, 2), D[src][q - hs].permute(0, 2, 1)
    r = D['own'][q].permute(0, 2, 1)
    return [t.to(dev) for t in (xq, xk, V, r)]


def sse_c(model, D, qi, ds, sd, src, inp, gen=None, nsh=0):
    """-> (ss, [sse real, sse shuffles...]) and per-lead uniform stats (<r, m>, <m, m>) per lead for the matched mean."""
    ss, out = 0.0, np.zeros(1 + nsh)
    with torch.no_grad():
        for c0 in range(0, len(qi), BS):
            q = torch.as_tensor(qi[c0:c0 + BS], device=dev)
            xq, xk, V, r = batch_c(D, q, ds, sd, src, inp)
            ss += (r * r).sum().item()
            for i in range(1 + nsh):
                pm = None if i == 0 else torch.rand(*V.shape[:3], generator=gen).argsort(-1).to(dev)
                out[i] += ((r - model(xq, xk, V, pm)) ** 2).sum().item()
    return ss, out


def mean_stats(D, qi, ds, sd, src):
    """Per-lead <r, m>, <m, m>, <r, r> for the uniform demo mean."""
    st = torch.zeros(3, NP, dtype=torch.float64)
    for c0 in range(0, len(qi), 512):
        q = torch.as_tensor(qi[c0:c0 + 512], device=dev)
        V = D[src][demo_idx(q, ds, sd)].to(dev).double().mean(1).unflatten(1, (NP, P))   # (b, NP, P, C)
        r = D['own'][q].to(dev).double().unflatten(1, (NP, P))
        st[0] += (r * V).sum((0, 2, 3)).cpu(); st[1] += (V * V).sum((0, 2, 3)).cpu(); st[2] += (r * r).sum((0, 2, 3)).cpu()
    return st


def mean_red(fit, ev):
    eta = fit[0] / fit[1].clamp_min(1e-12)
    return red(ev[2].sum().item(), (ev[2] - 2 * eta * ev[0] + eta ** 2 * ev[1]).sum().item())


def train_c(Dtr, qtr, Dva, qva, ds, sd, src, inp, seed):
    torch.manual_seed(seed)
    din = Dtr['eps'].shape[-1] if inp == 'eps' else H
    model = LRAtt(din).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bad, state = -1e9, 0, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        model.train()
        order = torch.as_tensor(qtr, device=dev)[torch.randperm(len(qtr), generator=g).to(dev)]
        for c0 in range(0, len(order), BS):
            xq, xk, V, r = batch_c(Dtr, order[c0:c0 + BS], ds, sd, src, inp)
            loss = ((r - model(xq, xk, V)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        ss, o = sse_c(model, Dva, qva, ds, sd, src, inp)
        v = red(ss, o[0])
        if v > best + 1e-6:
            best, bad, state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, best, ep + 1


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='1440,720')
    ap.add_argument('--stride', type=int, default=1); ap.add_argument('--tr_sub', type=int, default=1)
    ap.add_argument('--out', default='patchmem/logs/server')
    ap.add_argument('--smoke', action='store_true', help='crash test: first seed, first L, 64 / 256 queries, 2 epochs')
    a_ = ap.parse_args()
    global EPOCHS
    if a_.smoke:
        a_.seeds, a_.L, EPOCHS = a_.seeds.split(',')[0], a_.L.split(',')[0], 2
    sd = a_.stride
    global BS
    if a_.data == 'W':
        BS = 32                                                                         # Weather eps keys: din = 29 * 48
    assert P % sd == 0 and H % sd == 0, sd
    cf, arm = CFG[a_.data], ARM[a_.data]; res = {}
    os.makedirs(a_.out, exist_ok=True)
    os.environ.pop('R1_OBS', None); os.environ.update(arm['env'])
    from patchmem.exp_r1 import Exp_R1
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
            t0 = time.time()
            D, Q = {}, {}
            for sp in ('train', 'val', 'test'):
                X = split_x(ar, L + EXT, sp)
                origins = list(range(L, len(X) - H + 1, sd))
                D[sp] = {k: v.to(dev) for k, v in forecasts(m, X, L, origins).items()}               # all on the GPU
                if sp == 'train':                                                       # train data_x starts at 0 for any seq_len
                    Q[sp] = np.arange(9 * H // sd, len(origins))[::a_.tr_sub]           # room for M = 8 H demos + e_{s-H} keys
                else:
                    Q[sp] = np.arange(EXT // sd, len(origins))                          # = the official split origins
                if a_.smoke:
                    Q[sp] = Q[sp][:256 if sp == 'train' else 64]
            qt = torch.as_tensor(Q['test'], device=dev)
            print(f'  [sanity] L={L} T0r official test MSE {(D["test"]["own"][qt] ** 2).mean():.4f} (backbone {(D["test"]["bb"][qt] ** 2).mean():.4f}) '
                  f'| queries train {len(Q["train"])} val {len(Q["val"])} test {len(Q["test"])} | {time.time() - t0:.0f}s', flush=True)
            out = {}
            tag = f'{a_.data} T0 s{seed} L={L}'
            # ---- Part A
            print(f'===== A bias memory | {tag} | val / test MSEred (eta / coefficients fit on val)', flush=True)
            for Hh in HS:
                for src in ('own', 'bb'):
                    rows, fx, ad = eval_a(part_a(D['val'], Q['val'], sd, Hh, src), part_a(D['test'], Q['test'], sd, Hh, src))
                    out[f'A|{Hh}|{src}'] = dict(rows=rows, fixed=fx, adapt=ad)
                    f = lambda nm: f'{rows[nm][0]:+.3f}/{rows[nm][1]:+.3f}'
                    print(f'  H={Hh:3d} v={src:3s} | ' + ' '.join(f'{n}={f(n)}' for n in rows), flush=True)
                    print(f'  H={Hh:3d} v={src:3s} | val-best fixed {fx} test {rows[fx][1]:+.4f} | val-best adaptive {ad} test {rows[ad][1]:+.4f} '
                          f'| adaptive - fixed {rows[ad][1] - rows[fx][1]:+.4f}', flush=True)
            for Hh in HS:                                                               # A criterion: joint selection over src x method on val
                ra = {(src, n): out[f'A|{Hh}|{src}']['rows'][n] for src in ('own', 'bb') for n in out[f'A|{Hh}|own']['rows']}
                fx = max([k for k in ra if k[1].startswith('mean')], key=lambda k: ra[k][0])
                ad = max([k for k in ra if not k[1].startswith('mean')], key=lambda k: ra[k][0])
                print(f'  A criterion H={Hh}: adaptive {ad} test {ra[ad][1]:+.4f} - fixed {fx} test {ra[fx][1]:+.4f} = {ra[ad][1] - ra[fx][1]:+.4f} (>= 0.005)', flush=True)
            # ---- Part B
            print(f'===== B H->H fixed key | {tag} | test real / shuffle (eta fit on val)', flush=True)
            rb = {}
            for M in MB:
                for src in ('own', 'bb'):
                    sv, stt = part_b(D['val'], Q['val'], sd, M, src, gen), part_b(D['test'], Q['test'], sd, M, src, gen)
                    ev = {c: eval_b(sv[c], stt[c]) for c in sv}
                    for c, e in ev.items():
                        rb[(M, src) + c] = e
                    bat = max([c for c in ev if c[0] in ('cos', 'rbf')], key=lambda c: ev[c]['val'])
                    brd = max([c for c in ev if c[0] == 'rdg'], key=lambda c: ev[c]['val'])
                    mn = ev[('mean', 0)]
                    print(f'  M={M}H v={src:3s} | mean {mn["test"]:+.4f} | att {ev[bat]["test"]:+.4f} sh {ev[bat]["sh"]:+.4f} {bat} '
                          f'| ridge {ev[brd]["test"]:+.4f} sh {ev[brd]["sh"]:+.4f} {brd}', flush=True)
            kb = max([k for k in rb if k[2] != 'mean'], key=lambda k: rb[k]['val'])
            mb = rb[kb[:2] + ('mean', 0)]
            print(f'  B criterion [{kb}] test {rb[kb]["test"]:+.4f} | - matched mean {rb[kb]["test"] - mb["test"]:+.4f} (>= 0.01) '
                  f'| - shuffle {rb[kb]["test"] - rb[kb]["sh"]:+.4f} (>= 0.01)', flush=True)
            out['B'] = {'|'.join(map(str, k)): v for k, v in rb.items()}; out['B_best'] = str(kb)
            # ---- Part C
            print(f'===== C learned low-rank Wq/Wk (r={R}) | {tag} | test real / shuffle | matched uniform mean', flush=True)
            nv = len(Q['val']) // 2
            regimes = {'tr': (D['train'], Q['train'], D['val'], Q['val']), 'vh': (D['val'], Q['val'][:nv], D['val'], Q['val'][nv:])}
            for rg, (Dtr, qtr, Dse, qse) in regimes.items():
                rc = {}
                for M in MB:
                    ds = list(range(H, M * H + 1, P))
                    for src in ('own', 'bb'):
                        fit = mean_stats(Dtr, qtr, ds, sd, src)
                        mtest, msel = mean_red(fit, mean_stats(D['test'], Q['test'], ds, sd, src)), mean_red(fit, mean_stats(Dse, qse, ds, sd, src))
                        for inp in ('eps', 'ehh'):
                            t1 = time.time()
                            model, vsel, ne = train_c(Dtr, qtr, Dse, qse, ds, sd, src, inp, seed)
                            ss, o = sse_c(model, D['test'], Q['test'], ds, sd, src, inp, gen, NSH)
                            e = dict(sel=vsel, test=red(ss, o[0]), sh=float(np.mean([red(ss, x) for x in o[1:]])), mean_test=mtest, mean_sel=msel,
                                     T=float(torch.exp(model.logt)), eta=model.eta.detach().cpu().tolist(), epochs=ne)
                            rc[(M, src, inp)] = e
                            print(f'  [{rg}] M={M}H v={src:3s} x={inp:3s} | att {e["test"]:+.4f} sh {e["sh"]:+.4f} | mean {mtest:+.4f} '
                                  f'| sel {vsel:+.4f} (mean {msel:+.4f}) | T {e["T"]:.2f} ep {ne} {time.time() - t1:.0f}s', flush=True)
                kc = max(rc, key=lambda k: rc[k]['sel'])
                e = rc[kc]
                print(f'  C criterion [{rg}] [{kc}] test {e["test"]:+.4f} | - matched mean {e["test"] - e["mean_test"]:+.4f} (>= 0.01) '
                      f'| - shuffle {e["test"] - e["sh"]:+.4f} (>= 0.01)', flush=True)
                out[f'C|{rg}'] = {'|'.join(map(str, k)): v for k, v in rc.items()}; out[f'C|{rg}|best'] = str(kc)
            res[f's{seed}_L{L}'] = out
            json.dump(res, open(f'{a_.out}/icl_close_{a_.data}{"_smoke" if a_.smoke else ""}.json', 'w'))
            del D
        del ex, m; torch.cuda.empty_cache()
    print('DONE')


if __name__ == '__main__':
    main()
