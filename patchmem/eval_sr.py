"""PAT-315: two-granularity error model on the frozen standalone iTransformer (L96, H192): a target-time shock carried
over time (error transfer, PaTH) + an origin residual predicted from the current state.

Motivation (diag_err_lead.py): one forecast's errors over its H leads are not one error (orig R1 .06-.51), but all the
forecasts of the same target share it (diagonal R1 .70-.89); the lead x lead correlation is Toeplitz with daily stripes,
i.e. the autocorrelation of a target-indexed shock series.  So
  E[o, h] = s_{o+h} + r[o, h],    s_tau = mean of the in-stream forecasts' errors on y[tau] (the observation diagonal),
                                  r = the origin / lead-specific remainder.
At origin t every s_tau with tau <= t-1 is observed (one new value per step and channel); r[o] is observed for o <= t-H.

Shock layer (target-fixed, carried over time), two parts:
  base_sh  fixed linear transfer, closed form: ridge from the shock history s_(t-1-j), j < 96, (+ bias) to E[t, :]
           (pooled channels, fit on train origins, lam on val).  Everything below is fitted to rho = E - base_sh.
  carry    per channel a state z of d = 64 damped-rotation modes (rho_k e^(i theta_k); theta initialised at the
           weekly harmonics 2 pi k / (168 q), q = steps per hour; memory initialised log-spaced 4 h .. 2 weeks).
           At origin t the newest shock enters: z <- Lam T_t z + B s_(t-1) / sig_t^gamma; read m_x[t, h] =
           sig_t^gamma C . Lam^(h+1) z (extrapolation with the fixed Lam).  Arms differ only in the state-driven T_t:
  s1_R  T = I                                   (no state: fixed damped rotation, RoPE-like)
  s1_D  T = g(x) I, g = sigmoid                 (state-driven scalar decay, no direction mixing)
  s1_P  T = I - b(x) u(x) u(x)^T on R^(2d)      (PaTH: state-driven Householder)
  (s1_I theta = 0 and s8_P nb = 8 lead buckets remain available, not run by default)
x_t = [the channel's last 96 inputs (y - mu_t) / sd_t, d mu_t / sd_t, d log sd_t] (the backbone input state, as Pd).
Trained on all train origins (targets inside train), target rho[t, h], truncated BPTT (one day), 10 epochs, Adam.

Residual layer (origin-fixed, state -> error): top-k by cos of the instance-normalised input window (same channel,
demos o <= t-H), m = (1/k) sum cos v_o, k in KS chosen on val per arm, target rho:
  S_e  v = rho[o]   (the whole remaining error vector)    S_r  v = r[o] = E[o, h] - s_{o+h}
Combination y^ = f + base_sh + a_b m_S + b_b m_x, (a_b, b_b) per lead bucket (NB = 8) by least squares on all val
origins; arms alone fit only their own coefficient.  Test once.

Criteria (fixed before the runs; a dataset counts iff 3/3 seeds; datasets E M1 M2 W T EL; a claim holds iff >= 4/6):
  C1 PaTH carry      s1_P beats s1_R and s1_D (each = base_sh + carry, test MSE)
  C1b carry adds     s1_P beats base_sh
  C2 granularity     (diag_shock_ridge.py) the shock history 'sh' beats the full newest diagonal 'D'
  C3 residual value  S_r (+) s1_P beats S_e (+) s1_P
  C4 vs baselines    S_r (+) s1_P beats TEFL (official, H192, 5 datasets) and the ridge 'all' arm (eval_err_value,
                     ~EFC-CET) -- per dataset, holds iff >= 4 of the datasets that have the baseline
Also reported: the energy shares of s and r in E over train + val origins.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
import patchmem.eval_freshcast as fc
from patchmem.eval_err_value import H, L

Q = {'E': 1, 'T': 1, 'EL': 1, 'M1': 4, 'M2': 4, 'W': 6}                # steps per hour
PW = 168                                                                 # week in hours (mode frequencies)
NB = 8                                                                   # lead buckets of a_b, b_b and of s8
KS = (16, 64, 256)
EPS = 1e-12
SIG_FLOOR = 0.05
EPOCHS, LR = 10, 3e-3
CH = 128                                                                 # channels per carry training chunk
CHR, BQ = 32, 64                                                         # retrieval: channels per chunk, queries
CARRY_ARMS = {'s1_I': (1, 'I'), 's1_R': (1, 'R'), 's1_D': (1, 'D'), 's1_P': (1, 'P'), 's8_P': (8, 'P')}
D_MODES = 64
S_ARMS = ('S_e', 'S_r')
LS = 96                                                                  # shock-history lags of the base


def lead_bucket(h, nb, dev):
    return torch.arange(h, device=dev) * nb // h


class ShockSSM(nn.Module):
    """Carried shock state z in R^(d x 2) per channel (d damped-rotation modes, RoPE view): at origin t
      z <- Lam T_t z + B w_t,   Lam = diag(rho_k rot(theta_k)),   w_t = newest diagonal means / sig_t^gamma (nb)
      read  m[t, h] = sig_t^gamma Re-readout C_bucket(h) of Lam^(h+1) z      (forecast of the target t + h)
    T_t by kind: I and R identity (I: theta = 0, no rotation), D g(x) I, P I - b(x) u(x) u(x)^T (PaTH, on R^(2d))."""
    def __init__(self, nx, nb, kind, q, d):
        super().__init__()
        self.nb, self.kind, self.d = nb, kind, d
        th = 2 * math.pi * torch.arange(d).float() / (PW * q)               # weekly harmonics (incl. daily ones)
        if kind == 'I':
            self.register_buffer('theta', torch.zeros(d))
        else:
            self.theta = nn.Parameter(th)
        tau = q * torch.logspace(math.log10(4), math.log10(2 * PW), d)      # memory 4 h .. 2 weeks
        tau = tau[torch.randperm(d, generator=torch.Generator().manual_seed(0))]
        self.rho_p = nn.Parameter(torch.logit(torch.exp(-1 / tau)))
        self.B = nn.Parameter(torch.randn(d * 2, nb, generator=torch.Generator().manual_seed(1)) * 0.1)
        self.Cr = nn.Parameter(torch.zeros(nb, d, 2))
        if kind == 'P':
            self.Wu = nn.Linear(nx, 2 * d)
            self.wb = nn.Linear(nx, 1)
            nn.init.constant_(self.wb.bias, -2.0)                              # beta ~ 0.24 at init
        elif kind == 'D':
            self.wd = nn.Linear(nx, 1)
            nn.init.constant_(self.wd.bias, 6.0)                               # g ~ 0.998 at init
        self.gamma = nn.Parameter(torch.ones(()))

    def g(self):
        return self.gamma.clamp(0, 2)

    def step(self, z, x, w):
        """z (B d 2), x (B nx), w (B nb) -> z."""
        Bn = z.shape[0]
        if self.kind == 'P':
            u = F.normalize(self.Wu(x), dim=-1)
            beta = 2 * torch.sigmoid(self.wb(x))
            zf = z.flatten(1)
            z = (zf - beta * u * (u * zf).sum(-1, keepdim=True)).view(Bn, self.d, 2)
        elif self.kind == 'D':
            z = torch.sigmoid(self.wd(x))[:, :, None] * z
        rho = torch.sigmoid(self.rho_p)
        c, s = rho * torch.cos(self.theta), rho * torch.sin(self.theta)
        z = torch.stack([c * z[..., 0] - s * z[..., 1], s * z[..., 0] + c * z[..., 1]], -1)
        return z + (w @ self.B.T).view(Bn, self.d, 2)

    def read(self, z, h):
        """(B h): readout of Lam^(j) z, j = 1..h, with the lead bucket's C."""
        j = torch.arange(1, h + 1, device=z.device, dtype=z.dtype)[:, None]
        a = torch.sigmoid(self.rho_p)[None] ** j
        pc, ps = a * torch.cos(j * self.theta), a * torch.sin(j * self.theta)  # h d
        re = pc * z[:, None, :, 0] - ps * z[:, None, :, 1]                      # B h d
        im = ps * z[:, None, :, 0] + pc * z[:, None, :, 1]
        Cb = self.Cr[lead_bucket(h, self.nb, z.device)]                         # h d 2
        return (re * Cb[..., 0] + im * Cb[..., 1]).sum(-1)


def norm_windows(X, t, c, l):
    """x~ (len(c) len(t) l): input windows X[t-l:t, c], minus mean, over sqrt(var + 1e-5) (backbone instance norm)."""
    w = X[:, c][t[:, None] + torch.arange(-l, 0, device=X.device)].permute(2, 0, 1)
    w = w - w.mean(-1, keepdim=True)
    return w / torch.sqrt(w.var(-1, unbiased=False, keepdim=True) + 1e-5)


def topk_read(Kn, vals, kq, h, ks):
    """Kn (c n l) unit-norm keys, vals: list of (n c h) values, kq (b) query indices.  Returns per value {k: m (b c h)}:
    (1/k) sum over the top-k cos among indices <= kq - h of cos * v."""
    cos = (Kn[:, kq] @ Kn.transpose(1, 2)).float()                                 # c b n
    n = Kn.shape[1]
    ok = torch.arange(n, device=Kn.device)[None] <= (kq - h)[:, None]
    cos = cos.masked_fill(~ok[None], -float('inf'))
    km = min(max(ks), n)
    val, idx = cos.topk(km, -1)                                                    # c b km
    w = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    out = []
    for V in vals:
        Vc = V.permute(1, 0, 2)                                                    # c n h
        gv = torch.gather(Vc[:, None].expand(-1, len(kq), -1, -1), 2, idx[..., None].expand(-1, -1, -1, h))
        cw = (w[..., None] * gv).cumsum(2)
        out.append({k: (cw[:, :, min(k, km) - 1] / k).permute(1, 0, 2) for k in ks})
    return out


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    assert H == 192
    od = Path(args.out_dir)
    done = od / f'{code}_H{H}_s{seed}{"_smoke" if args.max_c or args.epochs != EPOCHS else ""}.json'
    if done.exists():
        print(f'[skip] {done}', flush=True)
        return
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    X, M, n_tr, n_va = ev.load_series(exp, code)
    q = Q[code]
    T0 = max(L, fc.LQ) + 1 if args.weeks is None else n_tr - args.weeks * PW * q
    m_tr = n_tr - T0
    assert T0 >= max(L, fc.LQ) + 1
    E = ev.stream_errors(exp, X, M, T0, args.bs)                                   # n H C
    C = X.shape[1] if args.max_c is None else args.max_c
    E = E[..., :C].permute(0, 2, 1).contiguous()                                   # n C H
    n = E.shape[0]
    times = torch.arange(T0, T0 + n)
    split = torch.full((n,), -1)
    split[times <= n_tr - H] = 0
    split[(times >= n_tr) & (times <= n_tr + n_va - H)] = 1
    split[times >= n_tr + n_va] = 2
    nel = {s: int((split == s).sum()) * C * H for s in (0, 1, 2)}
    spl = split.to(dev)
    base = float(E[split == 2].double().square().sum()) / nel[2]
    if args.max_c is None and abs(base - official) > 1e-4:
        raise AssertionError(f'base {base} official {official}')
    print(f'[base] {base:.5f} n {n} C {C}', flush=True)
    bk = lead_bucket(H, NB, dev)
    kh = torch.arange(H)

    # ---- shock s_tau (index tau - T0, n + H - 1) and per-bucket diagonal means (the carry writes), CPU
    ssum = torch.zeros(n + H - 1, C, dtype=torch.float64)
    bsum = torch.zeros(n + H - 1, C, NB, dtype=torch.float64)
    cnt = torch.zeros(n + H - 1)
    bcnt = torch.zeros(n + H - 1, NB)
    bkc = bk.cpu()
    for l in range(H):
        ssum[l:l + n] += E[:, :, l].double()
        bsum[l:l + n, :, bkc[l]] += E[:, :, l].double()
        cnt[l:l + n] += 1
        bcnt[l:l + n, bkc[l]] += 1
    shock = (ssum / cnt[:, None]).float()                                          # n+H-1 C
    wdiag = {1: shock[..., None], NB: (bsum / bcnt.clamp_min(1)[:, None]).float()}
    del ssum, bsum
    sidx = torch.arange(n)[:, None] + kh                                           # n H: tau - T0 of E[k, :, h]


    # energy shares of s and r (train + val origins)
    sh = {'E': 0.0, 's': 0.0, 'r': 0.0}
    for i in range(0, n, 256):
        k = torch.arange(i, min(n, i + 256))
        k = k[split[k] >= 0]
        k = k[split[k] <= 1]
        if len(k) == 0:
            continue
        e_ = E[k].to(dev).double()
        s_ = shock[sidx[k]].permute(0, 2, 1).to(dev).double()
        sh['E'] += float(e_.square().sum())
        sh['s'] += float(s_.square().sum())
        sh['r'] += float((e_ - s_).square().sum())
    shares = dict(s=sh['s'] / sh['E'], r=sh['r'] / sh['E'])
    print(f'[shares] s {shares["s"]:.3f} r {shares["r"]:.3f}', flush=True)

    # ---- shock layer, base: closed-form ridge from the shock history s_(t-1-j), j < LS, to E[t] (pooled channels)
    jl = torch.arange(LS)
    cb = torch.arange(LS + 1, device=dev)

    def shfeat(k, c):                                                              # (len k, len c, LS + 1) on dev
        idx = k[:, None] - 1 - jl                                                  # tau - T0, observed (<= t - 1)
        f = (shock[idx.clamp_min(0)][..., c] * (idx >= 0)[..., None]).permute(0, 2, 1)
        return torch.cat([f, torch.ones_like(f[..., :1])], -1).to(dev)

    def bmom(ks):
        m = dict(xx=torch.zeros(LS + 1, LS + 1, dtype=torch.float64, device=dev),
                 xy=torch.zeros(LS + 1, H, dtype=torch.float64, device=dev),
                 yy=torch.zeros(H, dtype=torch.float64, device=dev), n=0)
        for i in range(0, len(ks), 64):
            k = ks[i:i + 64]
            Fk = shfeat(k, torch.arange(C)).double().reshape(-1, LS + 1)
            Y = E[k].to(dev).double().reshape(-1, H)
            m['xx'] += Fk.T @ Fk
            m['xy'] += Fk.T @ Y
            m['yy'] += Y.square().sum(0)
            m['n'] += len(Y)
        return m

    bm = {z: bmom(torch.nonzero(split == z).flatten()) for z in (0, 1, 2)}
    bval = {lam: ev.mse(bm[1], cb, ev.solve(bm[0], cb, lam)) for lam in ev.LAMS}
    blam = min(bval, key=bval.get)
    Wb = ev.solve(bm[0], cb, blam).float()                                          # LS+1 H
    res = {'base_sh': dict(lam=blam, val_mse=bval[blam], test_mse=ev.mse(bm[2], cb, ev.solve(bm[0], cb, blam)),
                           val_grid={str(k): v for k, v in bval.items()})}
    print(f'[base_sh] lam {blam:g} test {res["base_sh"]["test_mse"]:.5f}', flush=True)
    del bm

    def basep(k, c):                                                               # (len k, len c, H) on dev
        return shfeat(k, c) @ Wb

    for i in range(0, n, 256):                                                     # E <- rho = E - base
        k = torch.arange(i, min(n, i + 256))
        E[k] -= basep(k, torch.arange(C)).cpu()

    def resid(k, c):                                                               # r = E - s (len k, len c, H), dev
        return E[k][:, c].to(dev) + basep(k, c) - shock[sidx[k]][..., c].permute(0, 2, 1).to(dev)

    def bsumf(x):                                                                  # (b C H) -> (b NB) float64
        return torch.zeros(x.shape[0], NB, dtype=torch.float64, device=dev).index_add_(1, bk, x.double().sum(1))

    def acc(dst, x, ks_):                                                          # dst (3 NB)
        s_ = spl[ks_]
        for z in (0, 1, 2):
            dst[z] += x[s_ == z].sum(0)

    def z3():
        return torch.zeros(3, NB, dtype=torch.float64, device=dev)

    # ---- residual layer: retrieval, pass 1 sums for every k, pass 2 stores m of the val-chosen k
    Xd = X[:, :C].to(dev)
    td = times.to(dev)
    ee = z3()
    for i in range(0, n, 256):
        kq = torch.arange(i, min(n, i + 256), device=dev)
        acc(ee, bsumf(E[i:i + len(kq)].to(dev).square()), kq)

    def retrieval(ks, store=None, kb=None):
        sums = {a: {k: {'em': z3(), 'mm': z3()} for k in ks} for a in S_ARMS}
        with torch.no_grad():
            for c in torch.arange(C).split(CHR):
                cg = c.to(dev)
                Kn = norm_windows(Xd, td, cg, L)
                Kn = (Kn / Kn.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half()
                Ec = E[:, c].to(dev)
                Rc = torch.cat([resid(torch.arange(j, min(n, j + 512)), c) for j in range(0, n, 512)])
                for i in range(0, n, BQ):
                    kq = torch.arange(i, min(n, i + BQ), device=dev)
                    ms = dict(zip(S_ARMS, topk_read(Kn, [Ec, Rc], kq, H, ks)))
                    e_ = Ec[kq]
                    for a in S_ARMS:
                        for k in ks:
                            acc(sums[a][k]['em'], bsumf(e_ * ms[a][k]), kq)
                            acc(sums[a][k]['mm'], bsumf(ms[a][k].square()), kq)
                        if store is not None:
                            store[a][i:i + len(kq), c[0]:c[-1] + 1] = ms[a][kb[a]].half().cpu()   # c contiguous
                del Kn, Ec, Rc
        return sums

    def fit1(em, mm):
        a = torch.where(mm[1] > 0, em[1] / mm[1].clamp_min(EPS), torch.zeros_like(mm[1]))
        return a, (ee - 2 * a * em + a ** 2 * mm).sum(-1)

    sums = retrieval(KS)
    kbest = {}
    for a in S_ARMS:
        kbest[a] = min(KS, key=lambda k: float(fit1(sums[a][k]['em'], sums[a][k]['mm'])[1][1]))
        coef, sse = fit1(sums[a][kbest[a]]['em'], sums[a][kbest[a]]['mm'])
        res[a] = dict(k=kbest[a], a=coef.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                      grid={str(k): float(fit1(sums[a][k]['em'], sums[a][k]['mm'])[1][1]) / nel[1] for k in KS})
        print(f'[{a}] k {kbest[a]} test {res[a]["test_mse"]:.5f}', flush=True)
    MS = {a: torch.zeros(n, C, H, dtype=torch.float16) for a in S_ARMS}
    retrieval(tuple(sorted(set(kbest.values()))), MS, kbest)
    ss = {a: z3() for a in S_ARMS}
    es = {a: z3() for a in S_ARMS}
    for i in range(0, n, 256):
        kq = torch.arange(i, min(n, i + 256), device=dev)
        e_ = E[i:i + len(kq)].to(dev)
        for a in S_ARMS:
            m_ = MS[a][i:i + len(kq)].to(dev).float()
            acc(ss[a], bsumf(m_.square()), kq)
            acc(es[a], bsumf(e_ * m_), kq)

    # ---- shock layer: carried state
    mu_w, sd_w = fc.win_stats(X.to(dev))
    trd = (split == 0).to(dev)
    dmu = ((mu_w[T0:T0 + n] - mu_w[T0 - 1:T0 + n - 1]) / sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR))[:, :C]
    dls = (torch.log(sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR)) -
           torch.log(sd_w[T0 - 1:T0 + n - 1].clamp_min(fc.SD_FLOOR)))[:, :C]
    dmu, dls = dmu / dmu[trd].std().clamp_min(1e-6), dls / dls[trd].std().clamp_min(1e-6)
    sig = sd_w[T0:T0 + n][:, :C].clamp_min(SIG_FLOOR).float()
    kq96 = torch.arange(-fc.LQ, 0, device=dev)
    nx = fc.LQ + 2
    k_tr_end = int(torch.nonzero(split == 0).max()) + 1
    burn = 2 * PW * q + H
    tb = 24 * q                                                                    # truncated BPTT: one day

    def x_at(k, c):
        t = T0 + k
        w = (Xd[t + kq96][:, c].T - mu_w[t, c][:, None]) / sd_w[t, c].clamp_min(fc.SD_FLOOR)[:, None]
        return torch.cat([w, dmu[k, c][:, None], dls[k, c][:, None]], 1).float()

    def step(net, z, k, cg, wd):
        """wd (n+H-1 B nb) diagonal means on dev for the channels cg.  Returns state, m (B H)."""
        x = x_at(k, cg)
        if z is None:
            z = torch.zeros(len(cg), net.d, 2, device=dev)
        sg = sig[k, cg][:, None] ** net.g()
        w = wd[k - 1] / sg if k >= 1 else torch.zeros(len(cg), net.nb, device=dev)   # newest target t - 1
        z = net.step(z, x, w)
        return z, sg * net.read(z, H)

    def train_carry(nb, kind):
        torch.manual_seed(seed)
        net = ShockSSM(nx, nb, kind, q, D_MODES).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        g = torch.Generator().manual_seed(seed)
        log = []
        for ep in range(args.epochs):
            tot, cn = 0.0, 0
            for c in torch.randperm(C, generator=g).split(CH):
                cg = c.to(dev)
                e_ = E[:k_tr_end, c].to(dev)
                wd = wdiag[nb][:k_tr_end, c].to(dev)
                state, loss, nl = None, 0.0, 0
                for k in range(k_tr_end):
                    state, m = step(net, state, k, cg, wd)
                    if k >= burn:
                        loss = loss + (m - e_[k]).square().mean()
                        nl += 1
                    if (k + 1) % tb == 0 or k == k_tr_end - 1:
                        if nl:
                            opt.zero_grad()
                            (loss / nl).backward()
                            opt.step()
                            tot, cn = tot + float(loss), cn + nl
                        loss, nl = 0.0, 0
                        state = state.detach()
            log.append(tot / max(cn, 1))
        net.eval()
        return net, log

    def eval_carry(net):
        o = {key: z3() for key in ('ex', 'xx', 'sx_S_e', 'sx_S_r')}
        call = torch.arange(C, device=dev)
        wd = wdiag[net.nb].to(dev)
        state = None
        with torch.no_grad():
            for k in range(n):
                state, m = step(net, state, k, call, wd)
                z = int(split[k])
                if z < 0:
                    continue
                e_, x_ = E[k].to(dev).double(), m.double()
                v = {'ex': e_ * x_, 'xx': x_ * x_}
                for a in S_ARMS:
                    v[f'sx_{a}'] = MS[a][k].to(dev).double() * x_
                for key, val in v.items():
                    o[key][z] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, val.sum(0))
        return o

    def fit_alone(o):
        b = torch.where(o['xx'][1] > 0, o['ex'][1] / o['xx'][1].clamp_min(EPS), torch.zeros_like(o['xx'][1]))
        return b, (ee - 2 * b * o['ex'] + b ** 2 * o['xx']).sum(-1)

    def fit_joint(o, a_):
        A11, A12, A22, b1, b2 = ss[a_][1], o[f'sx_{a_}'][1], o['xx'][1], es[a_][1], o['ex'][1]
        det = A11 * A22 - A12 ** 2
        okd = det > 1e-12 * (A11 * A22).clamp_min(EPS)
        s_only = b1 ** 2 / A11.clamp_min(EPS) >= b2 ** 2 / A22.clamp_min(EPS)
        a = torch.where(okd, (A22 * b1 - A12 * b2) / det.clamp_min(EPS),
                        torch.where(s_only, b1 / A11.clamp_min(EPS), torch.zeros_like(b1)))
        b = torch.where(okd, (A11 * b2 - A12 * b1) / det.clamp_min(EPS),
                        torch.where(s_only, torch.zeros_like(b1), b2 / A22.clamp_min(EPS)))
        sse = (ee - 2 * a * es[a_] - 2 * b * o['ex'] + a ** 2 * ss[a_] + 2 * a * b * o[f'sx_{a_}']
               + b ** 2 * o['xx'])
        return a, b, sse.sum(-1)

    arms = args.arms.split(',')
    for name in arms:
        nb, kind = CARRY_ARMS[name]
        net, log = train_carry(nb, kind)
        o = eval_carry(net)
        b, sse = fit_alone(o)
        res[name] = dict(b=b.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                         train_loss=log, gamma=float(net.g()))
        for a_ in S_ARMS:
            a, b2, sse2 = fit_joint(o, a_)
            res[f'{a_}(+){name}'] = dict(a=a.tolist(), b=b2.tolist(), val_mse=float(sse2[1]) / nel[1],
                                         test_mse=float(sse2[2]) / nel[2])
        print(f'[{name}] test {res[name]["test_mse"]:.5f} | S_r(+) {res[f"S_r(+){name}"]["test_mse"]:.5f}', flush=True)

    tefl = ev.load_tefl(code, seed)
    rp = Path('patchmem/logs/server/err_value') / f'{code}_errval_s{seed}.json'
    ridge = json.loads(rp.read_text())['arms']['all']['test_mse'] if rp.exists() else None
    g_ = lambda x: 100 * (base - x) / base
    for k_, v in res.items():
        v['gain'] = g_(v['test_mse'])
        print(f'[test] {k_:12s} mse {v["test_mse"]:.5f} gain {v["gain"]:+.3f}% (val {v["val_mse"]:.5f})', flush=True)
    refs = dict(tefl=tefl['mse'] if tefl else None, ridge_all=ridge)
    print(f'[test] base {base:.5f} refs {refs}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official, refs=refs,
               shares=shares, settings=dict(q=q, PW=PW, m_tr=m_tr, NB=NB, KS=KS, EPOCHS=args.epochs, max_c=args.max_c),
               **res)
    od.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {done}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=tuple(Q))
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--max_c', type=int, default=None)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--weeks', type=int, default=None, help='train-tail weeks (default: the whole train segment)')
    p.add_argument('--arms', default='s1_R,s1_D,s1_P')
    p.add_argument('--out_dir', default='patchmem/logs/server/err_sr')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
