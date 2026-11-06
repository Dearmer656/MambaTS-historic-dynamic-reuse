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

v2 (--base shrot --freeze_theta --bptt_h 168; diag_rot_ridge showed the fixed rotary state beats the 96 lags 5/6): the
base adds the fixed rotary shock state (theta at the weekly harmonics, memory 4 h .. 2 weeks; block 'rot' z_(t-1) or 'tail'
z_(t-97), chosen with lam on val; train moments after a 6-week warm-up), so the carry only sees what a fixed linear
rotary kernel cannot take; theta frozen, truncated BPTT one week.  Same arms and criteria C1 / C1b / C3 / C4.
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
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
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
CARRY_ARMS = {'s1_I': (1, 'I', 1), 's1_R': (1, 'R', 1), 's1_D': (1, 'D', 1), 's1_P': (1, 'P', 1), 's8_P': (8, 'P', 1),
              's1_P2': (1, 'P', 2), 's1_P4': (1, 'P', 4),                 # (lead buckets, T_t kind, Householders per step
              's1_Ponly': (1, 'P', 1, False),
              's1_G': (1, 'G', 1), 's1_GF': (1, 'GF', 1),
              's1_Pfix': (1, 'PF', 1), 's1_Pdiag': (1, 'PD', 1)}            # Step 2: fixed u / diagonal-only                   # per-mode state gate (+ per-mode phase shift)                             #  [, rotation: False = theta 0 + per-lead readout])
D_MODES = 64
S_ARMS = ('S_e', 'S_r')
LS = 96                                                                  # shock-history lags of the base


def lead_bucket(h, nb, dev):
    return torch.arange(h, device=dev) * nb // h


class ShockSSM(nn.Module):
    """Carried shock state z in R^(d x 2) per channel (d damped-rotation modes, RoPE view): at origin t
      z <- Lam T_t z + B w_t,   Lam = diag(rho_k rot(theta_k)),   w_t = newest diagonal means / sig_t^gamma (nb)
      read  m[t, h] = sig_t^gamma Re-readout C_bucket(h) of Lam^(h+1) z      (forecast of the target t + h)
    T_t by kind: I and R identity (I: theta = 0, no rotation), D g(x) I, P I - b(x) u(x) u(x)^T (PaTH, on R^(2d)),
    G per-mode gate g_k(x), GF gate + per-mode phase shift delta_k(x) (selective rotary); rot=False (P_only): theta 0, per-lead readout."""
    def __init__(self, nx, nb, kind, q, d, nh=1, rot=True):
        super().__init__()
        self.nb, self.kind, self.d, self.nh, self.rot = nb, kind, d, nh, rot
        th = 2 * math.pi * torch.arange(d).float() / (PW * q)               # weekly harmonics (incl. daily ones)
        if kind == 'I' or not rot:
            self.register_buffer('theta', torch.zeros(d))
        else:
            self.theta = nn.Parameter(th)
        tau = q * torch.logspace(math.log10(4), math.log10(2 * PW), d)      # memory 4 h .. 2 weeks
        tau = tau[torch.randperm(d, generator=torch.Generator().manual_seed(0))]
        self.rho_p = nn.Parameter(torch.logit(torch.exp(-1 / tau)))
        self.B = nn.Parameter(torch.randn(d * 2, nb, generator=torch.Generator().manual_seed(1)) * 0.1)
        self.Cr = nn.Parameter(torch.zeros(nb, d, 2))
        if not rot:                                                            # P_only: no extrapolation by Lam^(h+1),
            self.Ch = nn.Parameter(torch.zeros(H, d, 2))                       # one readout per lead from z_t
        if kind in ('P', 'PD'):                                                # PD: same generator, diagonal use
            self.Wu = nn.Linear(nx, 2 * d * nh)                                # nh = 1: identical to v1
            self.wb = nn.Linear(nx, nh)
            nn.init.constant_(self.wb.bias, -2.0)                              # beta ~ 0.24 at init
        elif kind == 'PF':                                                     # one global trainable u, beta_t from x
            self.uf = nn.Parameter(torch.randn(2 * d, generator=torch.Generator().manual_seed(2)))
            self.wb = nn.Linear(nx, 1)
            nn.init.constant_(self.wb.bias, -2.0)
        elif kind == 'D':
            self.wd = nn.Linear(nx, 1)
            nn.init.constant_(self.wd.bias, 6.0)                               # g ~ 0.998 at init
        elif kind in ('G', 'GF'):                                              # selective rotary: per-mode gate g_k(x)
            self.wg = nn.Linear(nx, d)
            nn.init.constant_(self.wg.bias, 6.0)
            if kind == 'GF':                                                   # + per-mode phase shift delta_k(x)
                self.wf = nn.Linear(nx, d)
                nn.init.zeros_(self.wf.weight)
                nn.init.zeros_(self.wf.bias)                                   # delta = 0 at init
        self.gamma = nn.Parameter(torch.ones(()))

    def g(self):
        return self.gamma.clamp(0, 2)

    def step(self, z, x, w):
        """z (B d 2), x (B nx), w (B nb) -> z."""
        Bn = z.shape[0]
        z = self.trans(z, x)
        rho = torch.sigmoid(self.rho_p)
        c, s = rho * torch.cos(self.theta), rho * torch.sin(self.theta)
        z = torch.stack([c * z[..., 0] - s * z[..., 1], s * z[..., 0] + c * z[..., 1]], -1)
        return z + (w @ self.B.T).view(Bn, self.d, 2)

    def u_beta(self, x):
        """(u (B 2d) unit, beta (B 1)) of a single-Householder kind (P / PF / PD, nh = 1)."""
        beta = 2 * torch.sigmoid(self.wb(x))[:, :1]
        if self.kind == 'PF':
            return F.normalize(self.uf, dim=0).expand(x.shape[0], -1), beta
        return F.normalize(self.Wu(x)[:, :2 * self.d], dim=-1), beta

    def trans(self, z, x):
        """The state-driven transition T_t only (before the fixed rotation Lam)."""
        Bn = z.shape[0]
        if self.kind in ('PF', 'PD'):
            u, beta = self.u_beta(x)
            zf = z.flatten(1)
            if self.kind == 'PF':                                              # I - beta_t u u^T, u fixed
                zf = zf - beta * u * (u * zf).sum(-1, keepdim=True)
            else:                                                              # I - beta_t diag(u_t^2)
                zf = zf * (1 - beta * u.square())
            return zf.view(Bn, self.d, 2)
        if self.kind == 'P':
            U = F.normalize(self.Wu(x).view(Bn, self.nh, 2 * self.d), dim=-1)
            beta = 2 * torch.sigmoid(self.wb(x))
            zf = z.flatten(1)
            for i in range(self.nh):                                           # T_t = H_1 ... H_nh, applied H_nh first
                u, b_ = U[:, self.nh - 1 - i], beta[:, self.nh - 1 - i:self.nh - i]
                zf = zf - b_ * u * (u * zf).sum(-1, keepdim=True)
            z = zf.view(Bn, self.d, 2)
        elif self.kind == 'D':
            z = torch.sigmoid(self.wd(x))[:, :, None] * z
        elif self.kind in ('G', 'GF'):
            z = torch.sigmoid(self.wg(x))[:, :, None] * z
            if self.kind == 'GF':
                dl = math.pi * torch.tanh(self.wf(x))                          # (B d), |delta| < pi
                cd, sd_ = torch.cos(dl), torch.sin(dl)
                z = torch.stack([cd * z[..., 0] - sd_ * z[..., 1], sd_ * z[..., 0] + cd * z[..., 1]], -1)
        return z

    def read(self, z, h):
        """(B h): readout of Lam^(j) z, j = 1..h, with the lead bucket's C (P_only: per-lead C_h . z)."""
        if not self.rot:
            return torch.einsum('bdc,hdc->bh', z, self.Ch[:h])
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
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    ltag = '' if L == 96 else f'_L{L}'
    done = od / f'{code}_H{H}{ltag}_s{seed}{"_smoke" if args.max_c or args.epochs != EPOCHS else ""}.json'
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
    if args.val_split:                                                             # val_fit (1) | val_select (3), in time
        vk = torch.nonzero(split == 1).flatten()
        split[vk[len(vk) // 2:]] = 3
    SEL = 3 if args.val_split else 1                                               # selection rows (epoch / k / lam / arm)
    nel = {s: max(int((split == s).sum()) * C * H, 1) for s in (0, 1, 2, 3)}
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
        k = k[(split[k] == 0) | (split[k] == 1) | (split[k] == 3)]
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
    ND = 2 * D_MODES
    NFB = LS + (2 * ND if args.base == 'shrot' else 0) + 1
    if args.base == 'shrot':                                                       # fixed rotary state (diag_rot_ridge)
        th_ = 2 * math.pi * torch.arange(D_MODES, dtype=torch.float64) / (PW * q)
        rho_ = torch.exp(-1 / (q * torch.logspace(math.log10(4), math.log10(2 * PW), D_MODES, dtype=torch.float64)))
        lam_c, gin = (rho_ * torch.exp(1j * th_)).to(dev), (1 - rho_).to(dev)
        Zr = torch.zeros(n, C, ND)                                                 # Zr[k] = z_(k-1), shocks <= t - 1
        zc = torch.zeros(C, D_MODES, dtype=torch.complex128, device=dev)
        sdv = shock.to(dev).double()
        for j in range(n - 1):
            zc = lam_c * zc + gin * sdv[j][:, None]
            Zr[j + 1] = torch.cat([zc.real, zc.imag], -1).float().cpu()
        del sdv, zc

    def shfeat(k, c):                                                              # (len k, len c, NFB) on dev
        idx = k[:, None] - 1 - jl                                                  # tau - T0, observed (<= t - 1)
        f = (shock[idx.clamp_min(0)][..., c] * (idx >= 0)[..., None]).permute(0, 2, 1)
        parts = [f]
        if args.base == 'shrot':
            parts += [Zr[k][:, c], Zr[(k - LS).clamp_min(0)][:, c] * (k >= LS)[:, None, None]]
        return torch.cat(parts + [torch.ones_like(f[..., :1])], -1).to(dev)

    def bmom(ks):
        m = dict(xx=torch.zeros(NFB, NFB, dtype=torch.float64, device=dev),
                 xy=torch.zeros(NFB, H, dtype=torch.float64, device=dev),
                 yy=torch.zeros(H, dtype=torch.float64, device=dev), n=0)
        for i in range(0, len(ks), 64):
            k = ks[i:i + 64]
            Fk = shfeat(k, torch.arange(C)).double().reshape(-1, NFB)
            Y = E[k].to(dev).double().reshape(-1, H)
            m['xx'] += Fk.T @ Fk
            m['xy'] += Fk.T @ Y
            m['yy'] += Y.square().sum(0)
            m['n'] += len(Y)
        return m

    kwarm = 3 * 2 * PW * q if args.base == 'shrot' else 0                         # rotary warm-up (3 x slowest tau)
    ktr = torch.nonzero(split == 0).flatten()
    bm = {z: bmom(ktr[ktr >= kwarm] if z == 0 else torch.nonzero(split == z).flatten())
          for z in ((0, 1, 2, 3) if args.val_split else (0, 1, 2))}
    one = torch.tensor([NFB - 1])
    cands = {'sh': torch.cat([jl, one])}
    if args.base == 'shrot':                                                       # rot or tail block, chosen on val
        cands = {'sh+rot': torch.cat([jl, torch.arange(LS, LS + ND), one]),
                 'sh+tail': torch.cat([jl, torch.arange(LS + ND, LS + 2 * ND), one])}
    bval = {(nm, lam): ev.mse(bm[SEL], cc.to(dev), ev.solve(bm[0], cc.to(dev), lam))
            for nm, cc in cands.items() for lam in ev.LAMS}
    bname, blam = min(bval, key=bval.get)
    cb = cands[bname].to(dev)
    Wb = ev.solve(bm[0], cb, blam).float()                                          # |cb| H
    res = {'base_sh': dict(block=bname, lam=blam, val_mse=bval[(bname, blam)],
                           test_mse=ev.mse(bm[2], cb, ev.solve(bm[0], cb, blam)),
                           val_grid={f'{nm}|{lam:g}': v for (nm, lam), v in bval.items()})}
    print(f'[base_sh] {bname} lam {blam:g} test {res["base_sh"]["test_mse"]:.5f}', flush=True)
    del bm

    def basep(k, c):                                                               # (len k, len c, H) on dev
        return shfeat(k, c)[..., cb] @ Wb

    for i in range(0, n, 256):                                                     # E <- rho = E - base
        k = torch.arange(i, min(n, i + 256))
        E[k] -= basep(k, torch.arange(C)).cpu()

    def resid(k, c):                                                               # r = E - s (len k, len c, H), dev
        return E[k][:, c].to(dev) + basep(k, c) - shock[sidx[k]][..., c].permute(0, 2, 1).to(dev)

    def bsumf(x):                                                                  # (b C H) -> (b NB) float64
        return torch.zeros(x.shape[0], NB, dtype=torch.float64, device=dev).index_add_(1, bk, x.double().sum(1))

    def acc(dst, x, ks_):                                                          # dst (3 NB)
        s_ = spl[ks_]
        for z in (0, 1, 2, 3):
            dst[z] += x[s_ == z].sum(0)

    def z3():                                                                      # rows train / val_fit / test / val_select
        return torch.zeros(4, NB, dtype=torch.float64, device=dev)

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
                Kn = norm_windows(Xd, td, cg, args.se_len or L)                     # S_e key window (default L)
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
        kbest[a] = min(KS, key=lambda k: float(fit1(sums[a][k]['em'], sums[a][k]['mm'])[1][SEL]))
        coef, sse = fit1(sums[a][kbest[a]]['em'], sums[a][kbest[a]]['mm'])
        res[a] = dict(k=kbest[a], a=coef.tolist(), val_mse=float(sse[SEL]) / nel[SEL], test_mse=float(sse[2]) / nel[2],
                      grid={str(k): float(fit1(sums[a][k]['em'], sums[a][k]['mm'])[1][SEL]) / nel[SEL] for k in KS})
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
    tb = args.bptt_h * q                                                           # truncated BPTT (hours)
    zero_ref = float(E[burn:k_tr_end].double().square().mean())                    # carry train loss with m = 0
    res['base_sh']['carry_zero_ref'] = zero_ref
    print(f'[zero] carry train target rho = E - base: mean rho^2 {zero_ref:.5f} (k {burn}..{k_tr_end})', flush=True)
    if args.zero_only:
        return

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

    def train_carry(nb, kind, nh=1, rot=True):
        torch.manual_seed(seed)
        net = ShockSSM(nx, nb, kind, q, D_MODES, nh, rot).to(dev)
        if args.freeze_theta and isinstance(net.theta, nn.Parameter):
            net.theta.requires_grad_(False)                                        # keep the weekly harmonics
        prm = [p_ for p_ in net.parameters() if p_.requires_grad]
        opt = torch.optim.Adam(prm, lr=args.lr)
        g = torch.Generator().manual_seed(seed)
        log, vlog, best, ulog = [], [], [float('inf'), None, -1], []
        nup, win = 0, [0.0, 0]
        for ep in range(args.epochs):
            lr_ep = args.lr * (0.5 * (1 + math.cos(math.pi * ep / args.epochs)) if args.lr_sched == 'cos' else 1.0)
            tot, cn, t_ep = 0.0, 0, time.time()
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
                            if args.log_updates:                                   # (epoch, mean loss, grad norm)
                                gn = torch.stack([p_.grad.detach().double().norm() for p_ in prm if p_.grad is not None]).norm()
                                ulog.append((ep, float(loss) / nl, float(gn)))
                            if args.lr_sched != 'const' or args.warmup:            # linear warmup over updates
                                for pg in opt.param_groups:
                                    pg['lr'] = lr_ep * min(1.0, (nup + 1) / max(args.warmup, 1))
                            opt.step()
                            nup += 1
                            tot, cn = tot + float(loss), cn + nl
                            win[0] += float(loss); win[1] += nl
                            if nup % args.log_every == 0:                          # running window of the last updates
                                print(f'[carry {kind}{nh}] ep {ep} upd {nup} train {win[0] / win[1]:.5f} '
                                      f'({100 * (zero_ref - win[0] / win[1]) / zero_ref:+.2f}% vs zero) '
                                      f'lr {opt.param_groups[0]["lr"]:.2e}', flush=True)
                                win[:] = [0.0, 0]
                        loss, nl = 0.0, 0
                        state = state.detach()
            log.append(tot / max(cn, 1))
            if args.early_stop:                                                    # keep the epoch best on val
                v_ = float(fit_alone(eval_carry(net))[1][SEL]) / nel[SEL]   # b on val_fit, chosen on val_select
                vlog.append(v_)
                if v_ < best[0]:
                    best[:] = [v_, copy.deepcopy(net.state_dict()), ep]
            print(f'[carry {kind}{nh}] epoch {ep} train {log[-1]:.5f} ({100 * (zero_ref - log[-1]) / zero_ref:+.2f}% vs zero '
                  f'{zero_ref:.5f})' + (f' val {vlog[-1]:.5f}' if args.early_stop else '') +
                  f' lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t_ep:.0f}s', flush=True)
        if args.early_stop:
            net.load_state_dict(best[1])
            log.append(dict(val=vlog, best_epoch=best[2]))
        if args.log_updates:
            log.append(dict(updates=ulog))
        net.eval()
        return net, log

    def eval_carry(net, dump=None, mech=None):
        o = {key: z3() for key in ('ex', 'xx', 'sx_S_e', 'sx_S_r')}
        if EXT:                                                                    # --ext: Gram sums of [S_e, P, ext...]
            npr = 2 + len(EXT)
            o['G'] = torch.zeros(4, NB, npr, npr, dtype=torch.float64, device=dev)
            o['g'] = torch.zeros(4, NB, npr, dtype=torch.float64, device=dev)
            if args.roll:                                                          # per origin x lead (val + test rows)
                o['rows'] = torch.nonzero(split >= 1).flatten()
                o['rpos'] = torch.full((n,), -1, dtype=torch.long)
                o['rpos'][o['rows']] = torch.arange(len(o['rows']))
                o['RG'] = torch.zeros(len(o['rows']), H, npr, npr, dtype=torch.float64, device=dev)
                o['Rg'] = torch.zeros(len(o['rows']), H, npr, dtype=torch.float64, device=dev)
                o['Re'] = torch.zeros(len(o['rows']), H, dtype=torch.float64, device=dev)
        call = torch.arange(C, device=dev)
        wd = wdiag[net.nb].to(dev)
        state = None
        u_prev = None
        with torch.no_grad():
            for k in range(n):
                if dump is not None and int(split[k]) == 2 and state is not None:
                    x = x_at(k, call)
                    zf = state.flatten(1)
                    zn = zf.norm(dim=-1).clamp_min(EPS)
                    if net.kind in ('P', 'PF', 'PD'):                      # beta, |T_t z - z| / |z|, u drift
                        if net.kind == 'P':                                # (as before, incl. nh > 1)
                            u = F.normalize(net.Wu(x), dim=-1)
                            beta = 2 * torch.sigmoid(net.wb(x))[:, 0]
                        else:
                            u, beta = net.u_beta(x)
                            beta = beta[:, 0]
                        dump['beta'].append(beta.cpu())
                        if net.kind == 'PD':
                            dump['hh_rel'].append((beta[:, None] * u.square() * zf).norm(dim=-1).div(zn).cpu())
                        else:
                            dump['hh_rel'].append((beta * (u * zf).sum(-1).abs() / zn).cpu())
                        if u_prev is not None:
                            dump['u_cos'].append((u * u_prev).sum(-1).cpu())
                        u_prev = u
                    elif net.kind == 'D':
                        dump['gate'].append(torch.sigmoid(net.wd(x))[:, 0].cpu())
                    elif net.kind in ('G', 'GF'):                              # per-mode gate: mean over modes (steps C)
                        gk = torch.sigmoid(net.wg(x))
                        dump['gate'].append(gk.mean(-1).cpu())
                        dump['gate_min'].append(gk.min(-1).values.cpu())
                        if net.kind == 'GF':
                            dump['phase'].append((math.pi * torch.tanh(net.wf(x))).abs().mean(-1).cpu())
                if mech is not None and int(split[k]) == 2 and state is not None:   # T_t per rotary mode, test
                    x = x_at(k, call)
                    zp = net.trans(state, x)
                    a0, a1 = torch.view_as_complex(state.contiguous()), torch.view_as_complex(zp.contiguous())
                    ok = (a0.abs() > 1e-8).double()
                    rat = (a1 / torch.where(a0.abs() > 1e-8, a0, torch.ones_like(a0))).to(torch.complex128)
                    mech['n'] += ok.sum(0)
                    mech['dlogamp'] += (torch.log(rat.abs().clamp_min(1e-12)) * ok).sum(0)
                    mech['dphase_abs'] += (torch.angle(rat).abs() * ok).sum(0)
                    mech['dphase_cos'] += (torch.cos(torch.angle(rat)) * ok).sum(0)
                    mech['dphase_sin'] += (torch.sin(torch.angle(rat)) * ok).sum(0)
                    w0 = a0.abs().double().square()                            # energy weights (tiny states: noisy phase)
                    mech['w_dlogamp'] += (torch.log(rat.abs().clamp_min(1e-12)) * w0).sum(0)
                    mech['w_dphase_abs'] += (torch.angle(rat).abs() * w0).sum(0)
                    mech['amp2_pre'] += a0.abs().double().square().sum(0)
                    mech['amp2_post'] += a1.abs().double().square().sum(0)
                    if net.kind == 'PD':                                       # diagonal: all of the update is self-mode
                        u, beta = net.u_beta(x)
                        upd = (beta * u.square() * state.flatten(1)).view(-1, net.d, 2)
                        mech['self2'] += upd.square().sum(-1).double().sum(0)
                    if net.kind in ('P', 'PF') and net.nh == 1:                # self vs cross-mode part of beta u u^T z
                        u, beta = net.u_beta(x)
                        u, beta = u.reshape(-1, net.d, 2), beta[:, :, None]
                        pj = (u * state).sum(-1)                               # (C d): u_j . z_j
                        tot = pj.sum(-1, keepdim=True)
                        mech['self2'] += (beta * u * pj[..., None]).square().sum(-1).double().sum(0)
                        mech['cross2'] += (beta * u * (tot - pj)[..., None]).square().sum(-1).double().sum(0)
                state, m = step(net, state, k, call, wd)
                z = int(split[k])
                if mech is not None and z == 2:
                    mech['MX'][k - k_te0] = m.cpu()
                if dump is not None and z == 2:
                    dump['m_rms'].append(m.square().mean(-1).sqrt().cpu())
                    dump['rho_rms'].append(E[k].square().mean(-1).sqrt())
                if z < 0:
                    continue
                e_, x_ = E[k].to(dev).double(), m.double()
                v = {'ex': e_ * x_, 'xx': x_ * x_}
                for a in S_ARMS:
                    v[f'sx_{a}'] = MS[a][k].to(dev).double() * x_
                for key, val in v.items():
                    o[key][z] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, val.sum(0))
                if EXT and z >= 1:
                    Rk = torch.stack([MS['S_e'][k].to(dev).double(), x_] +
                                     [EXT[nm][0][EXT[nm][1][k]].to(dev).double() for nm in EXT])      # p C H
                    o['G'][z] += torch.zeros(NB, npr, npr, dtype=torch.float64, device=dev).index_add_(
                        0, bk, torch.einsum('ich,jch->hij', Rk, Rk))
                    o['g'][z] += torch.zeros(NB, npr, dtype=torch.float64, device=dev).index_add_(
                        0, bk, torch.einsum('ich,ch->hi', Rk, e_))
                    if args.roll:
                        r_ = int(o['rpos'][k])
                        o['RG'][r_] = torch.einsum('ich,jch->hij', Rk, Rk)
                        o['Rg'][r_] = torch.einsum('ich,ch->hi', Rk, e_)
                        o['Re'][r_] = e_.square().sum(0)
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
    ROLL_W = [int(w) for w in args.roll.split(',')] if args.roll else []
    EXT = {}
    for nm in (args.ext.split(',') if args.ext else []):                          # raw FC / B corrections (eval_freshcast)
        nm, _, xdir = nm.partition('@')                                            # name@dir: e.g. A2@.../err_pathret/A2
        zx = np.load(Path(xdir or args.ext_dir) / f'{code}_H{H}_L{L}_s{seed}_ext.npz')
        idx = torch.full((n,), -1, dtype=torch.long)
        kk = torch.as_tensor(zx['origins']) - T0
        idx[kk] = torch.arange(len(kk))
        need = split >= 1
        assert (idx[need] >= 0).all(), f'{nm}: missing val/test origins'
        EXT[nm] = (torch.from_numpy(zx[nm][:, :C]), idx)
        print(f'[ext] {nm} {tuple(zx[nm].shape)}', flush=True)
    TPL_GRID = {'Rm': (1, (24, 48, 96, 192)), 'Dm': (24, (1, 2, 4, 8)), 'Wm': (168, (1, 2, 4, 8))}   # (period h, W)
    if args.tpl:                                                                   # causal running-mean error templates
        hh = torch.arange(H, device=dev)

        def tpl_chunk(Ec, P, W):
            """Ec (n cc H) rho on dev.  mean of rho_(t - P m)[h], m = m0(h) .. m0(h) + W - 1, m0 = ceil((h+1)/P)
            (targets observed by t - 1); fewer terms near the start.  Returns (n cc H)."""
            nn_ = Ec.shape[0]
            npad = -nn_ % P
            Sp = torch.cat([Ec, Ec.new_zeros(npad, *Ec.shape[1:])]).reshape(-1, P, *Ec.shape[1:]).cumsum(0)
            Sp = Sp.reshape(-1, *Ec.shape[1:])[:nn_]                               # Sp[k] = sum of rho_i, i <= k, i = k mod P
            t_ = torch.arange(nn_, device=dev)[:, None]
            m0 = (hh + P) // P                                                     # ceil((h + 1) / P)
            hi = t_ - P * m0[None]                                                 # (n H)
            lo = hi - P * W
            g = lambda ix: torch.where((ix >= 0)[:, None, :], Sp[ix.clamp_min(0), :, hh[None].expand(nn_, -1)].permute(0, 2, 1), 0.0)
            cnt = torch.where(hi >= 0, torch.clamp(hi // P + 1, max=W), torch.zeros_like(hi)).clamp_min(1)
            return (g(hi) - g(lo)) / cnt[:, None, :]

        tsum = {(nm, W): {'em': z3(), 'mm': z3()} for nm, (P, Ws) in TPL_GRID.items() if nm in args.tpl.split(',') for W in Ws}
        kall = torch.arange(n, device=dev)
        for c in torch.arange(C).split(64):
            Ec = E[:, c].to(dev)
            for (nm, W), d in tsum.items():
                m_ = tpl_chunk(Ec, TPL_GRID[nm][0] * q, W)
                acc(d['em'], bsumf(Ec * m_), kall)
                acc(d['mm'], bsumf(m_.square()), kall)
        for nm in args.tpl.split(','):
            P, Ws = TPL_GRID[nm]
            fitW = {W: fit1(tsum[(nm, W)]['em'], tsum[(nm, W)]['mm']) for W in Ws}
            Wb = min(Ws, key=lambda W: float(fitW[W][1][SEL]))
            res[f'tpl_{nm}'] = dict(W=Wb, P=P, val_mse=float(fitW[Wb][1][SEL]) / nel[SEL], test_mse=float(fitW[Wb][1][2]) / nel[2],
                                   grid={str(W): float(fitW[W][1][SEL]) / nel[SEL] for W in Ws})
            T_ = torch.zeros(n, C, H, dtype=torch.float16)
            for c in torch.arange(C).split(64):
                T_[:, c[0]:c[-1] + 1] = tpl_chunk(E[:, c].to(dev), P * q, Wb).half().cpu()
            EXT[nm] = (T_, torch.arange(n))
            print(f'[tpl] {nm} P {P} W {Wb} test {res[f"tpl_{nm}"]["test_mse"]:.5f} val grid '
                  + ' '.join(f'{k}:{v:.5f}' for k, v in res[f'tpl_{nm}']['grid'].items()), flush=True)
    k_te0 = int(torch.nonzero(split == 2).min())
    n_te = int((split == 2).sum())
    MECH = {}
    def roll_stack(o, sub, ix, key):
        """Causal rolling stack: at origin t, per lead bucket, least squares over the last W val/test origins j whose
        lead-h' target is observed (j + h' <= t - 1), pooled over the bucket's leads; W chosen on val_select."""
        rows = o['rows'].to(dev)
        tr = times.to(dev)[rows]                                                   # origin times of the stored rows
        p_ = len(sub)
        RG = o['RG'][:, :, ix][:, :, :, ix]
        Rg = o['Rg'][:, :, ix]
        CG = torch.cat([torch.zeros_like(RG[:1]), RG.cumsum(0)])                   # (R + 1) H p p
        Cg = torch.cat([torch.zeros_like(Rg[:1]), Rg.cumsum(0)])
        hh = torch.arange(H, device=dev)
        pos = torch.searchsorted(tr, tr[:, None] - hh[None] - 1, right=True)     # R H: # rows with j <= t - h' - 1
        own_G = torch.zeros(len(rows), NB, p_, p_, dtype=torch.float64, device=dev).index_add_(1, bk, RG)
        own_g = torch.zeros(len(rows), NB, p_, dtype=torch.float64, device=dev).index_add_(1, bk, Rg)
        own_e = torch.zeros(len(rows), NB, dtype=torch.float64, device=dev).index_add_(1, bk, o['Re'])
        zr = split.to(dev)[rows]
        out = {}
        for W in ROLL_W:
            lo = (pos - W).clamp_min(0)
            G = torch.zeros(len(rows), NB, p_, p_, dtype=torch.float64, device=dev)
            g = torch.zeros(len(rows), NB, p_, dtype=torch.float64, device=dev)
            for h0 in range(0, H, 48):                                             # memory: chunks of leads
                hs = hh[h0:h0 + 48]
                G.index_add_(1, bk[hs], CG[pos[:, hs], hs] - CG[lo[:, hs], hs])
                g.index_add_(1, bk[hs], Cg[pos[:, hs], hs] - Cg[lo[:, hs], hs])
            lam = 1e-6 * G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(EPS)
            cf = torch.linalg.solve(G + lam[..., None, None] * torch.eye(p_, device=dev, dtype=torch.float64),
                                    g[..., None])[..., 0]                         # R NB p
            sse = own_e - 2 * (cf * own_g).sum(-1) + torch.einsum('rbi,rbij,rbj->rb', cf, own_G, cf)   # R NB
            out[W] = {z: sse[zr == z].sum(0) for z in (1, 2, 3)}
        Wb = min(ROLL_W, key=lambda W: float(out[W][SEL].sum()))
        rk = 'roll' + key[5:]
        res[rk] = dict(W=Wb, val_mse=float(out[Wb][SEL].sum()) / nel[SEL], test_mse=float(out[Wb][2].sum()) / nel[2],
                       test_bucket_gain=(100 * (ee[2] - out[Wb][2]) / ee[2].clamp_min(EPS)).tolist(),
                       grid={str(W): dict(val=float(out[W][SEL].sum()) / nel[SEL], test=float(out[W][2].sum()) / nel[2])
                             for W in ROLL_W})
        print(f'[roll]  {rk:28s} test {res[rk]["test_mse"]:.5f} val {res[rk]["val_mse"]:.5f} W {Wb} | bucket gain% vs base_sh '
              + ' '.join(f'{x:+.2f}' for x in res[rk]['test_bucket_gain']), flush=True)

    AGE_BINS = ((0, 1), (1, 6), (6, 24), (24, 72), (72, 168), (168, None))      # injection age in steps (hours)

    def age_decomp(net, name, b):
        """Exact split of the carry correction by injection age.  T_t depends on x_t only (not on z), so
        z_t = sum_k Phi(t, k) B w_k: track every age 0..167 separately plus one remainder (>= 168); readout is linear.
        Uses the trained carry and its val_fit coefficient b (per bucket): only-bin and leave-bin-out SSE."""
        assert net.kind in ('P', 'PF', 'PD') and net.nh == 1 and net.rot
        A = 168
        call = torch.arange(C, device=dev)
        wd = wdiag[net.nb].to(dev)
        k_s = int(torch.nonzero(split >= 1).min()) - A - 2
        bb = torch.tensor(b, device=dev, dtype=torch.float64)[bk]                  # H
        nbin = len(AGE_BINS)
        acc_ = {key: torch.zeros(4, NB, dtype=torch.float64, device=dev) for key in ['full'] + [f'only{i}' for i in range(nbin)]
                + [f'loo{i}' for i in range(nbin)]}
        n_wk = (n - k_te0) // 168 + 1
        wk = {key: torch.zeros(n_wk, dtype=torch.float64, device=dev) for key in ['ee'] + list(acc_)}
        rho = torch.sigmoid(net.rho_p)
        c_, s_ = rho * torch.cos(net.theta), rho * torch.sin(net.theta)
        state, Z, Rm = None, None, None
        err = 0.0
        with torch.no_grad():
            for k in range(n):
                if k < k_s:
                    state, _ = step(net, state, k, call, wd)
                    continue
                if Z is None:                                                      # everything injected before k_s
                    Rm = state.clone() if state is not None else torch.zeros(C, net.d, 2, device=dev)
                    Z = torch.zeros(A, C, net.d, 2, device=dev)
                x = x_at(k, call)
                u, beta = net.u_beta(x)                                            # (C 2d), (C 1)
                def T(z_):                                                         # z_ (..., C, d, 2): T_t then Lam
                    zf = z_.flatten(-2)
                    if net.kind == 'PD':
                        zf = zf * (1 - beta * u.square())
                    else:
                        zf = zf - beta * u * (u * zf).sum(-1, keepdim=True)
                    z_ = zf.view(z_.shape)
                    return torch.stack([c_ * z_[..., 0] - s_ * z_[..., 1], s_ * z_[..., 0] + c_ * z_[..., 1]], -1)
                Z, Rm = T(Z), T(Rm)
                Rm = Rm + Z[A - 1]
                Z = torch.roll(Z, 1, 0)
                sg = sig[k, call][:, None] ** net.g()
                w = wd[k - 1] / sg if k >= 1 else torch.zeros(C, net.nb, device=dev)
                Z[0] = (w @ net.B.T).view(C, net.d, 2)
                state, m = step(net, state, k, call, wd)                           # reference path
                comp = [Z[lo:hi].sum(0) if hi is not None else Rm for lo, hi in AGE_BINS]
                err = max(err, float((sum(comp) - state).abs().max()))
                z = int(split[k])
                if z < 0:
                    continue
                e_ = E[k].to(dev).double()
                mb = [(sg * net.read(c, H)).double() * bb for c in comp]          # C H, scaled by b
                mf = sum(mb)
                vals = {'full': (e_ - mf).square()}
                for i, m_ in enumerate(mb):
                    vals[f'only{i}'] = (e_ - m_).square()
                    vals[f'loo{i}'] = (e_ - (mf - m_)).square()
                for key, v in vals.items():
                    acc_[key][z] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, v.sum(0))
                if z == 2:
                    w_ = (k - k_te0) // 168
                    wk['ee'][w_] += e_.square().sum()
                    for key, v in vals.items():
                        wk[key][w_] += v.sum()
        print(f'[age] {name}: max |sum of age parts - state| {err:.2e}', flush=True)
        out = {}
        for zn, z in (('val_select', SEL), ('test', 2)):
            g = lambda key: float(100 * (ee[z].sum() - acc_[key][z].sum()) / ee[z].sum())
            gf = g('full')
            out[zn] = dict(full=gf, only=[g(f'only{i}') for i in range(nbin)], drop=[gf - g(f'loo{i}') for i in range(nbin)],
                           bucket_full=(100 * (ee[z] - acc_['full'][z]) / ee[z].clamp_min(EPS)).tolist(),
                           bucket_drop=[(100 * (acc_[f'loo{i}'][z] - acc_['full'][z]) / ee[z].clamp_min(EPS)).tolist()
                                        for i in range(nbin)])
            print(f'[age] {name} {zn:10s} full {gf:+.3f}% | only ' + ' '.join(f'{x:+.3f}' for x in out[zn]['only'])
                  + ' | drop(leave-out) ' + ' '.join(f'{x:+.3f}' for x in out[zn]['drop']), flush=True)
        okw = wk['ee'] > 0
        out['week_drop_pos'] = [float(((wk[f'loo{i}'] - wk['full'])[okw] > 0).double().mean()) for i in range(nbin)]
        out['week_full_pos'] = float(((wk['ee'] - wk['full'])[okw] > 0).double().mean())
        out['bins'] = [list(x) for x in AGE_BINS]
        print(f'[age] {name} test weeks: full>0 {out["week_full_pos"]:.2f} | drop>0 per bin '
              + ' '.join(f'{x:.2f}' for x in out['week_drop_pos']), flush=True)
        res[f'age_{name}'] = out

    def src_pass(net, name, b):
        """--src: where the carry's gain comes from.  Pass 1 (exact replica of eval_carry, checked against its test MSE) stores
        per (val/test origin, channel): gain d = sum_h e^2 - (e - b m)^2 (all leads, leads < 24), and the transition
        statistics of T_t: beta, |T_t z - z| (hha), / |z| (hh), cos(u_t, u_t-1), |z|.  Then causal ablations of the trained
        net: T_t -> I at chosen val/test cells (state, readout, b unchanged; b also refit on val_fit)."""
        assert net.kind in ('P', 'PF') and net.nh == 1
        call = torch.arange(C, device=dev)
        wd = wdiag[net.nb].to(dev)
        rows = torch.nonzero(split >= 1).flatten()
        rpos = torch.full((n,), -1, dtype=torch.long)
        rpos[rows] = torch.arange(len(rows))
        k_v0 = int(rows.min())
        bt = torch.tensor(b, device=dev, dtype=torch.float64)
        bb = bt[bk]
        st = {key: torch.zeros(len(rows), C) for key in ('d_all', 'd_s', 'ee', 'beta', 'hha', 'hh', 'ucos', 'zn')}
        rho = torch.sigmoid(net.rho_p)
        c_, s_ = rho * torch.cos(net.theta), rho * torch.sin(net.theta)
        z0 = [None]

        UU = torch.zeros(C, 2 * net.d, 2 * net.d, device=dev)                      # per-channel sum u u^T over val rows
        umean = [None]
        gs = torch.Generator(device=dev).manual_seed(seed)

        def run_(mask=None, rec=False, umode=None):
            """mask (n C) bool, cpu: True -> T_t = I at that cell, or with umode the Householder at that cell keeps beta_t
            but uses another unit u: 'shuf' u of the input at a random other time (25..335 h back, one draw per step),
            'rand' random direction, 'mean' the channel's principal val u (top eigvec of sum u u^T; fixed, Pfix-like), 'z' z / |z| (radial shrink).
            Returns {'ex','xx'} per split x bucket."""
            acc_ = {key: z3() for key in ('ex', 'xx')}
            k0 = 0 if rec else k_v0
            state = torch.zeros(C, net.d, 2, device=dev) if rec else z0[0].clone()
            u_prev = None
            with torch.no_grad():
                for k in range(k0, n):
                    if rec and k == k_v0:
                        z0[0] = state.clone()
                    x = x_at(k, call)
                    zT = net.trans(state, x)
                    r = int(rpos[k])
                    if rec and r >= 0:
                        u, beta = net.u_beta(x)
                        zf = state.flatten(1)
                        zn = zf.norm(dim=-1)
                        hha = beta[:, 0] * (u * zf).sum(-1).abs()                  # |T_t z - z| (|u| = 1)
                        st['beta'][r], st['hha'][r], st['zn'][r] = beta[:, 0].cpu(), hha.cpu(), zn.cpu()
                        st['hh'][r] = (hha / zn.clamp_min(EPS)).cpu()
                        if u_prev is not None:
                            st['ucos'][r] = (u * u_prev).sum(-1).cpu()
                        u_prev = u
                        if int(split[k]) in (1, 3):
                            UU.add_(u[:, :, None] * u[:, None, :])
                    if mask is not None and umode is None:
                        zT = torch.where(mask[k].to(dev)[:, None, None], state, zT)
                    elif mask is not None:
                        u, beta = net.u_beta(x)
                        zf = state.flatten(1)
                        if umode == 'shuf':
                            ks = k - int(torch.randint(25, 336, (1,), generator=gs, device=dev))
                            u2 = net.u_beta(x_at(ks, call))[0]
                        elif umode == 'rand':
                            u2 = F.normalize(torch.randn(C, 2 * net.d, generator=gs, device=dev), dim=-1)
                        elif umode == 'mean':
                            if umean[0] is None:                                   # top eigvec of E[u u^T] (sign-free)
                                umean[0] = torch.linalg.eigh(UU.double())[1][..., -1].float()
                            u2 = umean[0]
                        else:
                            u2 = zf / zf.norm(dim=-1, keepdim=True).clamp_min(EPS)
                        zA = (zf - beta * u2 * (u2 * zf).sum(-1, keepdim=True)).view_as(state)
                        zT = torch.where(mask[k].to(dev)[:, None, None], zA, zT)
                    z_ = torch.stack([c_ * zT[..., 0] - s_ * zT[..., 1], s_ * zT[..., 0] + c_ * zT[..., 1]], -1)
                    sg = sig[k, call][:, None] ** net.g()
                    w = wd[k - 1] / sg if k >= 1 else torch.zeros(C, net.nb, device=dev)
                    state = z_ + (w @ net.B.T).view(C, net.d, 2)
                    m = sg * net.read(state, H)
                    z = int(split[k])
                    if z < 0:
                        continue
                    e_, x_ = E[k].to(dev).double(), m.double()
                    acc_['ex'][z] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, (e_ * x_).sum(0))
                    acc_['xx'][z] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, (x_ * x_).sum(0))
                    if rec:
                        d = e_.square() - (e_ - bb * x_).square()
                        st['d_all'][r], st['d_s'][r] = d.sum(-1).float().cpu(), d[:, :24].sum(-1).float().cpu()
                        st['ee'][r] = e_.square().sum(-1).float().cpu()
            return acc_

        def gains(a_):
            out = {}
            b2 = torch.where(a_['xx'][1] > 0, a_['ex'][1] / a_['xx'][1].clamp_min(EPS), torch.zeros_like(bt))
            for tag, bc in (('fixb', bt), ('refit', b2)):
                sse = ee - 2 * bc * a_['ex'] + bc ** 2 * a_['xx']
                out[tag] = {zn: float(100 * (ee[z].sum() - sse[z].sum()) / ee[z].sum()) for zn, z in (('val_select', SEL), ('test', 2))}
            return out

        full = gains(run_(rec=True))
        ref = 100 * (ee[2].sum() - res[name]['test_mse'] * nel[2]) / ee[2].sum()
        print(f'[src] {name}: replica test gain {full["fixb"]["test"]:+.4f}% vs eval_carry {float(ref):+.4f}%', flush=True)
        assert abs(full['fixb']['test'] - float(ref)) < 1e-3
        out = {'full': full}
        vr = (split[rows] == 1) | (split[rows] == 3)                               # val rows: thresholds from val only
        g = torch.Generator().manual_seed(seed)
        for tag, q_ in (('top10', 0.9),) if args.src_dir else (('top10', 0.9), ('top30', 0.7)):
            th = torch.quantile(st['hha'][vr].flatten()[torch.randperm(int(vr.sum()) * C, generator=g)[:4_000_000]], q_)
            sel = st['hha'] >= th                                                  # (rows C)
            frac = float(sel.float().mean())
            for kind_, mk in ((tag, sel), (f'rand{tag[3:]}', torch.rand(sel.shape, generator=g) < frac), (f'keep{tag[3:]}', ~sel)):
                M = torch.zeros(n, C, dtype=torch.bool)
                M[rows] = mk
                if args.src_dir and kind_.startswith('keep'):
                    continue
                out[kind_] = dict(frac=float(mk.float().mean()), **gains(run_(M)))
                if args.src_dir:                                                   # same cells, other directions
                    for um in ('shuf', 'rand', 'mean', 'z'):
                        out[f'{kind_}_u{um}'] = dict(frac=float(mk.float().mean()), **gains(run_(M, umode=um)))
        if not args.src_dir:
            M = torch.zeros(n, C, dtype=torch.bool)
            M[rows] = True
            out['noT'] = dict(frac=1.0, **gains(run_(M)))
        for key, v in out.items():
            print(f'[src] {name} {key:12s} frac {v.get("frac", 0):.3f} | test fixb {v["fixb"]["test"]:+.3f} refit {v["refit"]["test"]:+.3f}'
                  f' | val fixb {v["fixb"]["val_select"]:+.3f} refit {v["refit"]["val_select"]:+.3f}', flush=True)
        res[f'src_{name}'] = out
        tt = T0 + rows
        if args.src_dir:
            return
        np.savez(od / f'{code}_H{H}_s{seed}_{name}_src.npz', rows=rows.numpy(), times=tt.numpy(), split=split[rows].numpy(),
                 dmu=dmu[rows.to(dev)].float().cpu().numpy(), dls=dls[rows.to(dev)].float().cpu().numpy(),
                 **{key: v.numpy() for key, v in st.items()})

    for name in arms:
        nb, kind, nh, *rest = CARRY_ARMS[name]
        rot = rest[0] if rest else True
        t_start = time.time()
        if args.load_carry:                                                        # reuse a trained carry (--dump state)
            torch.manual_seed(seed)
            net = ShockSSM(nx, nb, kind, q, D_MODES, nh, rot).to(dev)
            ld = torch.load(Path(args.load_carry) / f'{code}_H{H}_s{seed}_{name}_dump.pt', map_location=dev, weights_only=False)
            net.load_state_dict(ld['state'])
            net.eval()
            log = ld['train_loss']
        else:
            net, log = train_carry(nb, kind, nh, rot)
        t_train = time.time() - t_start
        n_par = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
        dump = None
        if args.dump:
            dump = {key: [] for key in ('beta', 'hh_rel', 'u_cos', 'gate', 'gate_min', 'phase', 'm_rms', 'rho_rms')}
            init = ShockSSM(nx, nb, kind, q, D_MODES, nh, rot)                         # same deterministic theta / rho init
        mech = None
        if args.mech and name in args.mech.split(','):
            mech = {key: torch.zeros(D_MODES, dtype=torch.float64, device=dev)
                    for key in ('n', 'dlogamp', 'dphase_abs', 'w_dlogamp', 'w_dphase_abs', 'dphase_cos', 'dphase_sin', 'amp2_pre', 'amp2_post',
                                'self2', 'cross2')}
            mech['MX'] = torch.zeros(n_te, C, H)                               # fp32: exact decomposition
        o = eval_carry(net, dump, mech)
        if mech is not None:
            MECH[name] = mech
            MECH[name]['theta'] = net.theta.detach().double().cpu()
            MECH[name]['rho'] = torch.sigmoid(net.rho_p).detach().double().cpu()
        if args.dump:
            dump = {key: torch.stack(v) for key, v in dump.items() if v}   # (test steps, C)
            torch.save(dict(state=net.state_dict(), theta0=init.theta.detach().clone(), rho0=init.rho_p.detach().clone(),
                            q=q, PW=PW, train_loss=log, **dump), od / f'{code}_H{H}_s{seed}_{name}_dump.pt')
        b, sse = fit_alone(o)
        res[name] = dict(b=b.tolist(), val_mse=float(sse[SEL]) / nel[SEL], test_mse=float(sse[2]) / nel[2],
                         train_loss=log, gamma=float(net.g()), n_params=n_par, train_s=t_train)
        for a_ in S_ARMS:
            a, b2, sse2 = fit_joint(o, a_)
            res[f'{a_}(+){name}'] = dict(a=a.tolist(), b=b2.tolist(), val_mse=float(sse2[SEL]) / nel[SEL],
                                         test_mse=float(sse2[2]) / nel[2])
        print(f'[{name}] test {res[name]["test_mse"]:.5f} | S_r(+) {res[f"S_r(+){name}"]["test_mse"]:.5f}', flush=True)
        if args.age:
            age_decomp(net, name, res[name]['b'])
        if args.src:
            src_pass(net, name, res[name]['b'])
        if EXT:                                                                    # per-lead-bucket stacking, fit on val_fit
            names = ['S_e', 'P'] + list(EXT)
            T3 = ['Rm', 'Dm', 'Wm']
            subsets = [['FC'], ['B'], ['S_e'], ['P'], ['FC', 'P'], ['FC', 'S_e'], ['FC', 'S_e', 'P'], ['B', 'S_e'],
                       ['B', 'S_e', 'P'], ['FC', 'B', 'S_e'], ['FC', 'B', 'S_e', 'P'],
                       ['Rm'], ['Dm'], ['Wm'], ['Dm', 'Wm'], T3, T3 + ['P'], T3 + ['S_e'], T3 + ['S_e', 'P'],
                       T3 + ['B'], T3 + ['B', 'S_e', 'P'], T3 + ['FC', 'S_e', 'P'], T3 + ['FC', 'B', 'S_e', 'P']]
            for x in EXT:                                                          # eval_pathret arms, B window variants
                if x not in ('FC', 'B', 'Rm', 'Dm', 'Wm'):
                    subsets += [[x], [x, 'S_e'], [x, 'S_e', 'P'], [x, 'B', 'S_e', 'P']]
            for sub in subsets:
                if any(x not in names for x in sub):
                    continue
                ix = torch.tensor([names.index(x) for x in sub], device=dev)
                G1, g1 = o['G'][1][:, ix][:, :, ix], o['g'][1][:, ix]
                lam = 1e-9 * G1.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(EPS)
                cf = torch.linalg.solve(G1 + lam[:, None, None] * torch.eye(len(sub), device=dev, dtype=torch.float64),
                                        g1[..., None])[..., 0]                    # NB p
                sse = torch.stack([ee[z] - 2 * (cf * o['g'][z][:, ix]).sum(-1) +
                                   torch.einsum('bi,bij,bj->b', cf, o['G'][z][:, ix][:, :, ix], cf) for z in range(4)])
                key = 'stack[' + '+'.join(x if x != 'P' else name for x in sub) + ']'
                res[key] = dict(coef={x: cf[:, i].tolist() for i, x in enumerate(sub)}, val_mse=float(sse[SEL].sum()) / nel[SEL],
                                test_mse=float(sse[2].sum()) / nel[2],
                                test_bucket_gain=(100 * (ee[2] - sse[2]) / ee[2].clamp_min(EPS)).tolist())   # vs base_sh
                print(f'[stack] {key:28s} test {res[key]["test_mse"]:.5f} val {res[key]["val_mse"]:.5f} | bucket gain% vs base_sh '
                      + ' '.join(f'{x:+.2f}' for x in res[key]['test_bucket_gain']), flush=True)
                if args.roll:
                    roll_stack(o, sub, ix, key)

    if MECH:                                                                       # --mech: correction diagnostics, test
        LH = sorted({0, min(23, H - 1), H // 2 - 1, H - 1})
        meths = {'base': None, 'S_e': ('S_e', None)}
        for nm in MECH:
            meths[nm] = (None, nm)
            meths[f'S_e+{nm}'] = ('S_e', nm)
        coef = {}
        for mk, v in meths.items():
            if v is None:
                continue
            sa, xa = v
            if sa and xa:
                r_ = res[f'S_e(+){xa}']
                coef[mk] = (torch.tensor(r_['a'], device=dev)[bk], torch.tensor(r_['b'], device=dev)[bk])
            elif sa:
                coef[mk] = (torch.tensor(res['S_e']['a'], device=dev)[bk], None)
            else:
                coef[mk] = (None, torch.tensor(res[xa]['b'], device=dev)[bk])
        z2 = lambda: torch.zeros(C, H, dtype=torch.float64, device=dev)
        sm = {f'{mk}_{t_}': z2() for mk in meths for t_ in ('rd_tot', 'dd_tot', 'rd_inc', 'dd_inc')}
        sm['rr_tot'], sm['rr_inc'] = z2(), z2()
        ser = {key: torch.zeros(n_te, C, len(LH), dtype=torch.float16) for key in ['r_inc', 'd_base'] +
               [f'd_{mk}' for mk in meths if mk != 'base']}
        with torch.no_grad():
            for i in range(0, n_te, 64):
                kk = torch.arange(k_te0 + i, min(k_te0 + n_te, k_te0 + i + 64))
                rho_ = E[kk].to(dev).double()
                bp = basep(kk, torch.arange(C)).double()
                rt = rho_ + bp
                sm['rr_tot'] += rt.square().sum(0)
                sm['rr_inc'] += rho_.square().sum(0)
                ser['r_inc'][i:i + len(kk)] = rho_[..., LH].half().cpu()
                ser['d_base'][i:i + len(kk)] = bp[..., LH].half().cpu()
                for mk in meths:
                    di = torch.zeros_like(rho_)
                    if mk in coef:
                        ca, cb_ = coef[mk]
                        if ca is not None:
                            di = di + ca * MS['S_e'][kk].to(dev).double()
                        if cb_ is not None:
                            di = di + cb_ * MECH[meths[mk][1]]['MX'][kk - k_te0].to(dev).double()
                    dt = bp + di
                    sm[f'{mk}_rd_tot'] += (rt * dt).sum(0)
                    sm[f'{mk}_dd_tot'] += dt.square().sum(0)
                    sm[f'{mk}_rd_inc'] += (rho_ * di).sum(0)
                    sm[f'{mk}_dd_inc'] += di.square().sum(0)
                    if mk != 'base':
                        ser[f'd_{mk}'][i:i + len(kk)] = di[..., LH].half().cpu()
        rr = float(sm['rr_tot'].sum())
        for mk in meths:
            g_ = 100 * (2 * float(sm[f'{mk}_rd_tot'].sum()) - float(sm[f'{mk}_dd_tot'].sum())) / rr
            ref = {'base': 'base_sh', 'S_e': 'S_e'}.get(mk, mk.replace('S_e+', 'S_e(+)'))
            print(f'[mech] {mk:12s} gain {g_:+.3f}% (json test mse {res[ref]["test_mse"]:.5f}, '
                  f'decomp {(rr - 2 * float(sm[f"{mk}_rd_tot"].sum()) + float(sm[f"{mk}_dd_tot"].sum())) / nel[2]:.5f})',
                  flush=True)
        out_m = {key: v.cpu().numpy() for key, v in sm.items()}
        out_m.update({key: v.numpy() for key, v in ser.items()})
        for nm, mm in MECH.items():
            out_m.update({f'mode_{nm}_{key}': v.cpu().numpy() for key, v in mm.items() if key != 'MX'})
        np.savez_compressed(od / f'{code}_H{H}{ltag}_s{seed}_mech.npz', leads=np.array(LH), q=q,
                            methods=np.array(list(meths)), **out_m)
        del MECH
    tefl = ev.load_tefl(code, seed) if L == 96 else None                           # TEFL / ridge refs are L96 runs
    rp = Path('patchmem/logs/server/err_value') / f'{code}_errval_s{seed}.json'
    ridge = json.loads(rp.read_text())['arms']['all']['test_mse'] if rp.exists() and L == 96 else None
    fp = Path(args.fc_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    fcj = json.loads(fp.read_text()) if fp.exists() else None
    g_ = lambda x: 100 * (base - x) / base
    for k_, v in res.items():
        if 'test_mse' not in v:                                                    # age_* diagnostics
            continue
        v['gain'] = g_(v['test_mse'])
        print(f'[test] {k_:12s} mse {v["test_mse"]:.5f} gain {v["gain"]:+.3f}% (val {v["val_mse"]:.5f})', flush=True)
    refs = dict(tefl=tefl['mse'] if tefl else None, ridge_all=ridge,
                freshcast=fcj['fc']['test_mse'] if fcj else None, B=fcj['b']['test_mse'] if fcj else None)
    print(f'[test] base {base:.5f} refs {refs}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official, refs=refs,
               shares=shares, settings=dict(q=q, PW=PW, m_tr=m_tr, NB=NB, KS=KS, EPOCHS=args.epochs, max_c=args.max_c,
                             base=args.base, bptt_h=args.bptt_h, freeze_theta=args.freeze_theta,
                             early_stop=args.early_stop, val_split=args.val_split),
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
    p.add_argument('--lr', type=float, default=LR, help='carry Adam lr')
    p.add_argument('--lr_sched', default='const', choices=('const', 'cos'))
    p.add_argument('--warmup', type=int, default=0, help='linear lr warmup over this many updates')
    p.add_argument('--ext', default='', help='stack raw external corrections (comma list, e.g. FC,B) with S_e and the carry')
    p.add_argument('--ext_dir', default='patchmem/logs/server/err_stack/fc')
    p.add_argument('--tpl', default='', help='causal running-mean error templates as stacking regressors: Rm,Dm,Wm')
    p.add_argument('--age', action='store_true', help='split the carry correction by shock injection age (P / Pfix / Pdiag)')
    p.add_argument('--src_dir', action='store_true', help='with --src: only the top10 / rand10 cells, T -> I and other u')
    p.add_argument('--src', action='store_true', help='per origin x channel gain + T_t stats of P / Pfix, T_t -> I ablations')
    p.add_argument('--se_len', type=int, default=0, help='S_e key window length (0: the backbone input length L)')
    p.add_argument('--roll', default='', help='causal rolling stack: window grid in val/test origins, e.g. 168,336,672')
    p.add_argument('--load_carry', default='', help='dir with {code}_H{H}_s{seed}_{arm}_dump.pt: load instead of training')
    p.add_argument('--zero_only', action='store_true', help='print the carry zero reference and stop')
    p.add_argument('--log_every', type=int, default=100, help='print the mean train loss of the last N updates')
    p.add_argument('--log_updates', action='store_true', help='keep per-update train loss + grad norm in train_loss')
    p.add_argument('--weeks', type=int, default=None, help='train-tail weeks (default: the whole train segment)')
    p.add_argument('--arms', default='s1_R,s1_D,s1_P')
    p.add_argument('--out_dir', default='patchmem/logs/server/err_sr')
    p.add_argument('--base', default='sh', choices=('sh', 'shrot'), help='shrot: + fixed rotary state (rot or tail on val)')
    p.add_argument('--bptt_h', type=int, default=24, help='truncated BPTT length in hours')
    p.add_argument('--freeze_theta', action='store_true')
    p.add_argument('--fc_dir', default='patchmem/logs/server/error_path', help='FreshCast / B reference jsons')
    p.add_argument('--val_split', action='store_true', help='val_fit (first half) fits b / a, val_select picks')
    p.add_argument('--early_stop', action='store_true', help='keep the carry epoch with the best val MSE')
    p.add_argument('--mech', default='', help='carry arms (comma list) for the correction / mode diagnostics')
    p.add_argument('--dump', action='store_true', help='save the trained carry + test-stream T_t statistics')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
