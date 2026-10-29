"""PAT-315: predictable-error decomposition on top of FreshCast (eval_freshcast).

  y^ = f + w^ d + a_b m_state + b_b m_x,     d = r - f (FreshCast), u = e - w^ d (residual after FreshCast)

Term 1 (state-explainable) m_state[t,h] = w2(t)^T phi[t,h], one w per channel shared by all leads, from two one-step
GD (= linear attention) updates on demos of the same channel, all with s + H <= t:
  phi[t,h] = [1, d_h, rv_h - f_h, rs_h - f_h, level_t, log scale_t, onehot hour(t+h) (24), onehot day(t+h) (7)]
             (rv / rs: FreshCast value / shape readout means under its softmax; level, scale: LQ-window mu, sd vs
             the channel's train mean / std; features 1-5 z-scored per channel, clipped to +-Z_CLIP, std floored
             at SD_Z: d, rv, rs with train-tail stats, level / scale with the stats over all train windows)
  w1 = w0 + eta1 sum_{per} sum_h phi (u - w0^T phi)     per: s = t - age, ages = 24 j and 168 j (8 each, >= H)
  w2 = w1 + eta2 sum_{rec} sum_h phi (u - w1^T phi)     rec: the N most recent origins s <= t - H
  eta_k = eta~_k / sum_demos sum_h |phi|^2 (eta~ < 2: no overshoot);  w0 = ridge on the FreshCast calibration
  origins (all channels pooled).  Overlapping demos (in both sets) take both steps.
Term 2 (history-dependent increment), value rho = u - a_b m_state (S_2's val-fitted a_b; m_state at the demo's own
time), no q/k: per channel a carried state c_t in R^(NB x 168) indexed by (lead bucket, TARGET hour-of-week):
  c' = A(x_t) c_{t-1} on all slots; the slots written by the newly matured rho_{t-H} (lead k -> slot
  (bucket(k), (t-H+k) mod 168), leads sharing a slot averaged) become lam(x_t) c' + (1 - lam) rho_{t-H} /
  sig_{t-H}^gamma, the others keep c';  read m_path[t,h] = sig_t^gamma c_t[bucket(h), (t+h) mod 168].
  A(x) = I - beta(x) u u^T on the whole NB*168 state, u = normalize(W_u x) rolled by t along the hour axis (the
  direction is learned relative to the current hour, so mixing across lead buckets carries fresh long-lead errors),
  beta = 2 sigmoid, lam = sigmoid; x_t = [LQ window (y - mu_t) / sd_t, d mu_t / sd_t, d log sd_t].
  P: full; P_I: A = I (lam-EMA phase template of the residuals); TEFL: MLP(rho_{t-H-i}, i < 7) -> H.
  Trained on the train-tail origins (targets inside the train segment), truncated BPTT.
Stream: origins T0 = n_tr - M_TR .. end, contiguous; train tail and the gaps before val / test are demos only.
Selection (val reused: S_2's a_b scales the term-2 target before a_b, b_b are refit; test is held out):
(eta~1, eta~2, N) per family and a_b, b_b (NB lead buckets, least squares) on all validation origins;
test once.  Families: S_0 (w0 only), S_per (eta2 = 0), S_rec (eta1 = 0), S_2 (both > 0); P / P_I / TEFL use S_2.
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
import patchmem.eval_freshcast as fc
from patchmem.eval_error_path import H, L

M_TR = 2016                                                          # train-tail origins (12 weeks)
NB = 8
PW = 168                                                             # carried-state slots (hour of week)
J_PER = 8
ETAS = (0.0, 0.25, 0.5, 1.0, 1.5)
NS = (24, 168, 672)
NP = 6 + 24 + 7
Z_CLIP, SD_Z = 5.0, 0.05
EPS = 1e-12
SIG_FLOOR = 0.05
K_TEFL = 7
TB = 168                                                             # truncated-BPTT window
BURN = 336
EPOCHS_P, LR_P = 20, 3e-3
EPOCHS_T, LR_T, BS_T = 10, 1e-3, 4096
CH2 = 128
CH1 = 32                                                             # channels per term-1 chunk
BO = 64


def per_ages(h, j=J_PER):
    a = {24 * i for i in range(math.ceil(h / 24), math.ceil(h / 24) + j)}
    a |= {168 * i for i in range(math.ceil(h / 168), math.ceil(h / 168) + j)}
    return sorted(a)


def configs():
    return [(e1, e2, n) for e1 in ETAS for e2 in ETAS for n in NS if e2 > 0 or n == NS[0]]


FAMILIES = {'S_0': lambda e1, e2: e1 == 0 and e2 == 0, 'S_per': lambda e1, e2: e1 > 0 and e2 == 0,
            'S_rec': lambda e1, e2: e1 == 0 and e2 > 0, 'S_2': lambda e1, e2: e1 > 0 and e2 > 0}


# ---------------------------------------------------------------- term 1
def phi(d, rvf, rsf, lev, sc, t, norm):
    """phi (b C H NP) from d, rvf, rsf (b C H), lev, sc (b C), origin times t (b); norm = (mean, std) (5 C)."""
    b, C, h = d.shape
    mu, sd = norm
    z = lambda x, i: ((x - mu[i][:, None]) / sd[i][:, None]).clamp(-Z_CLIP, Z_CLIP)
    tt = t[:, None] + torch.arange(h, device=d.device)[None]                     # target times (b h)
    oh = torch.cat([F.one_hot(tt % 24, 24), F.one_hot((tt // 24) % 7, 7)], -1).to(d.dtype)
    lv = ((lev - mu[3]) / sd[3]).clamp(-Z_CLIP, Z_CLIP)[..., None].expand(b, C, h)
    sv = ((sc - mu[4]) / sd[4]).clamp(-Z_CLIP, Z_CLIP)[..., None].expand(b, C, h)
    base = torch.stack([torch.ones_like(d), z(d, 0), z(rvf, 1), z(rsf, 2), lv, sv], -1)
    return torch.cat([base, oh[:, None].expand(b, C, h, oh.shape[-1])], -1)


def demo_sums(P, u):
    """S (b C NP) = sum_h phi u, G (b C NP NP) = sum_h phi phi^T, in float64."""
    P, u = P.double(), u.double()
    return torch.einsum('bchp,bch->bcp', P, u), torch.einsum('bchp,bchq->bcpq', P, P)


def prefix(x):
    return torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])


def mv(G, w):
    return (G @ w[..., None])[..., 0]


def gd_weights(k, PS, PG, w0, ages, h, grid):
    """w2 (b C NP) for each (eta1, eta2, N) in grid, for stream indices k (b); PS, PG prefix sums (n+1 ...) of the
    per-index demo sums S, G.  Periodic demos k - age (age >= h), recent demos max(0, k-h-N+1) .. k-h."""
    C = PS.shape[1]
    w0 = w0.double().expand(len(k), C, -1)
    idx = k[:, None] - ages[None]
    ok = (idx >= 0).double()
    idc = idx.clamp_min(0)
    Sp = ((PS[idc + 1] - PS[idc]) * ok[..., None, None]).sum(1)
    Gp = ((PG[idc + 1] - PG[idc]) * ok[..., None, None, None]).sum(1)
    g1 = Sp - mv(Gp, w0)
    tr1 = Gp.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(EPS)[..., None]
    hi = (k - h + 1).clamp_min(0)
    rec = {}
    out = {}
    for e1, e2, n in grid:
        w1 = w0 + e1 * g1 / tr1
        if e2 == 0:
            out[e1, e2, n] = w1
            continue
        if n not in rec:
            lo = torch.minimum((k - h - n + 1).clamp_min(0), hi)
            GG = PG[hi] - PG[lo]
            rec[n] = PS[hi] - PS[lo], GG, GG.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(EPS)[..., None]
        SS, GG, tr2 = rec[n]
        out[e1, e2, n] = w1 + e2 * (SS - mv(GG, w1)) / tr2
    return out


# ---------------------------------------------------------------- term 2
def lead_bucket(h, nb, dev):
    return torch.arange(h, device=dev) * nb // h


class Carry(nn.Module):
    """(lead bucket x hour-of-week)-indexed carried residual state; full=False drops the Householder transition."""
    def __init__(self, nx, full=True, pw=PW, nb=NB):
        super().__init__()
        self.full, self.pw, self.nb = full, pw, nb
        self.Wu = nn.Linear(nx, nb * pw)
        self.wb = nn.Linear(nx, 1)
        self.wl = nn.Linear(nx, 1)
        nn.init.constant_(self.wb.bias, -2.0)                                    # beta ~ 0.24 at init
        nn.init.constant_(self.wl.bias, 2.0)                                     # lam ~ 0.88 at init
        self.gamma = nn.Parameter(torch.ones(()))

    def g(self):
        return self.gamma.clamp(0, 2)

    def step(self, c, x, t, w, wm):
        """c (B nb*pw) state, x (B nx), t absolute origin time (int), w (B nb*pw) write values (already
        / sig^gamma), wm (nb*pw) bool written slots."""
        if self.full:
            u = self.Wu(x).view(-1, self.nb, self.pw).roll(t % self.pw, -1).flatten(1)   # u[b,p] = u_rel[b,(p-t)%pw]
            u = F.normalize(u, dim=-1)
            beta = 2 * torch.sigmoid(self.wb(x))
            c = c - beta * u * (u * c).sum(-1, keepdim=True)
        lam = torch.sigmoid(self.wl(x))
        return torch.where(wm, lam * c + (1 - lam) * w, c), lam, (beta if self.full else None)


def write_vals(rho, t_src, h, pw, nb):
    """rho (B h) of origin t_src -> (B nb*pw) slot means and the written mask; lead k -> slot
    bucket(k) * pw + (t_src + k) % pw."""
    dev = rho.device
    sl = lead_bucket(h, nb, dev) * pw + (t_src + torch.arange(h, device=dev)) % pw
    cnt = torch.zeros(nb * pw, device=dev).index_add_(0, sl, torch.ones(h, device=dev))
    w = torch.zeros(rho.shape[0], nb * pw, device=dev, dtype=rho.dtype).index_add_(1, sl, rho)
    return w / cnt.clamp_min(1), cnt > 0


def read_slots(t, h, pw, nb, dev):
    return lead_bucket(h, nb, dev) * pw + (t + torch.arange(h, device=dev)) % pw


def carry_step(model, c, k, T0, rho_at, x_at, sig_at, h):
    """One stream index k (time T0 + k): write rho of index k - h (if >= 0), then read.  rho_at(k) (B h),
    x_at(k) (B nx), sig_at(k) (B).  Returns the new state (B nb*pw) and m (B h)."""
    pw, nb = model.pw, model.nb
    t = T0 + k
    x = x_at(k)
    B = x.shape[0]
    if c is None:
        c = torch.zeros(B, nb * pw, device=x.device)
    g = model.g()
    if k - h >= 0:
        w, wm = write_vals(rho_at(k - h) / sig_at(k - h)[:, None] ** g, t - h, h, pw, nb)
    else:
        w, wm = torch.zeros(B, nb * pw, device=x.device), torch.zeros(nb * pw, dtype=torch.bool, device=x.device)
    c, _, _ = model.step(c, x, t, w, wm)
    return c, sig_at(k)[:, None] ** g * c[:, read_slots(t, h, pw, nb, x.device)]


class Tefl(nn.Module):
    def __init__(self, h, k=K_TEFL, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k * h, hid), nn.GELU(), nn.Linear(hid, h))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- run
def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    if args.smoke:
        fc.N_TRAIN_Q, fc.EPOCHS = 2048, 1
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    Ctot = X.shape[1]
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_all = torch.arange(n_tr, n_tr + n_va - H + 1)
    val_o = val_all[torch.randperm(len(val_all), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]]
    val_o = val_o.sort().values
    eps, ur = fc.untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    model, rkr_log = fc.train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    chans = torch.arange(Ctot if args.max_c is None else args.max_c)
    C = len(chans)
    T0 = n_tr - M_TR
    n = len(X) - H + 1 - T0                                                       # stream length
    times = torch.arange(T0, T0 + n)
    split = torch.full((n,), -1)                                                  # 0 train tail, 1 val, 2 test
    split[times <= n_tr - H] = 0
    split[(times >= n_tr) & (times <= n_tr + n_va - H)] = 1
    split[times >= n_tr + n_va] = 2
    assert int((split == 2).sum()) == len(X) - H + 1 - n_tr - n_va
    is_cal = torch.zeros(n, dtype=torch.bool)
    is_cal[val_o - T0] = True
    kh = torch.arange(H, device=dev)
    mu_w, sd_w = ws

    # ---- stream: forecasts and FreshCast readouts, stored on the CPU
    E = torch.zeros(n, C, H)
    RV, RS = torch.zeros(n, C, H), torch.zeros(n, C, H)
    PV = torch.zeros(n, C)
    ex = torch.zeros(4, 3, dtype=torch.float64)                                   # (split 0..2, cal) x (ee, ed, dd)
    with torch.no_grad():
        for i in range(0, n, args.forecast_bs):
            ob = times[i:i + args.forecast_bs]
            Fb = fc.forecast(exp, Xc, Mc, ob)
            for j, t0 in enumerate(ob.tolist()):
                k = i + j
                for c in fc.chunks(C, dev):
                    t = torch.full_like(c, t0)
                    sp = fc.build(X, ws, sdesc, order, t, c)
                    R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
                    pi = model(sp['z'], sp['u'], sp['valid'])
                    vm = sp['val'].float()
                    sm = (~sp['val'] & sp['valid']).float()
                    f = Fb[j][:, c].T.float()
                    y = X[t[:, None] + kh, c[:, None]]
                    r = (pi[..., None] * R).sum(1)
                    pv = (pi * vm).sum(1)
                    ps = (pi * sm).sum(1)
                    # weighted readout means; the unweighted fallback only where the mass is exactly 0, so that
                    # pv rv + ps rs == r (pv + ps == 1: invalid candidates have pi == 0)
                    rv = torch.where((pv > 0)[:, None], ((pi * vm)[..., None] * R).sum(1) / pv.clamp_min(1e-30)[:, None],
                                     (vm[..., None] * R).sum(1) / vm.sum(1).clamp_min(1)[:, None])
                    rs = torch.where((ps > 0)[:, None], ((pi * sm)[..., None] * R).sum(1) / ps.clamp_min(1e-30)[:, None],
                                     (sm[..., None] * R).sum(1) / sm.sum(1).clamp_min(1)[:, None])
                    e, d = (y - f).double(), (r - f).double()
                    cc = c.cpu()
                    E[k, cc], RV[k, cc], RS[k, cc], PV[k, cc] = (y - f).cpu(), (rv - f).cpu(), (rs - f).cpu(), pv.cpu()
                    row = torch.stack([e.square().sum(), (e * d).sum(), d.square().sum()]).cpu()
                    if split[k] >= 0:
                        ex[split[k]] += row
                    if is_cal[k]:
                        ex[3] += row
            if (i // args.forecast_bs) % 100 == 0:
                print(f'[stream] {i}/{n}', flush=True)
    w_hat = min(max(float(ex[3, 1] / ex[3, 2].clamp_min(EPS)), 0.0), 1.0)
    nel = {s: int((split == s).sum()) * C * H for s in (0, 1, 2)}
    base = float(ex[2, 0]) / nel[2]
    fc_exact = float(ex[2, 0] - 2 * w_hat * ex[2, 1] + w_hat ** 2 * ex[2, 2]) / nel[2]
    if args.max_c is None and abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    print(f'[fc] w_hat {w_hat:.3f} base {base:.5f} fc {fc_exact:.5f}', flush=True)
    lev = ((mu_w[T0:T0 + n] - X[:n_tr].mean(0)) / X[:n_tr].std(0).clamp_min(1e-6))[:, chans].cpu()
    sc = torch.log(sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR) / X[:n_tr].std(0).clamp_min(1e-6))[:, chans].cpu()
    tr_m = split == 0
    lv_tr = ((mu_w[fc.LQ:n_tr + 1] - X[:n_tr].mean(0)) / X[:n_tr].std(0).clamp_min(1e-6))[:, chans].cpu()
    sc_tr = torch.log(sd_w[fc.LQ:n_tr + 1].clamp_min(fc.SD_FLOOR) / X[:n_tr].std(0).clamp_min(1e-6))[:, chans].cpu()
    chunks1 = lambda: torch.arange(C).split(CH1)

    def chunk(c):
        """GPU views of channel chunk c: u, d, rvf, rsf (n c H), lev, sc (n c)."""
        rv, rs, pv = RV[:, c].to(dev), RS[:, c].to(dev), PV[:, c].to(dev)
        d = pv[..., None] * rv + (1 - pv[..., None]) * rs
        return E[:, c].to(dev) - w_hat * d, d, rv, rs, lev[:, c].to(dev), sc[:, c].to(dev)

    def chunk_norm(c, d, rv, rs):
        """(mean, std) (5 c): d, rv, rs over the train-tail origins and leads, level / scale over all train
        windows; std floored at SD_Z (z is also clipped in phi)."""
        m = tr_m.to(dev)
        st = [x[m] for x in (d, rv, rs)]
        lv, sv = lv_tr[:, c].to(dev), sc_tr[:, c].to(dev)
        mu = torch.stack([x.mean((0, 2)) for x in st] + [lv.mean(0), sv.mean(0)])
        sd = torch.stack([x.std((0, 2)) for x in st] + [lv.std(0), sv.std(0)]).clamp_min(SD_Z)
        return mu, sd

    tdev = times.to(dev)
    ages = torch.tensor(per_ages(H), device=dev)
    grid = configs()

    def phis(c_parts, norm, ks):
        u, d, rv, rs, lv, s = c_parts
        return phi(d[ks], rv[ks], rs[ks], lv[ks], s[ks], tdev[ks], norm), u[ks]

    # ---- w0: ridge on the FreshCast calibration origins, all channels pooled
    cal_k = torch.nonzero(is_cal).flatten().to(dev)
    S0 = torch.zeros(NP, dtype=torch.float64, device=dev)
    G0 = torch.zeros(NP, NP, dtype=torch.float64, device=dev)
    norms = {}
    for c in chunks1():
        parts = chunk(c)
        norms[int(c[0])] = chunk_norm(c, *parts[1:4])
        P_, u_ = phis(parts, norms[int(c[0])], cal_k)
        s_, g_ = demo_sums(P_, u_)
        S0 += s_.sum((0, 1))
        G0 += g_.sum((0, 1))
        del parts
    w0 = torch.linalg.solve(G0 + 1e-6 * G0.trace() / NP * torch.eye(NP, device=dev, dtype=torch.float64), S0)
    print(f'[w0] {[round(float(x), 4) for x in w0]}', flush=True)

    # ---- term 1 grid: per config, split, bucket sums of u m and m m; config-free u u
    bk = (torch.arange(H, device=dev) * NB // H)
    pb = lambda x: torch.zeros(x.shape[0], NB, dtype=torch.float64, device=dev).index_add_(1, bk, x.sum(1))  # (b C H) -> (b NB)
    sm = {'uu': torch.zeros(3, NB, dtype=torch.float64, device=dev),
          'um': torch.zeros(len(grid), 3, NB, dtype=torch.float64, device=dev),
          'mm': torch.zeros(len(grid), 3, NB, dtype=torch.float64, device=dev)}
    spl = split.to(dev)

    def acc_split(dst, x, ks):
        """add per-origin bucket sums x (b NB) into dst (3 NB) by split."""
        s_ = spl[ks]
        for q in (0, 1, 2):
            dst[q] += x[s_ == q].sum(0)

    def term1_chunk(c, want=None):
        parts = chunk(c)
        norm = norms[int(c[0])]
        PS = torch.zeros(n + 1, len(c), NP, dtype=torch.float64, device=dev)
        PG = torch.zeros(n + 1, len(c), NP, NP, dtype=torch.float64, device=dev)
        for i in range(0, n, BO):                                                # prefix sums, batch by batch
            ks = torch.arange(i, min(n, i + BO), device=dev)
            P_, u_ = phis(parts, norm, ks)
            s_, g_ = demo_sums(P_, u_)
            PS[i + 1:i + 1 + len(ks)] = PS[i] + s_.cumsum(0)
            PG[i + 1:i + 1 + len(ks)] = PG[i] + g_.cumsum(0)
            del s_, g_
        ms = torch.zeros(n, len(c), H, device=dev) if want is not None else None
        for i in range(0, n, BO):
            ks = torch.arange(i, min(n, i + BO), device=dev)
            P_, u_ = phis(parts, norm, ks)
            W = gd_weights(ks, PS, PG, w0, ages, H, grid if want is None else [want])
            if want is not None:
                ms[ks] = torch.einsum('bchp,bcp->bch', P_.double(), W[want]).float()
                continue
            Pd, ud = P_.double(), u_.double()
            acc_split(sm['uu'], pb(ud.square()), ks)
            for gi, key in enumerate(grid):
                m = torch.einsum('bchp,bcp->bch', Pd, W[key])
                acc_split(sm['um'][gi], pb(ud * m), ks)
                acc_split(sm['mm'][gi], pb(m.square()), ks)
        return parts, ms

    for c in chunks1():
        term1_chunk(c)
    print('[term1] grid done', flush=True)

    def fit1(um, mm, uu):
        """val-fitted a_b and the resulting (val, test) SSE."""
        a = torch.where(mm[1] > 0, um[1] / mm[1].clamp_min(EPS), torch.zeros_like(mm[1]))
        sse = uu - 2 * a * um + a ** 2 * mm
        return a, sse.sum(-1)
    res = {}
    for fam, ok in FAMILIES.items():
        cand = [gi for gi, (e1, e2, _) in enumerate(grid) if ok(e1, e2)]
        best = min(cand, key=lambda gi: float(fit1(sm['um'][gi], sm['mm'][gi], sm['uu'])[1][1]))
        a, sse = fit1(sm['um'][best], sm['mm'][best], sm['uu'])
        res[fam] = dict(cfg=grid[best], a=a.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2])
    best2 = res['S_2']['cfg']

    # ---- store u (into RV) and a_b * m_state of S_2 (into RS); term 2 sees rho = u - a_b m_state
    a_s2 = torch.tensor(res['S_2']['a'], dtype=torch.float32, device=dev)
    for c in chunks1():
        parts, ms = term1_chunk(c, want=best2)
        RV[:, c], RS[:, c] = parts[0].cpu(), (ms * a_s2[bk]).cpu()
        del parts, ms
    U, MS = RV, RS
    del E, PV

    # ---- term-2 inputs
    dmu = ((mu_w[T0:T0 + n] - mu_w[T0 - 1:T0 + n - 1]) / sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR))[:, chans]
    dls = (torch.log(sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR)) - torch.log(sd_w[T0 - 1:T0 + n - 1].clamp_min(fc.SD_FLOOR)))[:, chans]
    trd = tr_m.to(dev)
    dmu, dls = dmu / dmu[trd].std().clamp_min(1e-6), dls / dls[trd].std().clamp_min(1e-6)
    sig = sd_w[T0:T0 + n][:, chans].clamp_min(SIG_FLOOR)
    kq = torch.arange(-fc.LQ, 0, device=dev)
    cd = chans.to(dev)

    def x_at(k, c):
        t = T0 + k
        w = (X[t + kq][:, cd[c]].T - mu_w[t, cd[c]][:, None]) / sd_w[t, cd[c]].clamp_min(fc.SD_FLOOR)[:, None]
        return torch.cat([w, dmu[k, c][:, None], dls[k, c][:, None]], 1)
    nx = fc.LQ + 2
    k_tr_end = int(torch.nonzero(tr_m).max()) + 1                                 # train-tail indices 0..k_tr_end-1

    def train_carry(full):
        torch.manual_seed(seed)
        net = Carry(nx, full).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LR_P)
        log = []
        g = torch.Generator().manual_seed(seed)
        for ep in range(EPOCHS_P):
            tot, cnt = 0.0, 0
            for c in torch.randperm(C, generator=g).split(CH2):
                cg = c.to(dev)
                rho = (U[:k_tr_end, c] - MS[:k_tr_end, c]).to(dev)
                state, loss, nl = None, 0.0, 0
                for k in range(k_tr_end):
                    state, m = carry_step(net, state, k, T0, lambda q: rho[q], lambda q: x_at(q, cg),
                                          lambda q: sig[q, cg], H)
                    if k >= BURN + H:
                        loss = loss + (m - rho[k]).square().mean()
                        nl += 1
                    if (k + 1) % TB == 0 or k == k_tr_end - 1:
                        if nl:
                            opt.zero_grad()
                            (loss / nl).backward()
                            opt.step()
                            tot, cnt = tot + float(loss), cnt + nl
                        loss, nl = 0.0, 0
                        state = state.detach()
                del rho
            log.append(tot / max(cnt, 1))
            print(f'[carry{"" if full else "_I"}] epoch {ep} loss {log[-1]:.5f} gamma {float(net.g()):.3f}', flush=True)
        net.eval()
        return net, log

    def eval_sums(predict):
        """bucket sums per split for m_x from predict(): yields (k, m_x (C H)) over the whole stream."""
        out = {q: torch.zeros(3, NB, dtype=torch.float64, device=dev) for q in ('ux', 'xx', 'sx', 'ss', 'us')}
        for k, mx in predict():
            q = int(split[k])
            if q < 0:
                continue
            u_, s_ = U[k].to(dev).double(), MS[k].to(dev).double()
            mx = mx.double()
            for key, v in (('ux', u_ * mx), ('xx', mx * mx), ('sx', s_ * mx), ('ss', s_ * s_), ('us', u_ * s_)):
                out[key][q] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, v.sum(0))
        return out

    def fit2(o):
        """val least squares (a_b, b_b) on [m_state, m_x]; (val, test) SSE."""
        uu = sm['uu']
        A11, A12, A22, b1, b2 = o['ss'][1], o['sx'][1], o['xx'][1], o['us'][1], o['ux'][1]
        det = A11 * A22 - A12 ** 2
        okd = det > 1e-12 * (A11 * A22).clamp_min(EPS)
        # singular bucket: keep the single term with the larger fit gain (b1^2 / A11 vs b2^2 / A22)
        s_only = b1 ** 2 / A11.clamp_min(EPS) >= b2 ** 2 / A22.clamp_min(EPS)
        a = torch.where(okd, (A22 * b1 - A12 * b2) / det.clamp_min(EPS),
                        torch.where(s_only, b1 / A11.clamp_min(EPS), torch.zeros_like(b1)))
        b = torch.where(okd, (A11 * b2 - A12 * b1) / det.clamp_min(EPS),
                        torch.where(s_only, torch.zeros_like(b1), b2 / A22.clamp_min(EPS)))
        sse = uu - 2 * a * o['us'] - 2 * b * o['ux'] + a ** 2 * o['ss'] + 2 * a * b * o['sx'] + b ** 2 * o['xx']
        return a, b, sse.sum(-1)

    def carry_pred(net):
        def gen():
            state = None
            with torch.no_grad():
                for k in range(n):
                    state, m = carry_step(net, state, k, T0, lambda q: (U[q] - MS[q]).to(dev),
                                          lambda q: x_at(q, slice(None)), lambda q: sig[q], H)
                    yield k, m
        return gen

    for name, full in (('P', True), ('P_I', False)):
        net, log = train_carry(full)
        a, b, sse = fit2(eval_sums(carry_pred(net)))
        res[f'S_2+{name}'] = dict(a=a.tolist(), a_state=(a * a_s2.double()).tolist(), b=b.tolist(), val_mse=float(sse[1]) / nel[1],
                                   test_mse=float(sse[2]) / nel[2], gamma=float(net.g()), train_loss=log)
        print(f'[{name}] test {res[f"S_2+{name}"]["test_mse"]:.5f}', flush=True)

    # ---- TEFL control
    torch.manual_seed(seed)
    tefl = Tefl(H).to(dev)
    opt = torch.optim.Adam(tefl.parameters(), lr=LR_T)
    lag = H + torch.arange(K_TEFL)
    k_ok = torch.arange(H + K_TEFL - 1, k_tr_end)
    rho_tr = (U[:k_tr_end] - MS[:k_tr_end]).to(dev)
    pairs = torch.cartesian_prod(k_ok, torch.arange(C))
    g = torch.Generator().manual_seed(seed)
    tlog = []
    for ep in range(EPOCHS_T):
        tot = 0.0
        for idx in torch.randperm(len(pairs), generator=g).split(BS_T):
            kk, cc = pairs[idx, 0].to(dev), pairs[idx, 1].to(dev)
            inp = rho_tr[kk[:, None] - lag.to(dev)[None], cc[:, None]].flatten(1)
            loss = (tefl(inp) - rho_tr[kk, cc]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        tlog.append(tot / len(pairs))
        print(f'[tefl] epoch {ep} loss {tlog[-1]:.5f}', flush=True)
    del rho_tr
    tefl.eval()

    def tefl_pred():
        with torch.no_grad():
            for k in range(n):
                if k - H - K_TEFL + 1 < 0:
                    yield k, torch.zeros(C, H, device=dev)
                    continue
                rho = torch.stack([(U[k - int(l)] - MS[k - int(l)]) for l in lag]).to(dev)   # K C H
                yield k, tefl(rho.transpose(0, 1).flatten(1))
    a, b, sse = fit2(eval_sums(tefl_pred))
    res['S_2+TEFL'] = dict(a=a.tolist(), a_state=(a * a_s2.double()).tolist(), b=b.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                          train_loss=tlog)

    # ---- report
    fc_post = float(sm['uu'][2].sum()) / nel[2]
    g_ = lambda x: 100 * (base - x) / base
    for k_, v in res.items():
        v['gain'] = g_(v['test_mse'])
        v['vs_fc'] = 100 * (fc_exact - v['test_mse']) / fc_exact
        print(f'[test] {k_:10s} mse {v["test_mse"]:.5f} gain {v["gain"]:+.3f}% vs fc {v["vs_fc"]:+.3f}% '
              f'(val {v["val_mse"]:.5f})', flush=True)
    print(f'[test] base {base:.5f} fc {fc_exact:.5f} ({g_(fc_exact):+.3f}%) fc_post {fc_post:.5f}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               fc=dict(w=w_hat, test_mse=fc_exact, gain=g_(fc_exact), test_mse_post=fc_post), w0=w0.tolist(),
               settings=dict(M_TR=M_TR, NB=NB, ETAS=ETAS, NS=NS, ages=per_ages(H), max_c=args.max_c, smoke=args.smoke),
               rkr_loss=rkr_log, **res)
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_decomp_s{seed}{"_smoke" if args.smoke else ""}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--max_c', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
