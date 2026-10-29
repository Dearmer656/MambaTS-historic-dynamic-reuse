"""PAT-315: FreshCast (arXiv 2610.07834) reproduced on our frozen iTransformer checkpoints, full test set.

Per channel c and origin t (series standardized with the training-segment scaler):
  query       x_q = y[t-LQ:t], LQ = 96, mean mu_q, std sd_q
  window s    the LQ steps y[s-LQ:s] ending at s, mean mu_c, std sd_c
Candidates (232 per query; the paper's 4 seasonal profiles are dropped, they change MSE by -0.01..+0.07%):
  seasonal analogs  s = t - P j, P in {24, 168}, j = 1..J (J = 8); continuation y[s + h mod P] (the period after s,
                    tiled), each with the value readout (raw values) and the shape readout -> 32 candidates
  shape analogs     the N = 200 windows most cosine-similar to the query among the recent pool
                    {s : t - K <= s, s + H <= t} (K = 112 days) and Q = 500 distant windows s < t - K picked by a
                    fixed random rank; continuation y[s:s+H], shape readout
  shape readout     mu_q + sd_q / sd_c (y+ - mu_c), sd floored at SD_FLOOR, readouts clipped to +-CLIP
Relational kernel regression: 12 descriptors (weekly / daily / value flags, cosine, log age, daily and weekly phase
distance, log level gap, log scale ratio, train-segment lag-24 / lag-168 autocorrelation and log mean squared
difference; the paper's profile flag and half-life are dropped with the profiles), score = linear + 12-64-64-1 MLP,
inverse temperature exp(b0 + clip(g(u), -2, 2)) with u = (mean, std, max of the shape-analog cosines, 3 series stats),
r = sum_i softmax_i(beta s_i) y^_i.  Trained per (dataset, H, seed) on N_TRAIN_Q training-segment queries with
loss min(||r - y||^2 / (H eps_c), C_MAX); eps_c = per-channel validation MSE of the untrained rule.
Untrained rule: lam * median of the value readouts of the last M periods of P + (1 - lam) * mean of the top-k shape
analogs; P, M, k, lam chosen per channel on the validation origins.
Combination y^ = f + w (r - f), w = clip(sum e d / sum d^2, 0, 1) on VAL_ORIGINS validation origins x all channels.
Test: every test origin and channel.  Also evaluates our obs-ICL B arm (phase pool, config from the obsicl2 json:
P, readout, beta, a, c) on the same full test set.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import H, L
import patchmem.eval_obs_icl as oi

LQ = 96
PS = (24, 168)
J = 8
K_STEPS = 112 * 24
N_SHAPE = 200
Q_DIST = 500
SD_FLOOR = 1e-2
CLIP = 1000.0
C_MAX = 20.0
N_TRAIN_Q = 31000
VAL_ORIGINS = 100
EPOCHS, BATCH, LR, WD, EMA = 30, 256, 2e-3, 1e-2, 0.995
NZ, NU = 12, 6
SEAS = [(p * j, p) for p in PS for j in range(1, J + 1)]             # (age, period); 0..7 daily, 8..15 weekly
N_SEAS = len(SEAS)
UR_M = (1, 2, 4, 8)
UR_K = (5, 20, 50, 200)
UR_LAM = (0.0, 0.25, 0.5, 0.75, 1.0)
CH = 128                                                             # channels per chunk


def win_stats(X):
    """mu[s], sd[s] of X[s-LQ:s] for s in [0, T]; NaN for s < LQ; sd is the raw population std."""
    Xd = X.double()
    z = torch.zeros(1, X.shape[1], dtype=torch.float64, device=X.device)
    c1, c2 = torch.cat([z, Xd.cumsum(0)]), torch.cat([z, Xd.square().cumsum(0)])
    mu = torch.full((len(X) + 1, X.shape[1]), float('nan'), dtype=torch.float64, device=X.device)
    sd = mu.clone()
    m = (c1[LQ:] - c1[:-LQ]) / LQ
    mu[LQ:] = m
    sd[LQ:] = ((c2[LQ:] - c2[:-LQ]) / LQ - m.square()).clamp_min(0).sqrt()
    return mu.float(), sd.float()


def series_desc(Xtr):
    """Per channel (lag-24 acf, lag-168 acf, log mean squared difference) on the training segment."""
    def acf(l):
        a, b = Xtr[l:] - Xtr[l:].mean(0), Xtr[:-l] - Xtr[:-l].mean(0)
        return (a * b).mean(0) / (a.std(0, unbiased=False) * b.std(0, unbiased=False)).clamp_min(1e-8)
    msd = (Xtr[1:] - Xtr[:-1]).square().mean(0)
    return torch.stack([acf(24), acf(168), torch.log(msd + 1e-6)], -1)


def distant(order, t):
    """Q_DIST windows with the smallest fixed random rank among LQ <= s < t - K; (B Q) s and validity."""
    ok = (order[None] >= LQ) & (order[None] < (t - K_STEPS)[:, None])
    k = torch.sort(ok.float(), dim=1, descending=True, stable=True).indices[:, :Q_DIST]
    return order[k], ok.gather(1, k)


def cosine(X, ws, s, c, qc, sq):
    """Cosine between the query shape and the windows ending at s (B S); 0 for flat windows."""
    mu, sd = ws
    kq = torch.arange(-LQ, 0, device=X.device)
    W = X[s[..., None] + kq, c[:, None, None]]                                   # B S LQ
    sc = sd[s, c[:, None]]
    cos = (W @ qc[..., None])[..., 0] / (LQ * sc * sq[:, None]).clamp_min(1e-12)
    return torch.where((sc > 1e-6) & (sq[:, None] > 1e-6), cos, torch.zeros_like(cos))


def build(X, ws, sdesc, order, t, c):
    """Candidate specs of queries (t, c): s, period p (0: H-step continuation), value flag, descriptors z, u, valid."""
    mu, sd = ws
    dev = X.device
    B = len(t)
    kq = torch.arange(-LQ, 0, device=dev)
    mq, sq = mu[t, c], sd[t, c]
    qc = X[t[:, None] + kq, c[:, None]] - mq[:, None]
    rec = t[:, None] - H - torch.arange(K_STEPS - H + 1, device=dev)[None]        # s + H <= t, s >= t - K
    ds, dv = distant(order, t)
    pool = torch.cat([rec, ds], 1)
    pval = torch.cat([rec >= LQ, dv], 1)
    cos = cosine(X, ws, pool.clamp_min(LQ), c, qc, sq).masked_fill(~pval, -math.inf)
    top, ti = cos.topk(N_SHAPE, 1)
    v_sh = pval.gather(1, ti)
    s_sh = torch.where(v_sh, pool.gather(1, ti), (t - H)[:, None])               # invalid -> a causal dummy
    top = torch.where(v_sh, top, torch.zeros_like(top))
    ages = torch.tensor([a for a, _ in SEAS], device=dev)
    pers = torch.tensor([p for _, p in SEAS], device=dev)
    s_se = t[:, None] - ages[None]
    cos_se = cosine(X, ws, s_se, c, qc, sq)
    s = torch.cat([s_se, s_se, s_sh], 1)
    p = torch.cat([pers[None].expand(B, -1), pers[None].expand(B, -1), torch.zeros_like(s_sh)], 1)
    val = torch.zeros_like(s, dtype=torch.bool)
    val[:, :N_SEAS] = True
    valid = torch.cat([torch.ones(B, 2 * N_SEAS, dtype=torch.bool, device=dev), v_sh], 1)
    cs = torch.cat([cos_se, cos_se, top], 1)
    d = (t[:, None] - s).float()
    dd, dw = d % 24, d % 168
    sqf = sq.clamp_min(SD_FLOOR)[:, None]
    gap = torch.log1p((mu[s, c[:, None]] - mq[:, None]).abs() / sqf).clamp_max(10)
    ratio = torch.log(sd[s, c[:, None]].clamp_min(SD_FLOOR) / sqf).clamp(-10, 10)
    z = torch.stack([(p == 168).float(), (p == 24).float(), val.float(), cs, torch.log1p(d / 24),
                     torch.minimum(dd, 24 - dd) / 12, torch.minimum(dw, 168 - dw) / 84, gap, ratio], -1)
    z = torch.cat([z, sdesc[c][:, None].expand(-1, s.shape[1], -1)], -1)
    vs = v_sh.float()
    n = vs.sum(1).clamp_min(1)
    m1 = (top * vs).sum(1) / n
    sdv = (((top - m1[:, None]) * vs).square().sum(1) / n).sqrt()
    u = torch.cat([torch.stack([m1, sdv, top.max(1).values], -1), sdesc[c]], -1)
    return dict(s=s, p=p, val=val, z=z, u=u, valid=valid)


def readout(X, ws, t, c, s, p, val):
    """Candidate predictions y^_i (B N H): value or shape readout of each continuation, clipped."""
    mu, sd = ws
    kh = torch.arange(H, device=X.device)
    off = torch.where(p[..., None] > 0, kh % p.clamp_min(1)[..., None], kh)
    cont = X[s[..., None] + off, c[:, None, None]]
    mc, sc = mu[s, c[:, None]][..., None], sd[s, c[:, None]].clamp_min(SD_FLOOR)[..., None]
    mq, sq = mu[t, c][:, None, None], sd[t, c].clamp_min(SD_FLOOR)[:, None, None]
    return torch.where(val[..., None], cont, mq + sq / sc * (cont - mc)).clamp(-CLIP, CLIP)


class RKR(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(NZ, 1)
        self.mlp = nn.Sequential(nn.Linear(NZ, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.b0 = nn.Parameter(torch.zeros(()))
        self.bnet = nn.Sequential(nn.Linear(NU, 16), nn.GELU(), nn.Linear(16, 1))
        for k, n in (('zm', NZ), ('zs', NZ), ('um', NU), ('us', NU)):
            self.register_buffer(k, torch.zeros(n) if k.endswith('m') else torch.ones(n))

    def forward(self, z, u, valid):
        zn = (z - self.zm) / self.zs
        sc = self.lin(zn)[..., 0] + self.mlp(zn)[..., 0]
        beta = torch.exp(self.b0 + self.bnet((u - self.um) / self.us)[..., 0].clamp(-2, 2))
        return torch.softmax((beta[:, None] * sc).masked_fill(~valid, -math.inf), -1)


def untrained_parts(R):
    """med (B 8 H): median of the value readouts of the last M periods, for (P, M); sh (B 4 H): top-k shape means."""
    med = [torch.quantile(R[:, b:b + m], 0.5, dim=1) for b in (0, J) for m in UR_M]   # even M: mean of middle two
    sh = [R[:, 2 * N_SEAS:2 * N_SEAS + k].mean(1) for k in UR_K]
    return torch.stack(med, 1), torch.stack(sh, 1)


def chunks(C, dev):
    for i in range(0, C, CH):
        yield torch.arange(i, min(C, i + CH), device=dev)


@torch.no_grad()
def untrained_fit(X, ws, sdesc, order, origins):
    """Per-channel grid of the untrained rule on the given origins: eps_c and the chosen (med, sh, lam) indices."""
    C, dev = X.shape[1], X.device
    a11 = torch.zeros(len(UR_M) * 2, C, dtype=torch.float64, device=dev)
    a22 = torch.zeros(len(UR_K), C, dtype=torch.float64, device=dev)
    a12 = torch.zeros(len(UR_M) * 2, len(UR_K), C, dtype=torch.float64, device=dev)
    kh = torch.arange(H, device=dev)
    for t0 in origins.tolist():
        for c in chunks(C, dev):
            t = torch.full_like(c, t0)
            sp = build(X, ws, sdesc, order, t, c)
            R = readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
            y = X[t[:, None] + kh, c[:, None]]
            med, sh = untrained_parts(R)
            e1, e2 = (med - y[:, None]).double(), (sh - y[:, None]).double()
            a11[:, c] += e1.square().sum(-1).T
            a22[:, c] += e2.square().sum(-1).T
            a12[..., c] += torch.einsum('bmh,bkh->mkb', e1, e2)
    lam = torch.tensor(UR_LAM, dtype=torch.float64, device=dev)[:, None, None, None]
    sse = lam ** 2 * a11[None, :, None] + 2 * lam * (1 - lam) * a12[None] + (1 - lam) ** 2 * a22[None, None]
    flat = sse.reshape(-1, C)                                                    # (lam, med, sh) x C
    best = flat.argmin(0)
    eps = (flat.min(0).values / (len(origins) * H)).float().clamp_min(1e-6)
    li, rest = best // (a11.shape[0] * a22.shape[0]), best % (a11.shape[0] * a22.shape[0])
    return eps, dict(lam=lam.flatten()[li].float(), med=rest // a22.shape[0], sh=rest % a22.shape[0])


def untrained_r(R, ur, c):
    med, sh = untrained_parts(R)
    i = torch.arange(len(c), device=R.device)
    lam = ur['lam'][c][:, None]
    return lam * med[i, ur['med'][c]] + (1 - lam) * sh[i, ur['sh'][c]]


def train_rkr(X, ws, sdesc, order, n_tr, eps, seed):
    dev = X.device
    g = torch.Generator().manual_seed(seed)
    C = X.shape[1]
    lo, hi = K_STEPS + LQ, n_tr - H
    t_all = torch.randint(lo, hi + 1, (N_TRAIN_Q,), generator=g).to(dev)
    c_all = torch.randint(0, C, (N_TRAIN_Q,), generator=g).to(dev)
    specs = []
    with torch.no_grad():
        for i in range(0, N_TRAIN_Q, CH):
            specs.append(build(X, ws, sdesc, order, t_all[i:i + CH], c_all[i:i + CH]))
    sp = {k: torch.cat([x[k] for x in specs]) for k in specs[0]}
    del specs
    torch.manual_seed(seed)
    model = RKR().to(dev)
    zv = sp['z'][sp['valid']]
    model.zm.copy_(zv.mean(0))
    model.zs.copy_(zv.std(0).clamp_min(1e-3))
    model.um.copy_(sp['u'].mean(0))
    model.us.copy_(sp['u'].std(0).clamp_min(1e-3))
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(N_TRAIN_Q / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    kh = torch.arange(H, device=dev)
    log = []
    for ep in range(EPOCHS):
        perm = torch.randperm(N_TRAIN_Q, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N_TRAIN_Q, BATCH):
            b = perm[i:i + BATCH]
            t, c = t_all[b], c_all[b]
            R = readout(X, ws, t, c, sp['s'][b], sp['p'][b], sp['val'][b])
            pi = model(sp['z'][b], sp['u'][b], sp['valid'][b])
            r = (pi[..., None] * R).sum(1)
            y = X[t[:, None] + kh, c[:, None]]
            loss = ((r - y).square().mean(1) / eps[c]).clamp_max(C_MAX).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA).add_(pm, alpha=1 - EMA)
            tot += float(loss) * len(b)
        log.append(tot / N_TRAIN_Q)
        print(f'[rkr] epoch {ep} loss {log[-1]:.4f}', flush=True)
    ema.eval()
    return ema, log


def b_arm(X, ost, t, c, ages, ro, beta):
    """Our obs-ICL B (phase pool): mu_t, sd_t (B 1), v^ (B H); same formulas as eval_obs_icl.predict."""
    mu, sd = ost
    dev = X.device
    s = t[:, None] - ages[None]
    cc = c[:, None]
    kr, kh = torch.arange(-oi.R, 0, device=dev), torch.arange(H, device=dev)
    mu_s, sd_s = mu[s, cc][..., None], sd[s, cc][..., None]
    K = (X[s[..., None] + kr, cc[..., None]] - mu_s) / sd_s
    cont = X[s[..., None] + kh, cc[..., None]]
    mu_t, sd_t = mu[t, c][:, None], sd[t, c][:, None]
    kt = (X[t[:, None] + kr, c[:, None]] - mu_t) / sd_t
    vals = {'shape': (cont - mu_s) / sd_s, 'value': (cont - mu_t[..., None]) / sd_t[..., None]}
    dot = (K @ kt[..., None])[..., 0] / oi.R
    if ro == 'both':
        V, dot = torch.cat([vals['shape'], vals['value']], 1), torch.cat([dot, dot], 1)
    else:
        V = vals[ro]
    return mu_t, sd_t, (torch.softmax(beta * dot, -1)[..., None] * V).sum(1)


def load_series(exp):
    import pandas as pd
    from data_provider.data_factory import data_provider
    args = exp.args
    T = len(pd.read_csv(Path(args.root_path) / args.data_path, usecols=[0]))
    n_tr, n_te = int(T * 0.7), int(T * 0.2)
    n_va = T - n_tr - n_te
    old = args.seq_len
    args.seq_len = n_tr + n_va
    try:
        ds, _ = data_provider(args, 'test')
    finally:
        args.seq_len = old
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    M = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    assert len(X) == T and len(ds) == n_te - H + 1, (len(X), T, len(ds))
    return X, M, n_tr, n_va


@torch.no_grad()
def forecast(exp, X, M, o):
    off = torch.arange(-L, 0)
    return exp.model(X[o[:, None] + off].to(exp.device), M[o[:, None] + off].to(exp.device), None, None).float()


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    Xc, Mc, n_tr, n_va = load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1]
    ws = win_stats(X)
    sdesc = series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_o = torch.arange(n_tr, n_tr + n_va - H + 1)
    val_o = val_o[torch.randperm(len(val_o), generator=torch.Generator().manual_seed(seed))[:VAL_ORIGINS]].sort().values
    test_o = torch.arange(n_tr + n_va, len(X) - H + 1)
    eps, ur = untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    print(f'[untrained] eps mean {float(eps.mean()):.4f}', flush=True)
    model, log = train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    bjson = Path(args.out_dir) / f'{code}_H{H}_L{L}_obsicl2_s{seed}.json'
    bcfg = None
    if bjson.exists():
        g = json.loads(bjson.read_text())['families']['phase:B']['global']
        bcfg = dict(P=g['P'], readout=g['readout'], beta=g['hyper'], a=g['a'], c=g['c'], sampled_gain=g['test_gain'])
        ost = oi.rolling(X, bcfg['P'])
        b_ages = torch.as_tensor(oi.pool_cands('phase')[0], device=dev)
    kh = torch.arange(H, device=dev)

    def sweep(origins, w_fc=None, w_ur=None):
        acc = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in
               ('ee', 'ed', 'dd', 'eu', 'uu', 'fc', 'ur', 'b', 'n')}
        share = torch.zeros(3, dtype=torch.float64, device=dev)
        for i in range(0, len(origins), args.forecast_bs):
            ob = origins[i:i + args.forecast_bs]
            F = forecast(exp, Xc, Mc, ob)                                            # b H C
            for j, t0 in enumerate(ob.tolist()):
                for c in chunks(C, dev):
                    t = torch.full_like(c, t0)
                    sp = build(X, ws, sdesc, order, t, c)
                    R = readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
                    pi = model(sp['z'], sp['u'], sp['valid'])
                    r = (pi[..., None] * R).sum(1)
                    ru = untrained_r(R, ur, c)
                    y = X[t[:, None] + kh, c[:, None]].double()
                    f = F[j][:, c].T.double()
                    e, d, du = y - f, r.double() - f, ru.double() - f
                    acc['ee'] += e.square().sum()
                    acc['ed'] += (e * d).sum()
                    acc['dd'] += d.square().sum()
                    acc['eu'] += (e * du).sum()
                    acc['uu'] += du.square().sum()
                    acc['n'] += e.numel()
                    share += torch.stack([pi[:, :N_SEAS].sum(), pi[:, N_SEAS:2 * N_SEAS].sum(),
                                          pi[:, 2 * N_SEAS:].sum()]).double()
                    if w_fc is not None:
                        acc['fc'] += (e - w_fc * d).square().sum()
                        acc['ur'] += (e - w_ur * du).square().sum()
                        if bcfg is not None:
                            mt, st, vh = b_arm(X, ost, t, c, b_ages, bcfg['readout'], bcfg['beta'])
                            yb = f + bcfg['a'] * (mt.double() - f) + bcfg['c'] * st.double() * vh.double()
                            acc['b'] += (y - yb).square().sum()
        out = {k: float(v) for k, v in acc.items()}
        out['share'] = (share / share.sum()).tolist()
        return out

    with torch.no_grad():
        v = sweep(val_o)
        w_fc = min(max(v['ed'] / max(v['dd'], 1e-12), 0.0), 1.0)
        w_ur = min(max(v['eu'] / max(v['uu'], 1e-12), 0.0), 1.0)
        print(f'[val] w_fc {w_fc:.3f} w_ur {w_ur:.3f}', flush=True)
        te = sweep(test_o, w_fc, w_ur)
    n = te['n']
    base = te['ee'] / n
    if abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')

    def arm(sse):
        m = sse / n
        return dict(test_mse=m, gain=100 * (base - m) / base)
    res = dict(fc=dict(w=w_fc, w_test_opt=te['ed'] / max(te['dd'], 1e-12), **arm(te['fc'])),
               untrained=dict(w=w_ur, **arm(te['ur'])), share=te['share'])
    if bcfg is not None:
        res['b'] = dict(**bcfg, **arm(te['b']))
    for k, x in res.items():
        if isinstance(x, dict):
            print(f'[test] {k:10s} mse {x["test_mse"]:.5f} gain {x["gain"]:+.3f}%', flush=True)
    print(f'[test] base {base:.5f} share seas-value/seas-shape/shape {te["share"]}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               n_test_origins=len(test_o), val_origins=val_o.tolist(), eps_mean=float(eps.mean()), rkr_loss=log,
               settings=dict(LQ=LQ, J=J, K=K_STEPS, N=N_SHAPE, Q=Q_DIST, sd_floor=SD_FLOOR, n_train_q=N_TRAIN_Q,
                             epochs=EPOCHS), **res)
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
