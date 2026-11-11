"""PAT-320 v2 arm e: FreshCast (arXiv 2610.07834), ported from our reproduction MambaTS-pytorch/patchmem/eval_freshcast.py
(PAT-315 / PAT-319, same constants and formulas) to the multi-dataset setting.

Per dataset with D steps per day: seasonal periods (D, 7D) instead of (24, 168), recent pool K = 112 days, phase / age
descriptors measured in days, train-segment autocorrelation at lags D and 7D. Everything else as in eval_freshcast.py:
232 candidates per query (16 seasonal analogs x value / shape readout + 200 shape analogs from the recent pool and Q = 500
distant windows), relational kernel regression scorer (12 descriptors, linear + MLP score, inverse temperature from 6 query
statistics), loss min(||r - y||^2 / (H eps_c), C_MAX) with eps_c = per-channel val MSE of the untrained rule.
Multi-task version: ONE scorer trained jointly on the training segments of all datasets (N_TRAIN_Q queries per dataset,
equal per-dataset share of every batch). FreshCast reads only the raw series, so its training segment is the whole
official training split. Prediction r(t, c) = sum_i softmax_i(beta s_i) y^_i; the combination with the backbone is done
by the caller (y = f + w_d (r - f), w_d fitted on val, the same rule for every PAT-320 arm)."""
import copy, math
import torch
import torch.nn as nn

H = 720
LQ, J, N_SHAPE, Q_DIST = 96, 8, 200, 500
SD_FLOOR, CLIP, C_MAX = 1e-2, 1000.0, 20.0
N_TRAIN_Q, VAL_ORIGINS = 31000, 100
EPOCHS, BATCH, LR, WD, EMA = 30, 256, 2e-3, 1e-2, 0.995
NZ, NU = 12, 6
UR_M, UR_K, UR_LAM = (1, 2, 4, 8), (5, 20, 50, 200), (0.0, 0.25, 0.5, 0.75, 1.0)
CH = 128
DAY = {'ETTh1': 24, 'ETTh2': 24, 'ETTm1': 96, 'ETTm2': 96, 'Weather': 144, 'ECL': 24, 'Traffic': 24}


def win_stats(X):
    """mu[s], sd[s] of X[s-LQ:s] for s in [0, T]; NaN for s < LQ."""
    Xd = X.double()
    z = torch.zeros(1, X.shape[1], dtype=torch.float64, device=X.device)
    c1, c2 = torch.cat([z, Xd.cumsum(0)]), torch.cat([z, Xd.square().cumsum(0)])
    mu = torch.full((len(X) + 1, X.shape[1]), float('nan'), dtype=torch.float64, device=X.device)
    sd = mu.clone()
    m = (c1[LQ:] - c1[:-LQ]) / LQ
    mu[LQ:] = m
    sd[LQ:] = ((c2[LQ:] - c2[:-LQ]) / LQ - m.square()).clamp_min(0).sqrt()
    return mu.float(), sd.float()


class Ctx:
    """Per-dataset FreshCast state: series X [T, C] (scaled), window stats, series descriptors, fixed random order."""
    def __init__(self, X, n_tr, D):
        self.X, self.n_tr, self.D, self.K = X, n_tr, D, 112 * D
        self.ws = win_stats(X)
        Xtr = X[:n_tr]

        def acf(l):
            a, b = Xtr[l:] - Xtr[l:].mean(0), Xtr[:-l] - Xtr[:-l].mean(0)
            return (a * b).mean(0) / (a.std(0, unbiased=False) * b.std(0, unbiased=False)).clamp_min(1e-8)
        msd = (Xtr[1:] - Xtr[:-1]).square().mean(0)
        self.sdesc = torch.stack([acf(D), acf(7 * D), torch.log(msd + 1e-6)], -1)
        self.order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(X.device)
        self.ages = torch.tensor([p * j for p in (D, 7 * D) for j in range(1, J + 1)], device=X.device)
        self.pers = torch.tensor([p for p in (D, 7 * D) for j in range(1, J + 1)], device=X.device)


def distant(cx, t):
    ok = (cx.order[None] >= LQ) & (cx.order[None] < (t - cx.K)[:, None])
    k = torch.sort(ok.float(), dim=1, descending=True, stable=True).indices[:, :Q_DIST]
    return cx.order[k], ok.gather(1, k)


def cosine(cx, s, c, qc, sq):
    X, (mu, sd) = cx.X, cx.ws
    kq = torch.arange(-LQ, 0, device=X.device)
    W = X[s[..., None] + kq, c[:, None, None]]
    sc = sd[s, c[:, None]]
    cos = (W @ qc[..., None])[..., 0] / (LQ * sc * sq[:, None]).clamp_min(1e-12)
    return torch.where((sc > 1e-6) & (sq[:, None] > 1e-6), cos, torch.zeros_like(cos))


def build(cx, t, c):
    X, (mu, sd), D = cx.X, cx.ws, cx.D
    dev, B = X.device, len(t)
    kq = torch.arange(-LQ, 0, device=dev)
    mq, sq = mu[t, c], sd[t, c]
    qc = X[t[:, None] + kq, c[:, None]] - mq[:, None]
    rec = t[:, None] - H - torch.arange(cx.K - H + 1, device=dev)[None]
    ds, dv = distant(cx, t)
    pool = torch.cat([rec, ds], 1)
    pval = torch.cat([rec >= LQ, dv], 1)
    cos = cosine(cx, pool.clamp_min(LQ), c, qc, sq).masked_fill(~pval, -math.inf)
    top, ti = cos.topk(N_SHAPE, 1)
    v_sh = pval.gather(1, ti)
    s_sh = torch.where(v_sh, pool.gather(1, ti), (t - H)[:, None])
    top = torch.where(v_sh, top, torch.zeros_like(top))
    ns = len(cx.ages)
    s_se = t[:, None] - cx.ages[None]
    cos_se = cosine(cx, s_se, c, qc, sq)
    s = torch.cat([s_se, s_se, s_sh], 1)
    p = torch.cat([cx.pers[None].expand(B, -1), cx.pers[None].expand(B, -1), torch.zeros_like(s_sh)], 1)
    val = torch.zeros_like(s, dtype=torch.bool)
    val[:, :ns] = True
    valid = torch.cat([torch.ones(B, 2 * ns, dtype=torch.bool, device=dev), v_sh], 1)
    cs = torch.cat([cos_se, cos_se, top], 1)
    d = (t[:, None] - s).float()
    dd, dw = d % D, d % (7 * D)
    sqf = sq.clamp_min(SD_FLOOR)[:, None]
    gap = torch.log1p((mu[s, c[:, None]] - mq[:, None]).abs() / sqf).clamp_max(10)
    ratio = torch.log(sd[s, c[:, None]].clamp_min(SD_FLOOR) / sqf).clamp(-10, 10)
    z = torch.stack([(p == 7 * D).float(), (p == D).float(), val.float(), cs, torch.log1p(d / D),
                     torch.minimum(dd, D - dd) / (D / 2), torch.minimum(dw, 7 * D - dw) / (3.5 * D), gap, ratio], -1)
    z = torch.cat([z, cx.sdesc[c][:, None].expand(-1, s.shape[1], -1)], -1)
    vs = v_sh.float()
    n = vs.sum(1).clamp_min(1)
    m1 = (top * vs).sum(1) / n
    sdv = (((top - m1[:, None]) * vs).square().sum(1) / n).sqrt()
    u = torch.cat([torch.stack([m1, sdv, top.max(1).values], -1), cx.sdesc[c]], -1)
    return dict(s=s, p=p, val=val, z=z, u=u, valid=valid)


def readout(cx, t, c, s, p, val):
    X, (mu, sd) = cx.X, cx.ws
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
    med = [torch.quantile(R[:, b:b + m], 0.5, dim=1) for b in (0, J) for m in UR_M]
    sh = [R[:, 2 * 2 * J:2 * 2 * J + k].mean(1) for k in UR_K]
    return torch.stack(med, 1), torch.stack(sh, 1)


@torch.no_grad()
def untrained_eps(cx, origins):
    """eps_c: per-channel MSE of the best untrained rule (per-channel grid) on the given val origins."""
    X = cx.X; C, dev = X.shape[1], X.device
    a11 = torch.zeros(len(UR_M) * 2, C, dtype=torch.float64, device=dev)
    a22 = torch.zeros(len(UR_K), C, dtype=torch.float64, device=dev)
    a12 = torch.zeros(len(UR_M) * 2, len(UR_K), C, dtype=torch.float64, device=dev)
    kh = torch.arange(H, device=dev)
    for t0 in origins.tolist():
        for i in range(0, C, CH):
            c = torch.arange(i, min(C, i + CH), device=dev); t = torch.full_like(c, t0)
            sp = build(cx, t, c)
            R = readout(cx, t, c, sp['s'], sp['p'], sp['val'])
            y = X[t[:, None] + kh, c[:, None]]
            med, sh = untrained_parts(R)
            e1, e2 = (med - y[:, None]).double(), (sh - y[:, None]).double()
            a11[:, c] += e1.square().sum(-1).T
            a22[:, c] += e2.square().sum(-1).T
            a12[..., c] += torch.einsum('bmh,bkh->mkb', e1, e2)
    lam = torch.tensor(UR_LAM, dtype=torch.float64, device=dev)[:, None, None, None]
    sse = lam ** 2 * a11[None, :, None] + 2 * lam * (1 - lam) * a12[None] + (1 - lam) ** 2 * a22[None, None]
    return (sse.reshape(-1, C).min(0).values / (len(origins) * H)).float().clamp_min(1e-6)


def train_joint(ctxs, eps, seed, log=print):
    """One scorer for all datasets. ctxs, eps: dicts by dataset name."""
    names = list(ctxs); dev = ctxs[names[0]].X.device
    g = torch.Generator().manual_seed(seed)
    Q, sp = {}, {}
    for d in names:
        cx = ctxs[d]; C = cx.X.shape[1]
        t_all = torch.randint(cx.K + LQ, cx.n_tr - H + 1, (N_TRAIN_Q,), generator=g).to(dev)
        c_all = torch.randint(0, C, (N_TRAIN_Q,), generator=g).to(dev)
        with torch.no_grad():
            parts = [build(cx, t_all[i:i + CH], c_all[i:i + CH]) for i in range(0, N_TRAIN_Q, CH)]
        sp[d] = {k: torch.cat([x[k] for x in parts]) for k in parts[0]}; Q[d] = (t_all, c_all)
        log(f'[fc] {d}: {N_TRAIN_Q} training queries, D {cx.D}, K {cx.K}')
    torch.manual_seed(seed)
    model = RKR().to(dev)
    zv = torch.cat([sp[d]['z'][sp[d]['valid']] for d in names]); uv = torch.cat([sp[d]['u'] for d in names])
    model.zm.copy_(zv.mean(0)); model.zs.copy_(zv.std(0).clamp_min(1e-3))
    model.um.copy_(uv.mean(0)); model.us.copy_(uv.std(0).clamp_min(1e-3)); del zv, uv
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    per = BATCH // len(names); n_step = math.ceil(N_TRAIN_Q / per)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * n_step)
    kh = torch.arange(H, device=dev); hist = []
    for ep in range(EPOCHS):
        perms = {d: torch.randperm(N_TRAIN_Q, generator=g).to(dev) for d in names}; tot = 0.0
        for i in range(n_step):
            loss = 0.0
            for d in names:
                cx = ctxs[d]; b = perms[d][i * per:(i + 1) * per]
                if len(b) == 0:
                    continue
                t, c = Q[d][0][b], Q[d][1][b]; s = sp[d]
                R = readout(cx, t, c, s['s'][b], s['p'][b], s['val'][b])
                r = (model(s['z'][b], s['u'][b], s['valid'][b])[..., None] * R).sum(1)
                y = cx.X[t[:, None] + kh, c[:, None]]
                loss = loss + ((r - y).square().mean(1) / eps[d][c]).clamp_max(C_MAX).mean() / len(names)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA).add_(pm, alpha=1 - EMA)
            tot += float(loss)
        hist.append(tot / n_step); log(f'[fc] epoch {ep} loss {hist[-1]:.4f}')
    ema.eval()
    return ema, hist


@torch.no_grad()
def predict(model, cx, t, c, ch=CH):
    """FreshCast retrieval prediction r [B, H] for queries (t, c)."""
    out = []
    for i in range(0, len(t), ch):
        tt, cc = t[i:i + ch], c[i:i + ch]
        sp = build(cx, tt, cc)
        R = readout(cx, tt, cc, sp['s'], sp['p'], sp['val'])
        out.append((model(sp['z'], sp['u'], sp['valid'])[..., None] * R).sum(1))
    return torch.cat(out)
