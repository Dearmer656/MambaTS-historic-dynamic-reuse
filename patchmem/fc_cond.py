"""PAT-316 follow-up: can a state / relative-position conditioned candidate representation raise FreshCast's rho
(alignment of the correction d = p - f with the backbone residual e = y - f; gain ~ rho^2)?

FreshCast reproduction (patchmem/eval_freshcast.py: same candidates, readouts, 12 relational descriptors, RKR head) on the
same frozen iTransformer checkpoint.  Four scorers, trained identically:
  a  FreshCast RKR (12 descriptors -> linear + MLP score, learned inverse temperature).
  b  a + the backbone forecast f: per candidate the disagreement R_i - f (mean, rms, first-24h mean, / query sd) and per
     query the matured backbone error of the forecast issued at t - H (mean, rms, / query sd).
  c  b + a static relative-position query-key term q_t^T R(t - s) k_s / sqrt(D): q, k linear in the standardized query /
     candidate windows (LQ = 96), R = RoPE rotation by the age t - s, 8 learnable periods (init 12 h .. 1 y).
  d  b + a path-conditioned query-key term q_t^T P(t, s) k_s / sqrt(D), P(t, s) = G_kt ... G_{ks+1}: one transform per
     day-patch k of the channel's own series, G_k = (I - 2 u u^T)(I - 2 w_k w_k^T), u = norm(a), w_k = norm(a + g(patch k))
     (two Householder reflections = a data-dependent rotation, identity when g = 0; g zero-initialised, so d starts as a
     position-free q-k term).  kt = t // 24 - 1, ks = s // 24 - 1 (last complete day before t / s: causal).
     P = Q_kt Q_ks^T with prefix products Q (orthogonal), computed by a doubling scan.
All four: training queries from the first 75% of the validation origins (the backbone's f there is out of sample; the
FreshCast training segment would give in-sample f), N_TRAIN_Q queries, 30 epochs, loss min(||e - w d||^2 / (H eps_c), C_MAX),
w = sigmoid(learned), eps_c = per-channel backbone MSE on the training origins.  Test combination y^ = f + w (p - f),
w = clip(<e, d> / <d, d>, 0, 1) refit on 100 origins drawn from the last 25% of the validation origins.

Criteria (fixed 2026-11-09 before any result; EL H192 L720, seeds 3047 / 2024 / 2025, full test set, test gain over the
backbone):
  GO (path conditioning raises rho):  gain(d) > gain(c) AND gain(d) > gain(a) in 3/3 seeds.
  NO-GO:                               gain(d) <= gain(c) in 3/3 seeds.
  Secondary: gain(b) > gain(a) in 3/3 (seeing f helps); 'beats FreshCast' = max(b, c, d) > the stored FreshCast gain
  (eval_freshcast json, trained on the training segment) in 3/3.  Reported: rho per arm, both test halves, arm a vs stored.

usage: ITR_H=192 ITR_L=720 python -m patchmem.fc_cond --data EL --seed 3047 --out_dir patchmem/logs/laptop/fc_cond [--smoke]
       python -m patchmem.fc_cond --selftest
"""
import argparse
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
from patchmem.eval_error_path import H, L

ARMS = ('a', 'b', 'c', 'd')
D, NP = 16, 24
NZX, NUX = 3, 2
PERIODS0 = (12.0, 24.0, 84.0, 168.0, 336.0, 720.0, 2190.0, 8760.0)
TRAIN_FRAC = 0.75
ZCLIP = 10.0


def prefix_Q(G):
    """G (C K D D) -> Q (C K D D), Q_k = G_k G_{k-1} ... G_0 (Hillis-Steele doubling scan)."""
    Q, step, K = G, 1, G.shape[1]
    while step < K:
        Q = torch.cat([Q[:, :step], Q[:, step:] @ Q[:, :-step]], 1)
        step *= 2
    return Q


def patches(X):
    """X (T C) -> (C K NP) day patches, K = T // NP."""
    K = len(X) // NP
    return X[:K * NP].T.reshape(X.shape[1], K, NP)


class Scorer(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        nz, nu = fc.NZ + (NZX if arm != 'a' else 0), fc.NU + (NUX if arm != 'a' else 0)
        self.lin = nn.Linear(nz, 1)
        self.mlp = nn.Sequential(nn.Linear(nz, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.b0 = nn.Parameter(torch.zeros(()))
        self.bnet = nn.Sequential(nn.Linear(nu, 16), nn.GELU(), nn.Linear(16, 1))
        self.w_logit = nn.Parameter(torch.zeros(()))
        for k, n in (('zm', nz), ('zs', nz), ('um', nu), ('us', nu)):
            self.register_buffer(k, torch.zeros(n) if k.endswith('m') else torch.ones(n))
        if arm in ('c', 'd'):
            self.wq = nn.Linear(fc.LQ, D, bias=False)
            self.wk = nn.Linear(fc.LQ, D, bias=False)
        if arm == 'c':
            self.log_per = nn.Parameter(torch.log(torch.tensor(PERIODS0)))
        if arm == 'd':
            self.a = nn.Parameter(torch.randn(D) / D ** 0.5)
            self.pnet = nn.Sequential(nn.Linear(NP + 2, 32), nn.GELU(), nn.Linear(32, D))
            nn.init.zeros_(self.pnet[2].weight)
            nn.init.zeros_(self.pnet[2].bias)

    def gens(self, Xp):
        """Xp (C K NP) -> G (C K D D) = (I - 2 u u^T)(I - 2 w w^T)."""
        m = Xp.mean(-1, keepdim=True)
        feat = torch.cat([Xp - m, m, Xp.std(-1, keepdim=True)], -1)
        u = Fn.normalize(self.a, dim=0)
        w = Fn.normalize(self.a + self.pnet(feat), dim=-1)
        I = torch.eye(D, device=Xp.device, dtype=Xp.dtype)
        Hu = I - 2 * u[:, None] * u[None]
        Hw = I - 2 * w[..., :, None] * w[..., None, :]
        return Hu @ Hw

    def path_Q(self, Xp, ch=64):
        return torch.cat([prefix_Q(self.gens(Xp[i:i + ch])) for i in range(0, len(Xp), ch)])

    def forward(self, sp, inp, Q=None, cidx=None):
        z, u = sp['z'], sp['u']
        if self.arm != 'a':
            z, u = torch.cat([z, inp['zx']], -1), torch.cat([u, inp['ux']], -1)
        zn = (z - self.zm) / self.zs
        sc = self.lin(zn)[..., 0] + self.mlp(zn)[..., 0]
        if self.arm in ('c', 'd'):
            q, k = self.wq(inp['qw']), self.wk(inp['kw'])                           # B D, B N D
            if self.arm == 'c':
                th = 2 * math.pi / torch.exp(self.log_per)                          # D/2
                ang = inp['age'][..., None] * th                                    # B N D/2
                q2, k2 = q.view(len(q), 1, D // 2, 2), k.view(*k.shape[:2], D // 2, 2)
                qk = ((q2[..., 0] * k2[..., 0] + q2[..., 1] * k2[..., 1]) * torch.cos(ang)
                      + (q2[..., 1] * k2[..., 0] - q2[..., 0] * k2[..., 1]) * torch.sin(ang)).sum(-1)
            else:
                Qt = Q[cidx, inp['kt']]                                             # B D D
                Qs = Q[cidx[:, None], inp['ks']]                                    # B N D D
                qt = torch.einsum('bij,bi->bj', Qt, q)                              # Q_kt^T q
                ks = torch.einsum('bnij,bni->bnj', Qs, k)                           # Q_ks^T k
                qk = (qt[:, None] * ks).sum(-1)
            sc = sc + qk / D ** 0.5
        beta = torch.exp(self.b0 + self.bnet((u - self.um) / self.us)[..., 0].clamp(-2, 2))
        return torch.softmax((beta[:, None] * sc).masked_fill(~sp['valid'], -math.inf), -1)


def inputs(X, ws, t, c, s, R, f, em):
    """Extra scorer inputs; f (B H) backbone forecast, em (B H) matured error of the forecast issued at t - H."""
    mu, sd = ws
    sq = sd[t, c].clamp_min(fc.SD_FLOOR)
    mq = mu[t, c]
    Dif = (R - f[:, None]) / sq[:, None, None]
    zx = torch.stack([Dif.mean(-1), Dif.square().mean(-1).sqrt(), Dif[..., :24].mean(-1)], -1).clamp(-ZCLIP, ZCLIP)
    emn = em / sq[:, None]
    ux = torch.stack([emn.mean(-1), emn.square().mean(-1).sqrt()], -1).clamp(-ZCLIP, ZCLIP)
    kq = torch.arange(-fc.LQ, 0, device=X.device)
    qw = ((X[t[:, None] + kq, c[:, None]] - mq[:, None]) / sq[:, None]).clamp(-ZCLIP, ZCLIP)
    ms, ss = mu[s, c[:, None]], sd[s, c[:, None]].clamp_min(fc.SD_FLOOR)
    kw = ((X[s[..., None] + kq, c[:, None, None]] - ms[..., None]) / ss[..., None]).clamp(-ZCLIP, ZCLIP)
    kt, ks = t // NP - 1, s // NP - 1
    assert bool((NP * (kt + 1) <= t).all()) and bool((NP * (ks + 1) <= s).all())
    return dict(zx=zx, ux=ux, qw=qw, kw=kw, age=(t[:, None] - s).float(), kt=kt, ks=ks)


class Ctx:
    """Series, candidate machinery and the cached backbone forecasts Fc[o - o0] (H C) for o0 <= o <= T - H."""
    def __init__(self, X, ws, sdesc, order, Fc, o0):
        self.X, self.ws, self.sdesc, self.order, self.Fc, self.o0 = X, ws, sdesc, order, Fc, o0
        self.kh = torch.arange(H, device=X.device)

    def query(self, t, c, sp=None):
        X = self.X
        if sp is None:
            sp = fc.build(X, self.ws, self.sdesc, self.order, t, c)
        R = fc.readout(X, self.ws, t, c, sp['s'], sp['p'], sp['val'])
        f = self.Fc[t - self.o0, :, c]                                              # B H
        y = X[t[:, None] + self.kh, c[:, None]]
        em = X[(t - H)[:, None] + self.kh, c[:, None]] - self.Fc[t - H - self.o0, :, c]
        return sp, R, f, y, inputs(X, self.ws, t, c, sp['s'], R, f, em)


def train_arm(arm, ctx, Xp, t_all, c_all, sp_all, eps, seed):
    dev = ctx.X.device
    torch.manual_seed(seed)
    model = Scorer(arm).to(dev)
    with torch.no_grad():                                                           # input normalisation
        nn_ = min(4096, len(t_all))
        zs, us = [], []
        for i in range(0, nn_, fc.CH):
            b = slice(i, min(nn_, i + fc.CH))
            sp = {k: v[b] for k, v in sp_all.items()}
            _, _, _, _, inp = ctx.query(t_all[b], c_all[b], sp)
            z, u = sp['z'], sp['u']
            if arm != 'a':
                z, u = torch.cat([z, inp['zx']], -1), torch.cat([u, inp['ux']], -1)
            zs.append(z[sp['valid']])
            us.append(u)
        zv, uv = torch.cat(zs), torch.cat(us)
        model.zm.copy_(zv.mean(0))
        model.zs.copy_(zv.std(0).clamp_min(1e-3))
        model.um.copy_(uv.mean(0))
        model.us.copy_(uv.std(0).clamp_min(1e-3))
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=fc.LR, weight_decay=fc.WD)
    nq = len(t_all)
    steps = fc.EPOCHS * math.ceil(nq / fc.BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    log = []
    for ep in range(fc.EPOCHS):
        perm = torch.randperm(nq, generator=g).to(dev)
        tot = 0.0
        for i in range(0, nq, fc.BATCH):
            b = perm[i:i + fc.BATCH]
            t, c = t_all[b], c_all[b]
            sp = {k: v[b] for k, v in sp_all.items()}
            _, R, f, y, inp = ctx.query(t, c, sp)
            Q = cidx = None
            if arm == 'd':
                cu, cidx = torch.unique(c, return_inverse=True)
                Q = prefix_Q(model.gens(Xp[cu]))
            pi = model(sp, inp, Q, cidx)
            d = (pi[..., None] * R).sum(1) - f
            w = torch.sigmoid(model.w_logit)
            loss = (((y - f) - w * d).square().mean(1) / eps[c]).clamp_max(fc.C_MAX).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(fc.EMA).add_(pm, alpha=1 - fc.EMA)
            tot += float(loss) * len(b)
        log.append(tot / nq)
        print(f'[{arm}] epoch {ep} loss {log[-1]:.4f} w {float(torch.sigmoid(model.w_logit)):.3f}', flush=True)
    ema.eval()
    return ema, log


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    t_start = time.time()
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1]
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    vo = torch.arange(n_tr, n_tr + n_va - H + 1)
    n_fit = int(len(vo) * TRAIN_FRAC)
    tr_o, wo = vo[:n_fit], vo[n_fit:]
    w_o = wo[torch.randperm(len(wo), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]].sort().values
    test_o = torch.arange(n_tr + n_va, len(X) - H + 1)
    if args.smoke:
        test_o = test_o[:: max(1, len(test_o) // 40)]
        w_o = w_o[:20]

    o0 = n_tr - H                                                                   # backbone forecasts, every needed origin
    o_all = torch.arange(o0, len(X) - H + 1)
    Fc = torch.zeros(len(o_all), H, C, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for i in range(0, len(o_all), args.forecast_bs):
            ob = o_all[i:i + args.forecast_bs]
            Fc[i:i + len(ob)] = fc.forecast(exp, Xc, Mc, ob).float()
    del exp
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f'[F] {tuple(Fc.shape)} forecasts cached ({time.time() - t_start:.0f}s)', flush=True)
    ctx = Ctx(X, ws, sdesc, order, Fc, o0)
    kh = torch.arange(H, device=dev)
    with torch.no_grad():                                                           # eps_c on the training origins
        se = torch.zeros(C, dtype=torch.float64, device=dev)
        for o in tr_o.tolist():
            se += (X[o + kh] - Fc[o - o0]).double().square().sum(0)
        eps = (se / (len(tr_o) * H)).float().clamp_min(1e-6)

    g = torch.Generator().manual_seed(seed)
    t_all = tr_o[torch.randint(0, len(tr_o), (fc.N_TRAIN_Q,), generator=g)].to(dev)
    c_all = torch.randint(0, C, (fc.N_TRAIN_Q,), generator=g).to(dev)
    with torch.no_grad():
        specs = [fc.build(X, ws, sdesc, order, t_all[i:i + fc.CH], c_all[i:i + fc.CH]) for i in range(0, fc.N_TRAIN_Q, fc.CH)]
    sp_all = {k: torch.cat([x[k] for x in specs]) for k in specs[0]}
    del specs
    Xp = patches(X)
    models, logs = {}, {}
    for arm in ARMS:
        t0 = time.time()
        models[arm], logs[arm] = train_arm(arm, ctx, Xp, t_all, c_all, sp_all, eps, seed)
        print(f'[{arm}] trained in {(time.time() - t0) / 60:.1f} min', flush=True)
    del sp_all
    with torch.no_grad():
        Qall = models['d'].path_Q(Xp)

    def sweep(origins, w=None):
        half = len(origins) // 2
        keys = ['ee', 'n'] + [f'{k}_{a}' for a in ARMS for k in ('ed', 'dd', 'sse')]
        S = [{k: torch.zeros((), dtype=torch.float64, device=dev) for k in keys} for _ in range(2)]
        with torch.no_grad():
            for j, t0 in enumerate(origins.tolist()):
                hs = S[int(j >= half)]
                for c in fc.chunks(C, dev):
                    t = torch.full_like(c, t0)
                    sp, R, f, y, inp = ctx.query(t, c)
                    e = (y - f).double()
                    hs['ee'] += e.square().sum()
                    hs['n'] += e.numel()
                    for a in ARMS:
                        pi = models[a](sp, inp, Qall, c)
                        d = ((pi[..., None] * R).sum(1) - f).double()
                        hs[f'ed_{a}'] += (e * d).sum()
                        hs[f'dd_{a}'] += d.square().sum()
                        if w is not None:
                            hs[f'sse_{a}'] += (e - w[a] * d).square().sum()
        return [{k: float(v) for k, v in s.items()} for s in S]

    v = sweep(w_o)
    vt = {k: v[0][k] + v[1][k] for k in v[0]}
    w = {a: min(max(vt[f'ed_{a}'] / max(vt[f'dd_{a}'], 1e-12), 0.0), 1.0) for a in ARMS}
    print(f'[val] w {w}', flush=True)
    S = sweep(test_o, w)
    tot = {k: S[0][k] + S[1][k] for k in S[0]}

    def summ(s):
        ee = s['ee']
        return dict(base_mse=ee / s['n'],
                    gain={a: 100 * (ee - s[f'sse_{a}']) / ee for a in ARMS},
                    rho={a: s[f'ed_{a}'] / max((ee * s[f'dd_{a}']) ** 0.5, 1e-300) for a in ARMS})
    res = dict(all=summ(tot), first_half=summ(S[0]), second_half=summ(S[1]))
    base = tot['ee'] / tot['n']
    if not args.smoke and abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    ref = Path(args.ref_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    ref_gain = json.loads(ref.read_text())['fc']['gain'] if ref.exists() else None
    gn = res['all']['gain']
    crit = dict(d_gt_c=gn['d'] > gn['c'], d_gt_a=gn['d'] > gn['a'], b_gt_a=gn['b'] > gn['a'],
                beats_freshcast=None if ref_gain is None else max(gn['b'], gn['c'], gn['d']) > ref_gain)
    for k, r in res.items():
        print(f'[test {k:11s}] base {r["base_mse"]:.5f} | gain % ' + ' '.join(f'{a} {r["gain"][a]:+.3f}' for a in ARMS)
              + ' | rho ' + ' '.join(f'{a} {r["rho"][a]:.3f}' for a in ARMS), flush=True)
    print(f'[crit] {crit}  stored FreshCast gain {ref_gain}', flush=True)
    peak_gpu = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    learned = dict(c_periods=torch.exp(models['c'].log_per).tolist())
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), official=official, base=base,
               smoke=bool(args.smoke), n_test_origins=len(test_o), n_train_origins=len(tr_o), w_origins=w_o.tolist(), w=w,
               ref_fc_gain=ref_gain, crit=crit, losses=logs, learned=learned, peak_gpu_gb=peak_gpu,
               peak_rss_gb=peak_rss, seconds=time.time() - t_start, **res)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f'{code}_H{H}_L{L}_fccond_s{seed}{"_smoke" if args.smoke else ""}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}  peak GPU {peak_gpu:.1f} GB, RSS {peak_rss:.1f} GB, {out["seconds"] / 60:.1f} min', flush=True)


def selftest():
    torch.manual_seed(0)
    G = torch.linalg.qr(torch.randn(3, 37, D, D))[0]
    Q = prefix_Q(G)
    ref = G[:, 0]
    for k in range(1, 37):
        ref = G[:, k] @ ref
        assert torch.allclose(Q[:, k], ref, atol=1e-4), k
    m = Scorer('d')
    Xp = torch.randn(2, 50, NP)
    Gm = m.gens(Xp)
    assert torch.allclose(Gm, torch.eye(D).expand_as(Gm), atol=1e-5)                 # g = 0 at init -> identity
    with torch.no_grad():
        m.pnet[2].bias.normal_()
    Qm = prefix_Q(m.gens(Xp))
    assert torch.allclose(Qm @ Qm.transpose(-1, -2), torch.eye(D).expand_as(Qm), atol=1e-4)    # orthogonal
    # transport identity: q^T G_kt..G_{ks+1} k == (Q_kt^T q) . (Q_ks^T k)
    q, k = torch.randn(D), torch.randn(D)
    kt, ks = 40, 12
    P = torch.eye(D)
    for j in range(ks + 1, kt + 1):
        P = m.gens(Xp)[0, j] @ P
    lhs = q @ P @ k
    rhs = (Qm[0, kt].T @ q) @ (Qm[0, ks].T @ k)
    assert torch.allclose(lhs, rhs, atol=1e-4), (lhs, rhs)
    # RoPE arm at age 0 equals the plain dot product; forward shapes of c / d
    B, N = 3, 5
    sp = dict(z=torch.randn(B, N, fc.NZ), u=torch.randn(B, fc.NU), valid=torch.ones(B, N, dtype=torch.bool))
    inp = dict(zx=torch.randn(B, N, NZX), ux=torch.randn(B, NUX), qw=torch.randn(B, fc.LQ), kw=torch.randn(B, N, fc.LQ),
               age=torch.zeros(B, N), kt=torch.full((B,), 40), ks=torch.full((B, N), 12))
    mc = Scorer('c')
    qk = (mc.wq(inp['qw'])[:, None] * mc.wk(inp['kw'])).sum(-1)
    mc0 = Scorer('c')
    mc0.load_state_dict(mc.state_dict())
    mc0.wq.weight.data.zero_()
    diff = torch.log(mc(sp, inp)) - torch.log(mc0(sp, inp))
    beta = torch.exp(mc.b0 + mc.bnet((torch.cat([sp['u'], inp['ux']], -1) - mc.um) / mc.us)[..., 0].clamp(-2, 2))
    exp_ = beta[:, None] * qk / D ** 0.5
    assert torch.allclose(diff - diff.mean(-1, keepdim=True), exp_ - exp_.mean(-1, keepdim=True), atol=1e-4)
    pd = m(sp, inp, Qm, torch.tensor([0, 1, 0]))
    assert pd.shape == (B, N) and torch.allclose(pd.sum(-1), torch.ones(B), atol=1e-5)
    # forecast-cache indexing: Fc[t - o0, :, c] -> (B H)
    Fc = torch.randn(10, H, 7)
    t, c = torch.tensor([3, 5]), torch.tensor([1, 6])
    assert torch.equal(Fc[t, :, c], torch.stack([Fc[3, :, 1], Fc[5, :, 6]]))
    print('SELFTEST OK', flush=True)


def main():
    if '--selftest' in sys.argv:
        return selftest()
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='EL')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/laptop/fc_cond')
    p.add_argument('--ref_dir', default='patchmem/logs/server/error_path')
    p.add_argument('--smoke', action='store_true', help='1 epoch on 2000 queries per arm, ~40 test origins, 20 w origins')
    args = p.parse_args()
    if args.smoke:
        fc.N_TRAIN_Q, fc.EPOCHS = 2000, 1
    run(args.data, args.seed, args)


if __name__ == '__main__':
    main()
