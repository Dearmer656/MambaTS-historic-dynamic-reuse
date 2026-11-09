"""PAT-315 PaTH retrieval: learned retrieval of observed continuations, transported along a daily Householder path.

Frozen iTransformer (L = ITR_L, H = ITR_H), Traffic, standardized data.  Per channel and origin t (hourly):
  mu_t, sd_t      mean / std of y[t-P:t], P = 168 (sd floored)
  key             x_j = (y[j-R:j] - mu_j) / sd_j, R = 96;  q_t = Wq x_t, k_j = Wk x_j (dk = 32)
  value           v_j = (y[j:j+H] - mu_j) / sd_j        (observed continuation at its own level / scale)
  memory          origins j with j + H <= t (matured) and t - j < 12 weeks
  weights         w_tj = softmax_j(tau q_t . k_j / sqrt(dk) + b[(t - j) mod 168])   (b: learned clock bias)
  path            v^_t = sum_j w_tj H_D ... H_(d(j)+1) v_j;  d(j) = calendar day of j, D = last complete day (ends <= t-1);
                  H_d = I - beta_d u_d u_d^T on the H-dim lead space, from day d's error profile g_d = the 24 lead-0
                  backbone errors of day d / sd, their mean and std:  u_d = normalize(Wu g_d), beta_d = 2 sigmoid(wb |g_d| + bb),
                  bb = -4 (starts near identity).  Memories of one day share their path: v^ = sum_d P_(t,d) A_d,
                  A_d = sum_(j in day d) w_tj v_j, evaluated by Horner over days.
  prediction      y^_t[h] = f_t[h] + a_h (mu_t - f_t[h]) + c_h sd_t v^_t[h]          (a_h, c_h learned per lead)
Arms: B0 = B-like (same hour-of-week memories only, plain normalized dot, beta 8, no path; only a, c trained);
      A1 = learned retrieval, H_d = I;  A2 = A1 + path;  A2s = A2 with the transition ORDER shuffled: every memory day d
      keeps its causal set {H_(d+1) .. H_D} but applies it in a random order (seeded by t; same in training and evaluation).
Train on train origins (memories inside train), early stop on val_select (second half of val), test once.
Writes json + raw corrections d = y^ - f for every val / test origin (eval_sr --ext format, key = arm name).
--causal_test: change y at times >= t and check that the prediction at origin t is unchanged.
Usage: ITR_L=720 ITR_H=192 python -m patchmem.eval_pathret --seed 3047 --arm A2
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
from patchmem.eval_err_value import H, L
from patchmem.eval_obs_icl import rolling

P_STAT, R, M_WEEKS, DK, DAY = 168, 96, 12, 32, 24
SD_MIN = 0.05


class PathRet(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.Wq = nn.Linear(R, DK, bias=False)
        self.Wk = nn.Linear(R, DK, bias=False)
        nn.init.orthogonal_(self.Wq.weight)
        self.Wk.weight.data.copy_(self.Wq.weight.data)
        self.log_tau = nn.Parameter(torch.tensor(math.log(8.0)))
        self.clock = nn.Parameter(torch.zeros(P_STAT))
        self.a = nn.Parameter(torch.full((H,), 0.2))
        self.c = nn.Parameter(torch.full((H,), 0.2))
        self.Wu = nn.Linear(DAY + 2, H)
        self.wb = nn.Linear(DAY + 2, 1)
        nn.init.constant_(self.wb.bias, -4.0)

    def logits(self, xq, xm, lag):
        """xq (C R), xm (C M R), lag (M) -> (C M)."""
        if self.arm == 'B0':
            lg = 8.0 * (xm @ xq[..., None])[..., 0] / R
            return lg.masked_fill(~(lag % P_STAT == 0)[None], -float('inf'))
        q, k = self.Wq(xq), self.Wk(xm)
        return self.log_tau.exp() * (k @ q[..., None])[..., 0] / (math.sqrt(DK) * R ** 0.5) + self.clock[lag % P_STAT][None]

    def reflect(self, S, g):
        u = F.normalize(self.Wu(g), dim=-1)
        beta = 2 * torch.sigmoid(self.wb(g.abs()))
        return S - beta * (S * u).sum(-1, keepdim=True) * u


def derived(Xd, F0, T0, n, C, hour0):
    """Everything computed from the observations: rolling stats, normalized lead-0 errors, day features."""
    mu, sd = rolling(Xd, P_STAT)
    sd = sd.clamp_min(SD_MIN)
    shock = torch.zeros(len(Xd), C, device=Xd.device)
    shock[T0:T0 + n] = (Xd[T0:T0 + n] - F0) / sd[T0:T0 + n]                      # error of the lead-0 forecast of time i
    n_day = (len(Xd) + hour0) // DAY + 1
    G = torch.zeros(n_day, C, DAY + 2, device=Xd.device)
    for d in range(n_day):
        i0 = d * DAY - hour0
        if i0 < T0 or i0 + DAY > T0 + n:
            continue
        s = shock[i0:i0 + DAY].T
        G[d] = torch.cat([s, s.mean(1, keepdim=True), s.std(1, keepdim=True)], 1)
    return mu, sd, G


def run(seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base('T', seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    X, Mk, n_tr, n_va = ev.load_series(exp, 'T')
    C = X.shape[1] if args.max_c is None else args.max_c
    T0 = max(L, P_STAT, R) + 1
    E = ev.stream_errors(exp, X, Mk, T0, args.bs)[..., :C]                     # n H C, origins T0 ..
    n = E.shape[0]
    kh = torch.arange(H)
    Fd = X[(torch.arange(n) + T0)[:, None] + kh][..., :C] - E                  # backbone forecasts f_o (n H C), fixed, cpu
    del E
    Xd = X[:, :C].to(dev)
    times = torch.arange(T0, T0 + n)
    split = torch.full((n,), -1)
    split[times <= n_tr - H] = 0
    split[(times >= n_tr) & (times <= n_tr + n_va - H)] = 1
    split[times >= n_tr + n_va] = 2
    vk = torch.nonzero(split == 1).flatten()
    split[vk[len(vk) // 2:]] = 3                                                 # val_fit 1 | val_select 3
    import pandas as pd
    hour0 = pd.Timestamp(pd.read_csv(Path(exp.args.root_path) / exp.args.data_path, usecols=[0], nrows=1).iloc[0, 0]).hour
    day_of = lambda i: (i + hour0) // DAY
    first_hour = lambda d: d * DAY - hour0
    F0 = Fd[:, 0].to(dev)
    st = dict(zip(('mu', 'sd', 'G'), derived(Xd, F0, T0, n, C, hour0)))
    kr, khd = torch.arange(-R, 0, device=dev), torch.arange(H, device=dev)

    def predict(net, t, X_=None, st_=None, gen=None):
        X_, st_ = (Xd, st) if X_ is None else (X_, st_)
        mu, sd, G = st_['mu'], st_['sd'], st_['G']
        j = torch.arange(max(t - M_WEEKS * P_STAT + 1, T0), t - H + 1, device=dev)
        xq = ((X_[t + kr] - mu[t]) / sd[t]).T                                       # C R
        xm = ((X_[j[:, None] + kr] - mu[j][:, None]) / sd[j][:, None]).permute(2, 0, 1)   # C M R
        vm = ((X_[j[:, None] + khd] - mu[j][:, None]) / sd[j][:, None]).permute(2, 0, 1)  # C M H
        w = torch.softmax(net.logits(xq, xm, t - j), -1)                            # C M
        dj = day_of(j)
        d_lo = int(dj[0])
        D = day_of(t) - 1
        while first_hour(D) + DAY > t:                                              # last day fully observed by t - 1
            D -= 1
        A = torch.zeros(C, D - d_lo + 1, H, device=dev).index_add_(1, dj - d_lo, w[..., None] * vm)
        if net.arm in ('B0', 'A1'):
            v = A.sum(1)
        elif net.arm == 'A2':
            v = torch.zeros_like(A[:, 0])
            for i in range(A.shape[1]):                                             # Horner: add memory day d_lo + i, then the
                v = v + A[:, i]                                                     # transition of day d_lo + i + 1 (<= D)
                if d_lo + i + 1 <= D:
                    v = net.reflect(v, G[d_lo + i + 1])
        else:                                                                       # A2s: each memory day keeps its transition set
            v = torch.zeros_like(A[:, 0])                                           # {d+1 .. D}, applied in a random order
            for i in range(A.shape[1]):
                S = A[:, i]
                days = list(range(d_lo + i + 1, D + 1))
                for k in torch.randperm(len(days), generator=gen).tolist():
                    S = net.reflect(S, G[days[k]])
                v = v + S
        f = Fd[t - T0].to(dev).T                                                    # C H
        yhat = f + net.a[None] * (mu[t][:, None] - f) + net.c[None] * sd[t][:, None] * v
        return yhat, X_[t + khd].T, f

    def gen_for(t):
        return torch.Generator().manual_seed(int(t))

    net = PathRet(args.arm).to(dev)
    if args.causal_test:
        t = int(torch.nonzero(split == 2).flatten()[100]) + T0
        with torch.no_grad():
            y0 = predict(net, t, gen=gen_for(t))[0]
            noise = float((predict(net, t, gen=gen_for(t))[0] - y0).abs().max())          # GPU atomics (index_add)
            X2 = Xd.clone()
            X2[t:] += torch.randn_like(X2[t:])
            st2 = dict(zip(('mu', 'sd', 'G'), derived(X2, F0, T0, n, C, hour0)))
            y1 = predict(net, t, X2, st2, gen=gen_for(t))[0]
            X3 = Xd.clone()
            X3[t - 1] += 1.0                                                        # sanity: the last observed hour does matter
            st3 = dict(zip(('mu', 'sd', 'G'), derived(X3, F0, T0, n, C, hour0)))
            y2 = predict(net, t, X3, st3, gen=gen_for(t))[0]
        ch = float((y1 - y0).abs().max())
        print(f'[causal] arm {args.arm} t {t}: max |change| with y[>=t] perturbed {ch:.3e} (rerun noise {noise:.3e}, must be '
              f'at float noise); with y[t-1] perturbed {float((y2 - y0).abs().max()):.3e} (> 0)', flush=True)
        assert ch <= max(10 * noise, 1e-5), ch
        return

    rte = torch.nonzero(split == 2).flatten()
    base = float((X[(rte + T0)[:, None] + kh][..., :C] - Fd[rte]).double().square().mean())
    assert args.max_c is not None or abs(base - official) < 1e-4, (base, official)
    print(f'[base] {base:.5f} n {n} C {C} hour0 {hour0}', flush=True)
    ktr = torch.nonzero(split == {'train': 0, 'val_fit': 1}[args.fit]).flatten()
    ktr = ktr[ktr >= M_WEEKS * P_STAT]                                             # full memory window
    print(f'[fit] {args.fit}: {len(ktr)} origins, {min(args.steps, len(ktr))} steps/epoch', flush=True)
    g = torch.Generator().manual_seed(seed)

    def evaluate(rows, store=None):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for k in rows.tolist():
                yh, y, f = predict(net, T0 + k, gen=gen_for(T0 + k))
                tot += float((yh - y).double().square().sum())
                cnt += y.numel()
                if store is not None:
                    store[0][store[1][k]] = (yh - f).half().cpu().numpy()         # C H
        return tot / max(cnt, 1)

    sel_rows = torch.nonzero(split == 3).flatten()
    params = [net.a, net.c] if args.arm == 'B0' else list(net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    log, vlog, best = [], [], [float('inf'), None, -1]
    for ep in range(args.epochs if args.arm != 'B0' else 2):
        t_ep = time.time()
        order = ktr[torch.randperm(len(ktr), generator=g)][:args.steps]
        tot, win = 0.0, 0.0
        for i, k in enumerate(order.tolist()):
            yh, y, _ = predict(net, T0 + k, gen=gen_for(T0 + k))
            loss = (yh - y).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            win += float(loss)
            if (i + 1) % args.log_every == 0:
                print(f'[{args.arm}] ep {ep} step {i + 1} train(last {args.log_every}) {win / args.log_every:.5f}', flush=True)
                win = 0.0
        v_ = evaluate(sel_rows[::args.val_stride])
        log.append(tot / len(order))
        vlog.append(v_)
        if v_ < best[0]:
            best[:] = [v_, copy.deepcopy(net.state_dict()), ep]
        print(f'[{args.arm}] epoch {ep} train {log[-1]:.5f} val_select(sub) {v_:.5f} {time.time() - t_ep:.0f}s', flush=True)
    net.load_state_dict(best[1])
    ext_rows = torch.nonzero(split >= 1).flatten()
    D_ = np.zeros((len(ext_rows), C, H), dtype=np.float16)
    idx = {int(k): i for i, k in enumerate(ext_rows.tolist())}
    res = {name: evaluate(torch.nonzero(split == z).flatten(), (D_, idx)) for z, name in ((1, 'val_fit'), (3, 'val_select'), (2, 'test'))}
    gain = 100 * (base - res['test']) / base
    print(f'[{args.arm}] test {res["test"]:.5f} gain {gain:+.3f}% val_select {res["val_select"]:.5f} best_ep {best[2]}', flush=True)
    od = Path(args.out_dir) / args.arm
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f'T_H{H}_L{L}_s{seed}_ext.npz', origins=(ext_rows + T0).numpy(), **{args.arm: D_})
    (od / f'T_H{H}_L{L}_s{seed}.json').write_text(json.dumps(dict(
        seed=seed, arm=args.arm, base=base, test_mse=res['test'], gain=gain, val_fit=res['val_fit'], val_select=res['val_select'],
        train_loss=log, val_log=vlog, best_epoch=best[2], a=net.a.tolist(), c=net.c.tolist(), clock=net.clock.tolist(),
        tau=float(net.log_tau.exp()), settings=dict(P=P_STAT, R=R, M_weeks=M_WEEKS, dk=DK, epochs=args.epochs,
                                                    steps=args.steps, lr=args.lr, hour0=hour0, fit=args.fit)), indent=1) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--arm', default='A2', choices=('B0', 'A1', 'A2', 'A2s'))
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--steps', type=int, default=1500, help='training origins per epoch (each with all channels)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--fit', default='train', choices=('train', 'val_fit'), help='origins the retrieval / coefficients are fit on')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--max_c', type=int, default=None)
    p.add_argument('--val_stride', type=int, default=4, help='val_select subsample for early stopping')
    p.add_argument('--log_every', type=int, default=200)
    p.add_argument('--causal_test', action='store_true')
    p.add_argument('--out_dir', default='patchmem/logs/server/err_pathret')
    a = p.parse_args()
    run(a.seed, a)


if __name__ == '__main__':
    main()
