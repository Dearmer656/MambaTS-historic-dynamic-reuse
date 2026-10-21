"""PAT-307 model-agnostic error-memory plug-ins on a FROZEN fixed-input/fixed-output forecaster f (here the official MambaTS, A0).
The memory is in-context: the forecasts f issued at the series' own past origins s_m = t - m*P (m = 1..M, P = patch_len) and their
residuals over the part already observed at t. Nothing about f's internals is needed except an optional key (the head input).

Cache (f frozen, eval mode, fixed ATSP order => deterministic): for every absolute origin s of a split, yhat(s) = f(x[s-L:s]) (H, C) and
z(s) = the head input per variable (C, l*d). Window i of the extended dataset (lookback Lext = L + M*P) has its current origin at
t = i + Lext; its past origin m is s_m = t - m*P, i.e. a lookup. The current forecast yhat(t) equals the official A0 test forecast.
Residual of origin m, lead k (0-based), normalised with the CURRENT window's RevIN std (last L steps):
  R[m, k] = (x[s_m + k] - yhat(s_m)[k]) / sd,   observed (usable at t) iff s_m + k < t  <=>  k < m*P.
  Full residual (TEFL: "the forecast issued one horizon ago"): m* = H / P, all H leads observed.
Arms (only the plug-in is trained; y' = yhat(t) + sd * delta):
  base    delta = 0                                              (reproduces A0)
  tefl    delta = ReLU(eps W1) W2, eps = R[m*] (H), per variable, shared; rank 64, W2 = 0   (TEFL's error module on frozen f)
  pos     delta[k] = sum_m beta[k, m] R[m, k] over observed (m, k), beta = 0      (position-addressed multi-origin, TEFL Eq.(1)-like)
  posfull pos restricted to full residuals (m >= m*)                                    (freshness ablation of pos)
  ret     delta[k] = sum_h W[h, k] sum_m softmax_m(q_h(z_t) . k_h(z_m) | m observed at lead k) R[m, k]     (content-addressed, lead-aligned)
  retfull ret restricted to full residuals (m >= m*)                                                     (freshness ablation)
usage: python patchmem/plug_a0.py --setting <A0 ckpt dir> --arm ret --seed 3047 --M 15 -- <run.py args of the A0 run>"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_dict
from utils.tools import set_seed


def load_a0(args, setting, seed):
    exp = Exp_Long_Term_Forecast(args)
    m = exp.model
    m.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'checkpoint.pth'), map_location=exp.device))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    if args.VPT_mode in [1]:
        m.reset_ids_shuffle()
    set_seed(seed)
    order = None
    with torch.no_grad():                                                                # one ATSP solve right after set_seed (as a0_trunc_eval)
        m(torch.zeros(1, args.seq_len, args.enc_in, device=exp.device), None, None, None)
        if m.encoder.ids_shuffle is not None:
            order = m.encoder.ids_shuffle[:1].clone()
    return m, exp.device, order


@torch.no_grad()
def cache_split(m, dev, data_x, L, H, order, bs=256):
    """yhat (N, H, C) and z (N, C, l*d) for every origin s in [L, len(data_x)] (index s - L); order: the fixed ATSP scan order."""
    zbuf = {}
    hk = m.head.register_forward_hook(lambda mod, inp, out: zbuf.__setitem__('z', inp[0]))
    X = torch.as_tensor(data_x, dtype=torch.float32)
    origins = list(range(L, len(data_x) + 1))
    Y, Z = [], []
    for i0 in range(0, len(origins), bs):
        ss = origins[i0:i0 + bs]
        bx = torch.stack([X[s - L:s] for s in ss]).to(dev)
        if order is not None:
            m.encoder.ids_shuffle = order.repeat(bx.shape[0], 1)
        y = m(bx, None, None, None)
        c = y.shape[-1]
        Y.append(y.cpu()); Z.append(zbuf['z'].reshape(bx.shape[0], c, -1).cpu())
    hk.remove()
    return torch.cat(Y), torch.cat(Z), order


class SplitData:
    """Windows i of the extended lookback Lext; current origin t = i + Lext (absolute index in data_x)."""
    def __init__(self, data_x, Yc, Zc, L, H, M, P):
        self.x = data_x if torch.is_tensor(data_x) else torch.as_tensor(data_x, dtype=torch.float32)
        self.Y, self.Z, self.L, self.H, self.M, self.P = Yc, Zc, L, H, M, P
        self.Lext = L + M * P
        self.n = len(data_x) - self.Lext - H + 1

    def batch(self, idx, dev):
        L, H, M, P = self.L, self.H, self.M, self.P
        t = torch.as_tensor(idx) + self.Lext                                            # (b,)
        ms = torch.arange(M + 1)
        s = t[:, None] - ms[None, :] * P                                                 # (b, M+1) origins, m = 0 is the current one
        yhat = self.Y[s - L]                                                             # (b, M+1, H, C)
        z = self.Z[s - L]                                                                # (b, M+1, C, D)
        k = torch.arange(H)
        tidx = s[:, :, None] + k[None, None, :]                                          # absolute index of lead k of origin m
        obs = tidx < t[:, None, None]                                                    # (b, M+1, H) observed at t
        xt = self.x[tidx.clamp(max=len(self.x) - 1)]                                     # (b, M+1, H, C)
        look = self.x[(t[:, None] - L + torch.arange(L)[None, :])]                       # (b, L, C) current lookback (A0's RevIN window)
        mu = look.mean(1, keepdim=True)
        sd = torch.sqrt(look.var(1, keepdim=True, unbiased=False) + 1e-5)                # (b, 1, C)
        R = (xt - yhat) / sd[:, None]
        R = R * obs[..., None]
        y = self.x[t[:, None] + k[None, :]]                                              # (b, H, C) target
        return dict(yhat0=yhat[:, 0].to(dev), R=R[:, 1:].to(dev), obs=obs[:, 1:].to(dev), z0=z[:, 0].to(dev), zm=z[:, 1:].to(dev),
                    sd=sd.to(dev), y=y.to(dev))


class Plugin(nn.Module):
    def __init__(self, arm, H, M, P, dz, rank=64, nh=4, dk=8):
        super().__init__()
        self.arm, self.H, self.M, self.P = arm, H, M, P
        self.mstar = H // P
        assert H % P == 0 and self.mstar <= M, (H, P, M)
        if arm == 'tefl':
            self.W1 = nn.Linear(H, rank, bias=False)
            self.W2 = nn.Linear(rank, H, bias=False); nn.init.zeros_(self.W2.weight)
        elif arm in ('pos', 'posfull'):
            self.beta = nn.Parameter(torch.zeros(H, M))
        elif arm in ('ret', 'retfull'):
            self.nh, self.dk = nh, dk
            self.Wq = nn.Linear(dz, nh * dk, bias=False)
            self.Wk = nn.Linear(dz, nh * dk, bias=False)
            self.log_tau = nn.Parameter(torch.zeros(nh))
            self.W = nn.Parameter(torch.zeros(nh, H))                                    # per head, per lead; zero: base at init
        self.audit = None

    def forward(self, d):
        R, obs = d['R'], d['obs']                                                        # (b, M, H, C), (b, M, H)
        b, M, H, C = R.shape
        if self.arm == 'base':
            delta = torch.zeros_like(d['yhat0'])
        elif self.arm == 'tefl':
            eps = R[:, self.mstar - 1].permute(0, 2, 1)                                  # (b, C, H): origin m* (index m*-1), all observed
            delta = self.W2(torch.relu(self.W1(eps))).permute(0, 2, 1)
        elif self.arm in ('pos', 'posfull'):
            use = obs if self.arm == 'pos' else obs & (torch.arange(1, M + 1, device=R.device) >= self.mstar)[None, :, None]
            delta = torch.einsum('km,bmkc->bkc', self.beta, R * use[..., None])
        else:
            avail = obs if self.arm == 'ret' else obs & (torch.arange(1, M + 1, device=R.device) >= self.mstar)[None, :, None]
            q = self.Wq(d['z0']).view(b, C, self.nh, self.dk)                            # (b, C, nh, dk)
            kk = self.Wk(d['zm']).view(b, M, C, self.nh, self.dk)                        # (b, M, C, nh, dk)
            logit = torch.einsum('bchd,bmchd->bchm', q, kk) / (self.dk ** 0.5 * self.log_tau.exp())[None, None, :, None]
            if self.audit == 'uniform':
                logit = torch.zeros_like(logit)
            av = avail.permute(0, 2, 1)[:, None, None]                                   # (b, 1, 1, H, M)
            lg = logit[:, :, :, None, :].expand(b, C, self.nh, H, M).masked_fill(~av, float('-inf'))
            anyv = av.any(-1, keepdim=True)
            a = torch.softmax(lg.masked_fill(~anyv, 0.0), -1) * anyv                     # (b, C, nh, H, M); no observed origin -> 0
            mh = torch.einsum('bchkm,bmkc->bchk', a, R)                                  # (b, C, nh, H)
            delta = torch.einsum('hk,bchk->bkc', self.W, mh)
            self.last_attn = a.detach()
        if self.audit == 'zero':
            delta = torch.zeros_like(delta)
        return d['yhat0'] + delta * d['sd']


def evaluate(pl, sd_, dev, bs=256, rng_=None):
    pl.eval()
    se, ae, n = 0.0, 0.0, 0
    lo, hi = rng_ if rng_ is not None else (0, sd_.n)
    with torch.no_grad():
        for i0 in range(lo, hi, bs):
            d = sd_.batch(list(range(i0, min(i0 + bs, hi))), dev)
            p = pl(d)
            se += (p - d['y']).square().mean((1, 2)).sum().item(); ae += (p - d['y']).abs().mean((1, 2)).sum().item(); n += p.shape[0]
    return se / n, ae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True); ap.add_argument('--arm', default='ret')
    ap.add_argument('--seed', type=int, default=3047); ap.add_argument('--M', type=int, default=15)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default=None); ap.add_argument('--cache', default=None)
    ap.add_argument('--fit', default='train', choices=['train', 'val'],
                    help="train: fit on the train split (f's in-sample residuals), early stop on val; val: fit on the first 70%% of val windows "
                         "(out-of-sample residuals), early stop on the last 30%% (separated by H + Lext windows); test untouched")
    a, rest = ap.parse_known_args()
    args = official_args([r for r in rest if r != '--'])
    L, H, P = args.seq_len, args.pred_len, args.patch_len
    m, dev, order = load_a0(args, a.setting, a.seed)
    Data = data_dict[args.data]
    splits = {}
    cache = a.cache or f'patchmem/logs/plug_cache/{a.setting}_M{a.M}_v2.pt'
    if os.path.exists(cache):
        C = torch.load(cache)
    else:
        C = {}
        for flag in ('train', 'val', 'test'):
            ds = Data(root_path=args.root_path, data_path=args.data_path, flag=flag, size=[L + a.M * P, args.label_len, H],
                      features=args.features, target=args.target, timeenc=1, freq=args.freq)
            t0 = time.time()
            Y, Z, _ = cache_split(m, dev, ds.data_x, L, H, order)
            C[flag] = dict(x=torch.as_tensor(ds.data_x, dtype=torch.float32), Y=Y, Z=Z)
            print(f'[cache] {flag}: {len(Y)} origins in {time.time() - t0:.0f}s | ATSP order {None if order is None else order[0].tolist()}', flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(C, cache)
    for flag in ('train', 'val', 'test'):
        splits[flag] = SplitData(C[flag]['x'], C[flag]['Y'], C[flag]['Z'], L, H, a.M, P)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pl = Plugin(a.arm, H, a.M, P, dz=C['test']['Z'].shape[-1]).to(dev)
    npar = sum(p.numel() for p in pl.parameters())
    print(f'[plug] arm {a.arm} | L {L} H {H} P {P} M {a.M} (Lext {L + a.M * P}) | m* {H // P} | params {npar} | windows train {splits["train"].n} '
          f'val {splits["val"].n} test {splits["test"].n}', flush=True)
    res = dict(arm=a.arm, setting=a.setting, seed=a.seed, M=a.M, params=npar, fit=a.fit)
    if a.fit == 'train':
        fit_sd, fit_rng, es_sd, es_rng = splits['train'], (0, splits['train'].n), splits['val'], (0, splits['val'].n)
    else:
        nv = splits['val'].n; cut = int(0.7 * nv)
        fit_sd, fit_rng, es_sd, es_rng = splits['val'], (0, cut), splits['val'], (min(cut + H + L + a.M * P, nv - 1), nv)
    print(f'[plug] fit {a.fit}: fit windows {fit_rng}, early-stop windows {es_rng}', flush=True)
    if npar:
        opt = torch.optim.Adam(pl.parameters(), lr=a.lr)
        best, bad, state = float('inf'), 0, None
        rng = np.random.default_rng(a.seed)
        for ep in range(a.epochs):
            pl.train(); t0 = time.time(); tl = []
            nfit = fit_rng[1] - fit_rng[0]
            for idx in np.array_split(fit_rng[0] + rng.permutation(nfit), max(nfit // a.bs, 1)):
                d = fit_sd.batch(idx.tolist(), dev)
                loss = (pl(d) - d['y']).square().mean()
                opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
            v = evaluate(pl, es_sd, dev, rng_=es_rng)[0]
            print(f'[plug] epoch {ep + 1} {time.time() - t0:.0f}s | train {np.mean(tl):.5f} | val mse {v:.5f}', flush=True)
            if v < best:
                best, bad, state = v, 0, {k: x.clone() for k, x in pl.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        pl.load_state_dict(state)
        res['best_val'] = best
    for flag in ('val', 'test'):
        mse, mae = evaluate(pl, splits[flag], dev)
        res[flag] = dict(mse=mse, mae=mae)
        audits = ''
        if a.arm != 'base':
            pl.audit = 'zero'; z = evaluate(pl, splits[flag], dev)[0]; pl.audit = None
            res[flag]['zero'] = z; audits = f' | zero audit {z:.4f}'
            if a.arm in ('ret', 'retfull'):
                pl.audit = 'uniform'; u = evaluate(pl, splits[flag], dev)[0]; pl.audit = None
                res[flag]['uniform'] = u; audits += f' | uniform attn {u:.4f}'
        print(f'[plug] {flag} | arm {a.arm} | L {L} H {H} | mse {mse:.4f} mae {mae:.4f}{audits}', flush=True)
    out = a.out or f'patchmem/logs/plug/{args.data}_L{L}_H{H}_{a.arm}_M{a.M}_fit{a.fit}_s{a.seed}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
