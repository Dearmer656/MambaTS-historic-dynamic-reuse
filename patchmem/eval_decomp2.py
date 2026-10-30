"""PAT-315: standalone predictable-error decomposition on the frozen backbone (no FreshCast underneath).

  y^ = f + a_b m_state + b_b m_x          e = y - f (backbone error), a_b, b_b per lead bucket (NB) by val least squares

Term 1 (input-explainable) S: the state is the backbone INPUT x itself.  x~ = the backbone's input window of the channel
(length L), normalised by its mean / std (as the backbone's own instance norm).  Demos: same channel, stream origins
s <= t - H (error vector fully observed), top-k by cos(x~_t, x~_s); linear-attention read-out
  m_state[t] = (1 / k) sum_{s in topk} cos(x~_t, x~_s) e_s          k in KS chosen on val.
Term 2 (history-dependent increment) X, carried state c in R^(NB x 168) indexed by (lead bucket, target hour of week),
c' = A(x_t) c, A = I - beta u u^T with u learned from the LQ window + d mu + d log sd and rolled with t (eval_decomp.Carry);
written slots <- lam c' + (1 - lam) w, read m_x[t,h] = sig_t^gamma c[bucket(h), (t+h) mod 168].  Writes:
  diag  at origin t the newest observed target tau = t-1 has errors of origins tau-l at every lead l (the observation
        diagonal): slot (bucket(l), tau mod 168) <- mean_{l in bucket} rho[tau-l, l] / sig_t^gamma        (fresh)
  full  the newly matured whole vector rho[t-H] (lead k -> slot (bucket(k), (t-H+k) mod 168)) / sig_{t-H}^gamma (v2)
Pd_err: as Pd, but u, beta, lam are computed from the errors themselves (newest observed diagonal D[k] / sig_t per
lead bucket, its change from k - 1) plus d mu, d log sd, instead of the LQ input window (2 NB + 2 inputs).
Arms Pd (diag + Householder), Pd_I (diag, A = I: lam-EMA weekly template), Pf (full + Householder), TEFL (MLP on the 7
most recent diagonals).  Each X is trained twice on the train-tail origins (truncated BPTT): alone (rho = e) and
stacked (rho = e - a~_b m_state, a~_b fitted on the train-tail origins so that no validation label enters training;
k is the val-selected one).  Joint fits use the raw m_state, so the reported a is its coefficient.
Reported (test MSE, gain vs base and vs FreshCast from its json): S; X alone; S (+) X = S with X-alone, (a_b, b_b) jointly
on val; S -> X = S with X-stacked.  Stream: origins T0 = n_tr - M_TR .. end, contiguous; train tail and gaps are demos.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
import patchmem.eval_decomp as dc
from patchmem.eval_error_path import H, L

M_TR = 2016
NB, PW = dc.NB, dc.PW
KS = (16, 64, 256)
EPS = 1e-12
CHR = 32                                                             # channels per retrieval chunk
BQ = 64                                                              # query origins per retrieval batch
K_TEFL = 7
Z_ERR = 5.0                                                          # clip of the standardised Pd_err inputs


# ---------------------------------------------------------------- term 1
def norm_windows(X, t, c, l):
    """x~ (len(c) len(t) l): input windows X[t-l:t, c], minus mean, over sqrt(var + 1e-5) (backbone instance norm)."""
    w = X[:, c][t[:, None] + torch.arange(-l, 0, device=X.device)].permute(2, 0, 1)
    w = w - w.mean(-1, keepdim=True)
    return w / torch.sqrt(w.var(-1, unbiased=False, keepdim=True) + 1e-5)


def topk_read(Kn, E, kq, h, ks):
    """Kn (c n l) unit-norm keys of all stream indices (any float dtype), E (n c h) errors, kq (b) query indices.
    Returns {k: m (b c h)}: (1/k) sum over the top-k cos among indices <= kq - h of cos * E."""
    q = Kn[:, kq]                                                                # c b l
    cos = (q @ Kn.transpose(1, 2)).float()                                       # c b n
    n = Kn.shape[1]
    ok = torch.arange(n, device=Kn.device)[None] <= (kq - h)[:, None]           # b n
    cos = cos.masked_fill(~ok[None], -float('inf'))
    km = min(max(ks), n)
    val, idx = cos.topk(km, -1)                                                  # c b km
    w = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    Ec = E.permute(1, 0, 2)                                                      # c n h
    g = torch.gather(Ec[:, None].expand(-1, len(kq), -1, -1), 2,
                     idx[..., None].expand(-1, -1, -1, Ec.shape[-1]))            # c b km h
    cw = (w[..., None] * g).cumsum(2)
    return {k: (cw[:, :, min(k, km) - 1] / k).permute(1, 0, 2) for k in ks}


# ---------------------------------------------------------------- term 2
def diagonal(R):
    """D (n c h): D[k, :, l] = R[k-1-l, :, l] (0 where k-1-l < 0) -- errors of the newest observed target at time k."""
    D = torch.zeros_like(R)
    for l in range(R.shape[-1]):
        D[l + 1:, :, l] = R[:R.shape[0] - l - 1, :, l]
    return D


def write_diag(d, k, t, h, pw, nb):
    """d (B h) diagonal at stream index k (time t): lead l -> slot bucket(l) * pw + (t-1) % pw, valid if k-1-l >= 0."""
    dev = d.device
    l = torch.arange(h, device=dev)
    valid = (k - 1 - l >= 0).float()
    sl = dc.lead_bucket(h, nb, dev) * pw + (t - 1) % pw
    cnt = torch.zeros(nb * pw, device=dev).index_add_(0, sl, valid)
    w = torch.zeros(d.shape[0], nb * pw, device=dev, dtype=d.dtype).index_add_(1, sl, d * valid)
    return w / cnt.clamp_min(1), cnt > 0


def carry_step2(model, c, k, T0, mode, src_at, x_at, sig_at, h):
    """One stream index k.  mode 'diag': src_at(k) is the diagonal D[k]; 'full': src_at(k - h) is rho[k - h]."""
    pw, nb = model.pw, model.nb
    t = T0 + k
    x = x_at(k)
    B = x.shape[0]
    if c is None:
        c = torch.zeros(B, nb * pw, device=x.device)
    g = model.g()
    if mode == 'diag':
        w, wm = write_diag(src_at(k) / sig_at(k)[:, None] ** g, k, t, h, pw, nb)
    elif k - h >= 0:
        w, wm = dc.write_vals(src_at(k - h) / sig_at(k - h)[:, None] ** g, t - h, h, pw, nb)
    else:
        w, wm = torch.zeros(B, nb * pw, device=x.device), torch.zeros(nb * pw, dtype=torch.bool, device=x.device)
    c, _, _ = model.step(c, x, t, w, wm)
    return c, sig_at(k)[:, None] ** g * c[:, dc.read_slots(t, h, pw, nb, x.device)]


# ---------------------------------------------------------------- run
def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    done = Path(args.out_dir) / f'{code}_H{H}_L{L}_{args.tag}_s{seed}.json'
    if args.max_c is None and done.exists():                                     # finished elsewhere (laptop)
        print(f'[skip] {done} exists', flush=True)
        return
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    fcj = Path(args.out_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    fc_mse = json.loads(fcj.read_text())['fc']['test_mse'] if fcj.exists() else None
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1] if args.max_c is None else args.max_c
    T0 = n_tr - M_TR
    n = len(X) - H + 1 - T0
    times = torch.arange(T0, T0 + n)
    split = torch.full((n,), -1)
    split[times <= n_tr - H] = 0
    split[(times >= n_tr) & (times <= n_tr + n_va - H)] = 1
    split[times >= n_tr + n_va] = 2
    nel = {s: int((split == s).sum()) * C * H for s in (0, 1, 2)}
    spl = split.to(dev)
    kh = torch.arange(H, device=dev)
    bk = dc.lead_bucket(H, NB, dev)

    # ---- stream: backbone errors (n C H) on the CPU
    E = torch.zeros(n, C, H)
    with torch.no_grad():
        for i in range(0, n, args.forecast_bs):
            ob = times[i:i + args.forecast_bs]
            F_ = fc.forecast(exp, Xc, Mc, ob)[..., :C]                               # b H C
            y = X[ob.to(dev)[:, None] + kh][..., :C]
            E[i:i + len(ob)] = (y - F_).permute(0, 2, 1).cpu()
    base = sum(float(E[i:i + 256][split[i:i + 256] == 2].double().square().sum()) for i in range(0, n, 256)) / nel[2]
    if args.max_c is None and abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    print(f'[base] {base:.5f} fc {fc_mse}', flush=True)

    def bsum(x):                                                                 # (b C H) -> (b NB) float64
        return torch.zeros(x.shape[0], NB, dtype=torch.float64, device=dev).index_add_(1, bk, x.double().sum(1))

    def acc(dst, x, ks_):                                                        # dst (3 NB)
        s_ = spl[ks_]
        for q in (0, 1, 2):
            dst[q] += x[s_ == q].sum(0)

    # ---- term 1: top-k input retrieval, all k at once; keep m for every k (CPU), choose k on val
    sums1 = {k: {'em': torch.zeros(3, NB, dtype=torch.float64, device=dev),
                 'mm': torch.zeros(3, NB, dtype=torch.float64, device=dev)} for k in KS}
    ee = torch.zeros(3, NB, dtype=torch.float64, device=dev)
    Mk = {k: torch.zeros(n, C, H, dtype=torch.float16) for k in KS}
    td = times.to(dev)
    with torch.no_grad():
        for c in torch.arange(C).split(CHR):
            cg = c.to(dev)
            Kn = norm_windows(X, td, cg, L)
            Kn = (Kn / Kn.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half()
            Ec = E[:, c].to(dev)
            for i in range(0, n, BQ):
                kq = torch.arange(i, min(n, i + BQ), device=dev)
                ms = topk_read(Kn, Ec, kq, H, KS)
                e_ = Ec[kq]
                for k in KS:
                    acc(sums1[k]['em'], bsum(e_ * ms[k]), kq)
                    acc(sums1[k]['mm'], bsum(ms[k].square()), kq)
                    Mk[k][i:i + len(kq), c[0]:c[-1] + 1] = ms[k].half().cpu()   # c is a contiguous range
            del Kn, Ec
        for i in range(0, n, 256):
            kq = torch.arange(i, min(n, i + 256), device=dev)
            acc(ee, bsum(E[i:i + len(kq)].to(dev).square()), kq)

    def fit1(em, mm):
        a = torch.where(mm[1] > 0, em[1] / mm[1].clamp_min(EPS), torch.zeros_like(mm[1]))
        return a, (ee - 2 * a * em + a ** 2 * mm).sum(-1)
    k_best = min(KS, key=lambda k: float(fit1(sums1[k]['em'], sums1[k]['mm'])[1][1]))
    a_s, sse = fit1(sums1[k_best]['em'], sums1[k_best]['mm'])
    res = {'S': dict(k=k_best, a=a_s.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                     grid={str(k): float(fit1(sums1[k]['em'], sums1[k]['mm'])[1][1]) / nel[1] for k in KS})}
    MS = Mk[k_best]                                                              # raw m_state (n C H) float16
    del Mk
    # stacking scale from the train-tail origins, so no validation label enters term-2 training
    em0, mm0 = sums1[k_best]['em'][0], sums1[k_best]['mm'][0]
    a_tr = torch.where(mm0 > 0, em0 / mm0.clamp_min(EPS), torch.zeros_like(mm0)).float().cpu()
    res['S']['a_train'] = a_tr.tolist()
    print(f'[S] k {k_best} test {res["S"]["test_mse"]:.5f}', flush=True)

    # ---- term-2 inputs (as eval_decomp)
    mu_w, sd_w = fc.win_stats(X)
    tr_m = split == 0
    trd = tr_m.to(dev)
    dmu = ((mu_w[T0:T0 + n] - mu_w[T0 - 1:T0 + n - 1]) / sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR))[:, :C]
    dls = (torch.log(sd_w[T0:T0 + n].clamp_min(fc.SD_FLOOR)) - torch.log(sd_w[T0 - 1:T0 + n - 1].clamp_min(fc.SD_FLOOR)))[:, :C]
    dmu, dls = dmu / dmu[trd].std().clamp_min(1e-6), dls / dls[trd].std().clamp_min(1e-6)
    sig = sd_w[T0:T0 + n][:, :C].clamp_min(dc.SIG_FLOOR)
    kq96 = torch.arange(-fc.LQ, 0, device=dev)
    nx = fc.LQ + 2
    k_tr_end = int(torch.nonzero(tr_m).max()) + 1

    def x_at(k, c):
        t = T0 + k
        w = (X[t + kq96][:, c].T - mu_w[t, c][:, None]) / sd_w[t, c].clamp_min(fc.SD_FLOOR)[:, None]
        return torch.cat([w, dmu[k, c][:, None], dls[k, c][:, None]], 1)

    def err_feats(D):
        """Pd_err transition input (n C 2 NB + 2) on dev: the newest observed diagonal D[k] / sig_t averaged per
        lead bucket, its change from k - 1, d mu, d log sd; each standardised by its train-tail std, clipped."""
        cnt = torch.zeros(NB, device=dev).index_add_(0, bk, torch.ones(H, device=dev))
        bm = torch.zeros(n, C, NB, device=dev)
        for i in range(0, n, 256):
            d_ = D[i:i + 256].to(dev) / sig[i:i + 256][..., None]
            bm[i:i + 256] = torch.zeros(*d_.shape[:2], NB, device=dev).index_add_(2, bk, d_) / cnt
        dbm = torch.cat([torch.zeros_like(bm[:1]), bm[1:] - bm[:-1]])
        xe = torch.cat([bm, dbm, dmu[..., None], dls[..., None]], -1)
        sd_ = xe[trd].flatten(0, -2).std(0).clamp_min(1e-6)
        return (xe / sd_).clamp(-Z_ERR, Z_ERR)

    def train_carry(R, D, full, mode, xe=None):
        """R (n C H) target residuals, D (n C H) diagonals (mode diag) -- trained on the train-tail indices.
        xe: Pd_err transition inputs (else x_at)."""
        xf = x_at if xe is None else (lambda k, c: xe[k, c])
        torch.manual_seed(seed)
        net = dc.Carry(nx if xe is None else xe.shape[-1], full).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=dc.LR_P)
        g = torch.Generator().manual_seed(seed)
        log = []
        for ep in range(dc.EPOCHS_P):
            tot, cnt = 0.0, 0
            for c in torch.randperm(C, generator=g).split(dc.CH2):
                cg = c.to(dev)
                r_ = R[:k_tr_end, c].to(dev)
                s_ = (D if mode == 'diag' else R)[:k_tr_end, c].to(dev)
                state, loss, nl = None, 0.0, 0
                for k in range(k_tr_end):
                    state, m = carry_step2(net, state, k, T0, mode, lambda q: s_[q], lambda q: xf(q, cg),
                                           lambda q: sig[q, cg], H)
                    if k >= dc.BURN + H:
                        loss = loss + (m - r_[k]).square().mean()
                        nl += 1
                    if (k + 1) % dc.TB == 0 or k == k_tr_end - 1:
                        if nl:
                            opt.zero_grad()
                            (loss / nl).backward()
                            opt.step()
                            tot, cnt = tot + float(loss), cnt + nl
                        loss, nl = 0.0, 0
                        state = state.detach()
            log.append(tot / max(cnt, 1))
        net.eval()
        return net, log

    def carry_pred(net, R, D, mode, xe=None):
        xf = x_at if xe is None else (lambda k, c: xe[k, c])
        state = None
        src = D if mode == 'diag' else R
        with torch.no_grad():
            for k in range(n):
                state, m = carry_step2(net, state, k, T0, mode, lambda q: src[q].to(dev),
                                       lambda q: xf(q, torch.arange(C, device=dev)), lambda q: sig[q], H)
                yield k, m

    def train_tefl(R, D):
        torch.manual_seed(seed)
        net = dc.Tefl(H).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=dc.LR_T)
        lag = torch.arange(K_TEFL, device=dev)
        k_ok = torch.arange(H + K_TEFL - 1, k_tr_end)
        R_tr, D_tr = R[:k_tr_end].to(dev), D[:k_tr_end].to(dev)
        pairs = torch.cartesian_prod(k_ok, torch.arange(C))
        g = torch.Generator().manual_seed(seed)
        log = []
        for ep in range(dc.EPOCHS_T):
            tot = 0.0
            for idx in torch.randperm(len(pairs), generator=g).split(dc.BS_T):
                kk, cc = pairs[idx, 0].to(dev), pairs[idx, 1].to(dev)
                inp = D_tr[kk[:, None] - lag[None], cc[:, None]].flatten(1)
                loss = (net(inp) - R_tr[kk, cc]).square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            log.append(tot / len(pairs))
        del R_tr, D_tr
        net.eval()
        return net, log

    def tefl_pred(net, D):
        with torch.no_grad():
            for k in range(n):
                if k - K_TEFL + 1 < 0:
                    yield k, torch.zeros(C, H, device=dev)
                    continue
                inp = D[k - K_TEFL + 1:k + 1].flip(0).to(dev)                     # lag 0 first, K C H
                yield k, net(inp.transpose(0, 1).flatten(1))

    def eval_sums(gen):
        o = {q: torch.zeros(3, NB, dtype=torch.float64, device=dev) for q in ('ex', 'xx', 'sx', 'ss', 'es')}
        for k, mx in gen:
            q = int(split[k])
            if q < 0:
                continue
            e_, s_, x_ = E[k].to(dev).double(), MS[k].to(dev).double(), mx.double()
            for key, v in (('ex', e_ * x_), ('xx', x_ * x_), ('sx', s_ * x_), ('ss', s_ * s_), ('es', e_ * s_)):
                o[key][q] += torch.zeros(NB, dtype=torch.float64, device=dev).index_add_(0, bk, v.sum(0))
        return o

    def fit_alone(o):
        b = torch.where(o['xx'][1] > 0, o['ex'][1] / o['xx'][1].clamp_min(EPS), torch.zeros_like(o['xx'][1]))
        return b, (ee - 2 * b * o['ex'] + b ** 2 * o['xx']).sum(-1)

    def fit_joint(o):
        A11, A12, A22, b1, b2 = o['ss'][1], o['sx'][1], o['xx'][1], o['es'][1], o['ex'][1]
        det = A11 * A22 - A12 ** 2
        okd = det > 1e-12 * (A11 * A22).clamp_min(EPS)
        s_only = b1 ** 2 / A11.clamp_min(EPS) >= b2 ** 2 / A22.clamp_min(EPS)
        a = torch.where(okd, (A22 * b1 - A12 * b2) / det.clamp_min(EPS),
                        torch.where(s_only, b1 / A11.clamp_min(EPS), torch.zeros_like(b1)))
        b = torch.where(okd, (A11 * b2 - A12 * b1) / det.clamp_min(EPS),
                        torch.where(s_only, torch.zeros_like(b1), b2 / A22.clamp_min(EPS)))
        sse = ee - 2 * a * o['es'] - 2 * b * o['ex'] + a ** 2 * o['ss'] + 2 * a * b * o['sx'] + b ** 2 * o['xx']
        return a, b, sse.sum(-1)

    arm_defs = {'Pd': (True, 'diag'), 'Pd_I': (False, 'diag'), 'Pf': (True, 'full'), 'TEFL': (None, 'diag'),
                'Pd_err': (True, 'diag')}
    arms = [(a, *arm_defs[a]) for a in args.arms.split(',')]
    for tag in ('alone', 'stacked'):
        R = E if tag == 'alone' else E - MS.float() * a_tr[bk.cpu()]
        D = diagonal(R)
        for name, full, mode in arms:
            if name == 'TEFL':
                net, log = train_tefl(R, D)
                o = eval_sums(tefl_pred(net, D))
            else:
                xe = err_feats(D) if name == 'Pd_err' else None
                net, log = train_carry(R, D, full, mode, xe)
                o = eval_sums(carry_pred(net, R, D, mode, xe))
                del xe
            if tag == 'alone':
                b, sse = fit_alone(o)
                res[name] = dict(b=b.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                                 train_loss=log, gamma=float(net.g()) if name != 'TEFL' else None)
                a, b2, sse2 = fit_joint(o)
                res[f'S(+){name}'] = dict(a=a.tolist(), b=b2.tolist(), val_mse=float(sse2[1]) / nel[1],
                                          test_mse=float(sse2[2]) / nel[2])
            else:
                a, b2, sse2 = fit_joint(o)
                res[f'S->{name}'] = dict(a=a.tolist(), b=b2.tolist(), val_mse=float(sse2[1]) / nel[1],
                                         test_mse=float(sse2[2]) / nel[2], train_loss=log)
            print(f'[{tag} {name}] done', flush=True)
        del D, R

    g_ = lambda x: 100 * (base - x) / base
    for k_, v in res.items():
        v['gain'] = g_(v['test_mse'])
        v['vs_fc'] = None if fc_mse is None else 100 * (fc_mse - v['test_mse']) / fc_mse
        print(f'[test] {k_:10s} mse {v["test_mse"]:.5f} gain {v["gain"]:+.3f}% '
              f'vs fc {"-" if v["vs_fc"] is None else "%+.3f%%" % v["vs_fc"]} (val {v["val_mse"]:.5f})', flush=True)
    print(f'[test] base {base:.5f} fc {fc_mse}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               fc_test_mse=fc_mse, settings=dict(M_TR=M_TR, NB=NB, KS=KS, max_c=args.max_c), **res)
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_{args.tag}_s{seed}{"_smoke" if args.max_c else ""}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=32)
    p.add_argument('--max_c', type=int, default=None)
    p.add_argument('--arms', default='Pd,Pd_I,Pf,TEFL')
    p.add_argument('--tag', default='decomp2')
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
