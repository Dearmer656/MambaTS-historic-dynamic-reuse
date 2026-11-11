"""PAT-320 v2: does PaTH error carry beat FreshCast in the multi-task setting? (user hypothesis, 2026-11-12)

Backbone: the joint channel-independent Timer-XL of pat320/joint.py trained with --train_frac 0.5, i.e. on the first half
of every official training split. The second half [t_fit0, train_end) is out of sample for the backbone and is where the
correction arms are fitted (v1 fitted on 60% of val only and every learned head overfitted from the first evaluation).

Arms, all on the same frozen backbone f, all trained jointly across the 7 datasets (one model per arm):
  a  base       y = f (official rolling forecast, 672 -> 720)
  c  nopath     error-carry head of pat320/carry.py, softmax(q . key) over K = 14 one-step error tokens
  d  path       same head, keys transported by the Householder path built from the backbone states (PaTH)
  e  freshcast  FreshCast (pat320/fc.py), one relational-kernel scorer for all datasets, trained on the raw training splits
Fitting: c / d on origins o in [t_fit0 + 96, train_end - 1], loss only on target steps < train_end (val start); memory
tokens only from t >= t_fit0 (out of sample). Early stopping of c / d (every 250 steps, patience 4) on the stop set: 2000
random (variable, origin) pairs per dataset with origin in [val_start, test_start - 720] (whole target inside val).
Combination (same rule for c, d, e): y = f + w_d * Delta, Delta = head output (c, d) or r - f (e), w_d = clip(<e, Delta> /
<Delta, Delta>, 0, 1) per dataset on the stop set, as FreshCast's own combination rule.
Test: every official test origin [test_start, T - 96] and variable; MSE / MAE per H in 96/192/336/720 (prefix of 720).
usage: python pat320/v2.py --ckpt pat_logs/p320v2/joint_tf50_s2021.pth --seed 2021 [--smoke]"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat320.joint import DATA, HS, L, P
from pat320.carry import HMAX, Frozen, Head, build_cache, features, load_full, sample, win
from pat320 import fc

ARMS = ('nopath', 'path', 'freshcast')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--train_frac', type=float, default=0.5); ap.add_argument('--K', type=int, default=14)
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--stop_n', type=int, default=2000); ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--test_bs', type=int, default=4096); ap.add_argument('--data', default=','.join(DATA))
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--out', default=str(ROOT / 'pat_logs' / 'p320v2'))
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.stop_n = 40, 20, 200
        fc.N_TRAIN_Q, fc.EPOCHS, fc.VAL_ORIGINS = 512, 1, 4
    names = a.data.split(','); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs(a.out, exist_ok=True)
    log = lambda s: print(s, flush=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    bb = Frozen(a.ckpt, dev)
    S, C_, T0 = {}, {}, {}
    for d in names:
        x, b = load_full(d); x = torch.tensor(x, device=dev); S[d] = (x, b); T0[d] = int(a.train_frac * b[0])
        bb.set(d); t1 = time.time(); C_[d] = build_cache(bb, x, T0[d])
        log(f'[cache] {d}: C {x.shape[1]} T {x.shape[0]} | fit span [{T0[d]}, {b[0]}) val [{b[0]}, {b[1]}) '
            f'test [{b[1]}, {x.shape[0]}) | {time.time() - t1:.0f}s')
    srng = np.random.default_rng(0)                                   # fixed stop set, independent of the seed
    stop = {}
    for d, (x, b) in S.items():
        v, o = sample(srng, x.shape[1], b[0], b[1] - HMAX, a.stop_n)
        stop[d] = (torch.tensor(v, device=dev), torch.tensor(o, device=dev))

    # ---------------- c / d: error-carry heads ----------------
    def stop_loss(head):
        head.eval(); r = {}
        with torch.no_grad():
            for d, (v, o) in stop.items():
                x, b = S[d]; bb.set(d); tot = 0.0
                for i in range(0, len(o), 512):
                    vv, oo = v[i:i + 512], o[i:i + 512]
                    base, hq, e, hk, ok = features(bb, x, C_[d], T0[d], vv, oo, a.K)
                    tot += (base + head(hq, e, hk, ok) - win(x, vv, oo, HMAX)).square().mean(1).sum().item()
                r[d] = tot / len(o)
        head.train(); return r

    heads, info = {}, {}
    for arm in ('nopath', 'path'):
        torch.manual_seed(a.seed)
        head = Head(arm, bb.m.head.in_features, a.K).to(dev); opt = torch.optim.Adam(head.parameters(), lr=a.lr)
        r0 = stop_loss(head); best = float(np.mean(list(r0.values()))); best_r = r0; best_step = 0; bad = 0
        state = {k: t.clone() for k, t in head.state_dict().items()}
        log(f'[{arm}] params {sum(p.numel() for p in head.parameters())} | step 0 stop {best:.5f} '
            + ' '.join(f'{d} {x:.4f}' for d, x in r0.items()))
        t0 = time.time(); tl = []
        for step in range(1, a.steps + 1):
            loss = 0.0; per = a.bs // len(names)
            for d in names:
                x, b = S[d]; bb.set(d)
                v, o = sample(rng, x.shape[1], T0[d] + P, b[0] - 1, per)
                v, o = torch.tensor(v, device=dev), torch.tensor(o, device=dev)
                base, hq, e, hk, ok = features(bb, x, C_[d], T0[d], v, o, a.K)
                y = win(x, v, o, HMAX)                                # target steps >= val start are not used
                msk = (o[:, None] + torch.arange(HMAX, device=dev)[None] < b[0]).float()
                loss = loss + ((base + head(hq, e, hk, ok) - y).square() * msk).sum() / msk.sum() / len(names)
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
            if step % a.eval_every == 0:
                r = stop_loss(head); m = float(np.mean(list(r.values())))
                log(f'[{arm}] step {step} {time.time() - t0:.0f}s | fit {np.mean(tl[-a.eval_every:]):.5f} | stop {m:.5f} '
                    + ' '.join(f'{d} {x:.4f}' for d, x in r.items()))
                if m < best:
                    best, best_r, best_step, bad = m, r, step, 0
                    state = {k: t.clone() for k, t in head.state_dict().items()}
                else:
                    bad += 1
                    if bad >= a.patience:
                        break
        head.load_state_dict(state); head.eval(); heads[arm] = head
        info[arm] = dict(best_step=best_step, stop0=r0, stop=best_r)
        torch.save(state, os.path.join(a.out, f'head_{arm}_s{a.seed}' + ('_smoke' if a.smoke else '') + '.pth'))

    # ---------------- e: FreshCast ----------------
    ctxs, eps = {}, {}
    for d, (x, b) in S.items():
        ctxs[d] = fc.Ctx(x, b[0], fc.DAY[d])
        vo = torch.arange(b[0], b[1] - HMAX + 1)
        vo = vo[torch.randperm(len(vo), generator=torch.Generator().manual_seed(a.seed))[:fc.VAL_ORIGINS]].sort().values
        eps[d] = fc.untrained_eps(ctxs[d], vo.to(dev))
        log(f'[fc] {d}: untrained eps mean {float(eps[d].mean()):.4f}')
    t1 = time.time(); scorer, fc_hist = fc.train_joint(ctxs, eps, a.seed, log); log(f'[fc] trained {time.time() - t1:.0f}s')
    torch.save(scorer.state_dict(), os.path.join(a.out, f'fc_s{a.seed}' + ('_smoke' if a.smoke else '') + '.pth'))

    @torch.no_grad()
    def deltas(d, v, o):
        """base [B, 720] and the three arms' corrections [B, 720] (before w_d)."""
        x, b = S[d]; bb.set(d)
        base, hq, e, hk, ok = features(bb, x, C_[d], T0[d], v, o, a.K)
        out = {arm: heads[arm](hq, e, hk, ok) for arm in ('nopath', 'path')}
        out['freshcast'] = fc.predict(scorer, ctxs[d], o, v) - base
        return base, out

    # ---------------- combination weights on the stop set ----------------
    W = {}
    for d, (v, o) in stop.items():
        x, b = S[d]; num = {k: 0.0 for k in ARMS}; den = {k: 0.0 for k in ARMS}
        for i in range(0, len(o), 512):
            vv, oo = v[i:i + 512], o[i:i + 512]
            base, dl = deltas(d, vv, oo); err = win(x, vv, oo, HMAX) - base
            for k in ARMS:
                num[k] += (err * dl[k]).sum().item(); den[k] += dl[k].square().sum().item()
        W[d] = {k: min(max(num[k] / max(den[k], 1e-12), 0.0), 1.0) for k in ARMS}
        log(f'[w] {d} ' + ' '.join(f'{k} {W[d][k]:.3f}' for k in ARMS))

    # ---------------- test ----------------
    res = dict(seed=a.seed, ckpt=a.ckpt, train_frac=a.train_frac, K=a.K, info=info, fc_loss=fc_hist, w=W, test={})
    for d, (x, b) in S.items():
        T, C = x.shape; t1 = time.time()
        acc = {k: {H: [0.0, 0.0] for H in HS} for k in ('base',) + ARMS}; cnt = {H: 0 for H in HS}
        vs, os_ = torch.meshgrid(torch.arange(C, device=dev), torch.arange(b[1], T - HS[0] + 1, device=dev), indexing='ij')
        vs, os_ = vs.reshape(-1), os_.reshape(-1)
        for i in range(0, len(os_), a.test_bs):
            v, o = vs[i:i + a.test_bs], os_[i:i + a.test_bs]
            base, dl = deltas(d, v, o)
            preds = {'base': base, **{k: base + W[d][k] * dl[k] for k in ARMS}}
            for H in HS:
                m = o <= T - H
                if m.any():
                    y = win(x, v[m], o[m], H); cnt[H] += y.numel()
                    for k, pr in preds.items():
                        df = pr[m, :H] - y
                        acc[k][H][0] += df.square().sum().item(); acc[k][H][1] += df.abs().sum().item()
        res['test'][d] = {k: {H: dict(mse=acc[k][H][0] / cnt[H], mae=acc[k][H][1] / cnt[H]) for H in HS} for k in acc}
        rr = res['test'][d]
        log(f'[test] {d} | ' + ' | '.join(f'{k} {np.mean([rr[k][H]["mse"] for H in HS]):.4f}' for k in rr)
            + f' | {time.time() - t1:.0f}s')
    json.dump(res, open(os.path.join(a.out, f'v2_s{a.seed}' + ('_smoke' if a.smoke else '') + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
