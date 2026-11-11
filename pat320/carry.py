"""PAT-320 stage 2: error-carry head on the FROZEN joint Timer-XL (pat320/joint.py), trained across all 7 datasets.

Memory of origin o (one variable): K one-step error tokens E[o - 96k], k = 1..K, with E[t] = y[t:t+96] - f(x[t-672:t])[:96]
(the backbone's first rolled token), plus the backbone state h_t = last-layer output of the last input token of that window.
Errors from training-split time steps are in-sample for the backbone and are masked out (only val/test errors are used).
Readout: Delta(o) = W_out sum_k a_k U E[o-96k], a = softmax_k(q_o . P_{o,k} key_k / sqrt(dh)) over unmasked k,
  P_{o,k} = H_{o-96(k-1)} ... H_{o}  applied to key_k (Householder H_t = I - beta_t w_t w_t^T, w_t, beta_t from h_t);
  arm 'nopath': P = I; arm 'path': as above. U: 96 -> r, W_out: r -> 720 (zero init). y_hat = rollout(o) + Delta(o).
Fit: origins in the first 60% of every dataset's val split (target fully inside val), early stop on origins after a gap of
L + 720 steps; equal weight per dataset. Test: every official test window (origin = start + 672), per H in 96/192/336/720.
usage: python pat320/carry.py --ckpt pat_logs/p320/joint_s2021.pth --arm path --seed 2021"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat320.joint import DATA, HS, L, P, build_model, set_task

HMAX = HS[-1]


def load_full(name):
    """Whole scaled series [T, C] (scaler fit on train) and split starts (train_end, val_end) exactly as OpenLTM."""
    df = pd.read_csv(ROOT / 'dataset' / DATA[name])
    if name.startswith('ETTh'):
        b = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif name.startswith('ETTm'):
        b = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        n = len(df); ntr, nte = int(n * 0.7), int(n * 0.2); b = [ntr, n - nte, n]
    data = df[df.columns[1:]].values if isinstance(df[df.columns[0]][2], str) else df.values
    sc = StandardScaler().fit(data[:b[0]])
    return sc.transform(data[:b[2]]).astype(np.float32), b


class Frozen:
    """Backbone with a hook returning the last-layer state of the last input token."""
    def __init__(self, ckpt, dev):
        sd = torch.load(ckpt, map_location=dev); self.task_emb = 'task_emb.weight' in sd   # joint_te_* checkpoints
        self.m = build_model(dev, self.task_emb); self.m.load_state_dict(sd); self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self._h = None
        self.m.blocks.register_forward_hook(lambda mod, inp, out: setattr(self, '_h', out[0]))

    def set(self, d):
        """Dataset of the following calls (used only by task-embedding backbones)."""
        set_task(self.m, d)

    @torch.no_grad()
    def step(self, x):
        """x [B, L, 1] -> (first forecast token [B, 96], state [B, D]). use_norm acts inside; state is pre-head."""
        y = self.m(x, None, None)[:, -P:, 0]
        return y, self._h[:, -1]

    @torch.no_grad()
    def rollout(self, x, H=HMAX):
        out = []
        for _ in range(math.ceil(H / P)):
            y = self.m(x, None, None)[:, -P:]; out.append(y); x = torch.cat([x[:, P:], y], 1)
        return torch.cat(out, 1)[:, :H, 0]


def win(x, v, s, n):
    """x [T, C] tensor; v, s [B] -> x[s:s+n, v] as [B, n]."""
    return x[s[:, None] + torch.arange(n, device=x.device), v[:, None]]


@torch.no_grad()
def build_cache(bb, x, t0, bs=8192):
    """One-step error token E[v, t] = y[t:t+96] - f(x[t-672:t])[:96] and state h[v, t] for every t in [t0, T-96]
    (t0 = end of the training split: earlier errors are in-sample). fp16, on the device. Index t - t0."""
    T, C = x.shape; ts = torch.arange(t0, T - P + 1, device=x.device)
    E = torch.empty(C, len(ts), P, dtype=torch.float16, device=x.device)
    Hs = torch.empty(C, len(ts), bb.m.head.in_features, dtype=torch.float16, device=x.device)
    vv, tt = torch.meshgrid(torch.arange(C, device=x.device), ts, indexing='ij'); vv, tt = vv.reshape(-1), tt.reshape(-1)
    for i in range(0, len(tt), bs):
        v, t = vv[i:i + bs], tt[i:i + bs]
        f1, h = bb.step(win(x, v, t - L, L).unsqueeze(-1))
        E[v, t - t0] = (win(x, v, t, P) - f1).half(); Hs[v, t - t0] = h.half()
    return E, Hs


@torch.no_grad()
def features(bb, x, cache, t0, v, o, K):
    """Origins o (variable v, o >= t0): base rollout [B, 720], query state [B, D], error tokens [B, K, 96],
    their states [B, K, D], mask [B, K] (True = error token from a non-training time step)."""
    E, Hs = cache; B = len(o)
    base = bb.rollout(win(x, v, o - L, L).unsqueeze(-1))
    hq = Hs[v, o - t0].float()
    t = o[:, None] - P * torch.arange(1, K + 1, device=x.device)
    ok = t >= t0
    ti = (t - t0).clamp(min=0); vv = v[:, None].expand(B, K)
    return base, hq, E[vv, ti].float(), Hs[vv, ti].float(), ok


class Head(nn.Module):
    def __init__(self, arm, d, K, r=16, dh=64, nh=4):
        super().__init__()
        self.arm, self.K, self.nh, self.dh = arm, K, nh, dh
        self.q = nn.Linear(d, nh * dh); self.k = nn.Linear(d + P, nh * dh)
        self.w = nn.Linear(d, nh * dh); self.b = nn.Linear(d, nh)
        self.U = nn.Linear(P, r); self.out = nn.Linear(r * nh, HMAX)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, hq, e, hk, ok):
        B, K, _ = e.shape; nh, dh = self.nh, self.dh
        q = self.q(hq).view(B, nh, dh)
        key = self.k(torch.cat([hk, e], -1)).view(B, K, nh, dh)
        if self.arm == 'path':
            # states along the path, newest first: index 0 = origin o, index k = token o - 96k
            hs = torch.cat([hq[:, None], hk], 1)                                       # [B, K+1, D]
            w = nn.functional.normalize(self.w(hs).view(B, K + 1, nh, dh), dim=-1)
            beta = 2 * torch.sigmoid(self.b(hs)).unsqueeze(-1)                          # [B, K+1, nh, 1]
            outk = []
            for k in range(K):                                       # key_k is transported by H_{k-1..0} (newer tokens)
                z = key[:, k]
                for m in range(k - 1, -1, -1):                       # H_{o-96m}, applied from the oldest to the newest
                    z = z - beta[:, m] * w[:, m] * (w[:, m] * z).sum(-1, keepdim=True)
                outk.append(z)
            key = torch.stack(outk, 1)
        s = torch.einsum('bhd,bkhd->bhk', q, key) / math.sqrt(dh)
        s = s.masked_fill(~ok[:, None], -1e9)
        a = torch.softmax(s, -1)
        a = a * ok.any(-1)[:, None, None].float()                    # no usable memory -> no correction
        z = torch.einsum('bhk,bkr->bhr', a, self.U(e))
        return self.out(z.reshape(B, -1))


def origins_val(T_b, K):
    """Fit / stop origin ranges inside the val split (v1 = test start). Stop origins [cut, v1 - 720] have their whole
    720-step target inside val; fit origins [v0, cut) are trained only on target steps before cut (loss mask), so no fit
    target step is a stop target step. (Version 1 required whole fit targets plus a gap of L + 720, which left Traffic,
    val = 1756 steps, with no fit origins.)"""
    v0, v1 = T_b[0], T_b[1]
    lo, hi = v0, v1 - HMAX
    cut = lo + int(0.6 * (hi - lo))
    return (lo, cut - 1), (cut, hi)


def sample(rng, C, lo, hi, n):
    v = rng.integers(0, C, n); o = rng.integers(lo, hi + 1, n); return v, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arm', default='path', choices=['path', 'nopath'])
    ap.add_argument('--seed', type=int, default=2021); ap.add_argument('--K', type=int, default=14)
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--stop_n', type=int, default=2000); ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--data', default=','.join(DATA)); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--test_bs', type=int, default=1024); ap.add_argument('--out', default=str(ROOT / 'pat_logs' / 'p320'))
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.eval_every, a.stop_n = 40, 20, 200
    names = a.data.split(','); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    bb = Frozen(a.ckpt, dev)
    S = {}
    for d in names:
        x, b = load_full(d); S[d] = (torch.tensor(x, device=dev), b)
    C_ = {}
    for d, (x, b) in S.items():
        bb.set(d); t1 = time.time(); C_[d] = build_cache(bb, x, b[0])
        print(f'[cache] {d}: C {x.shape[1]} T {x.shape[0]} | {time.time() - t1:.0f}s', flush=True)
    head = Head(a.arm, bb.m.head.in_features, a.K).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    srng = np.random.default_rng(0)                                   # fixed stop set, independent of the seed
    stop = {}
    for d, (x, b) in S.items():
        (_, _), (slo, shi) = origins_val(b, a.K)
        v, o = sample(srng, x.shape[1], slo, shi, a.stop_n)
        stop[d] = (torch.tensor(v, device=dev), torch.tensor(o, device=dev))

    def stop_loss():
        head.eval(); r = {}
        with torch.no_grad():
            for d, (v, o) in stop.items():
                x, b = S[d]; bb.set(d); tot = 0.0
                for i in range(0, len(o), 512):
                    vv, oo = v[i:i + 512], o[i:i + 512]
                    base, hq, e, hk, ok = features(bb, x, C_[d], b[0], vv, oo, a.K)
                    y = win(x, vv, oo, HMAX)
                    tot += (base + head(hq, e, hk, ok) - y).square().mean(1).sum().item()
                r[d] = tot / len(o)
        head.train(); return r

    r0 = stop_loss(); best = float(np.mean(list(r0.values()))); best_r = r0
    state = {k: t.clone() for k, t in head.state_dict().items()}; bad = 0; best_step = 0
    print(f'[carry] {a.arm} K{a.K} | params {sum(p.numel() for p in head.parameters())} | step 0 stop {best:.5f} '
          + ' '.join(f'{d} {x:.4f}' for d, x in r0.items()), flush=True)
    t0 = time.time(); tl = []
    for step in range(1, a.steps + 1):
        loss = 0.0
        per = a.bs // len(names)
        for d in names:
            x, b = S[d]; bb.set(d); (flo, fhi), _ = origins_val(b, a.K)
            v, o = sample(rng, x.shape[1], flo, fhi, per)
            v, o = torch.tensor(v, device=dev), torch.tensor(o, device=dev)
            base, hq, e, hk, ok = features(bb, x, C_[d], b[0], v, o, a.K)
            cut = fhi + 1; y = win(x, v, o, HMAX)                       # target steps >= cut belong to the stop set
            msk = (o[:, None] + torch.arange(HMAX, device=dev)[None] < cut).float()
            loss = loss + ((base + head(hq, e, hk, ok) - y).square() * msk).sum() / msk.sum() / len(names)
        opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        if step % a.eval_every == 0:
            r = stop_loss(); m = float(np.mean(list(r.values())))
            print(f'[carry] step {step} {time.time() - t0:.0f}s | fit {np.mean(tl[-a.eval_every:]):.5f} | stop {m:.5f} '
                  + ' '.join(f'{d} {x:.4f}' for d, x in r.items()), flush=True)
            if m < best:
                best, best_r, best_step, bad = m, r, step, 0; state = {k: t.clone() for k, t in head.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
    head.load_state_dict(state); head.eval()
    res = dict(arm=a.arm, K=a.K, seed=a.seed, ckpt=a.ckpt, best_step=best_step, stop0=r0, stop=best_r, test={})
    for d, (x, b) in S.items():
        bb.set(d); T, C = x.shape
        se = {H: [0.0, 0.0] for H in HS}; se0 = {H: 0.0 for H in HS}; cnt = {H: 0 for H in HS}
        lo, hi = b[1], T - HS[0]                                     # official test origins (start + 672)
        vs, os_ = torch.meshgrid(torch.arange(C, device=dev), torch.arange(lo, hi + 1, device=dev), indexing='ij')
        vs, os_ = vs.reshape(-1), os_.reshape(-1)
        with torch.no_grad():
            for i in range(0, len(os_), a.test_bs):
                v, o = vs[i:i + a.test_bs], os_[i:i + a.test_bs]
                base, hq, e, hk, ok = features(bb, x, C_[d], b[0], v, o, a.K)
                corr = base + head(hq, e, hk, ok)
                for H in HS:
                    m = o <= T - H
                    if m.any():
                        y = win(x, v[m], o[m], H)
                        d1 = corr[m, :H] - y; d0 = base[m, :H] - y
                        se[H][0] += d1.square().sum().item(); se[H][1] += d1.abs().sum().item()
                        se0[H] += d0.square().sum().item(); cnt[H] += d1.numel()
        res['test'][d] = {H: dict(mse=se[H][0] / cnt[H], mae=se[H][1] / cnt[H], base_mse=se0[H] / cnt[H]) for H in HS}
        rr = res['test'][d]
        print(f'[test] {d} | ' + ' | '.join(f'H{H} {rr[H]["mse"]:.4f} (base {rr[H]["base_mse"]:.4f})' for H in HS)
              + f' | avg {np.mean([rr[H]["mse"] for H in HS]):.4f} vs {np.mean([rr[H]["base_mse"] for H in HS]):.4f}', flush=True)
    tag = f'carry_{a.arm}_K{a.K}_s{a.seed}_' + Path(a.ckpt).stem + ('_smoke' if a.smoke else '')
    json.dump(res, open(os.path.join(a.out, f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
