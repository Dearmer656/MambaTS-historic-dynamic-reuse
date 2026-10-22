"""PAT-311 Timer-XL + in-context error feedback trained EXACTLY with Timer-XL's recipe (user design, 2026-10-22):
the official objective L_tf (teacher-forced next-token MSE over the 7 positions), the official train windows (96-step future), early
stopping on the official valid_last loss (the last position's next-patch MSE on the val windows), lradj type1, Adam, bs 32, 10 epochs,
patience 3. Backbone models/timer_xl.py unchanged (use_norm done out of place by fb_timerxl.bb_forward, the same operations).
ctrl = no module = the official Timer-XL recipe re-implemented.

Errors (teacher-forced pass, detached): token j's prediction error  e_{j+1} = (x_{j+1} - out_j) / sd,  j = 0..N-2  (N = 7).
The error is attached to the token that MADE the prediction (token j): value v_j = U e_{j+1} (U: P -> r), key k_j = W_K z_j.
errpath (PaTH convention: the path between key j and query i is H_{j+1} ... H_i, it includes the query's own H_i and excludes H_j):
  position i (i = 1..N-1) corrects its own next-patch forecast with the errors already observed there (j <= i-1, causal):
    delta_i = W2 ReLU(W1 [ sum_{j<=i-1} softmax_j(q_i . k_j / tau) (H_i ... H_{j+1}) v_j ]_heads),  q_i = W_Q z_i
    e.g. e_1 (token 0) reaches position 1 through H_1 (one step), position 2 through H_1 H_2 (two steps), ...
  position 0 has no observed error (not corrected): 6 corrected forecasts per window; L_tf is taken on the corrected outputs.
  H_s = I - beta_s w_s w_s^T in R^r, w_s = normalize(W_w z_s), beta_s = 2 sigmoid(W_b z_s); z (head input) detached; nh = 4, d_k = 8.
Rolling test (the official loop, corrected forecasts fed back): the memory is the lookback's errors at the origin ONLY (the forecasts
inside the rollout have no observed error); step 0 = position N-1 of the teacher-forced pass; step k >= 1: the values continue along
the path through the state of the window's new last token, v <- H(z_last(k)) v (e_{j+1} has crossed (N-1-j) + k transitions), keys
stay, query = W_Q z_last(k).
Arms: ctrl | errpath | o (H = I: the path ablation) | t0r (TEFL-style: delta_i = W2 ReLU(W1 [e_{i-1}, e_i]), zeros where absent; in the
rollout the origin's [e_{N-2}, e_{N-1}] for every step). W2 = 0: every arm == ctrl at init. Module lr = 10x the backbone lr (fixed ratio).
Error dropout (--keep p < 1, training only; user 2026-10-23, against the in-sample-error generalisation gap): per sample, variable and
lookback slot, the error e_{j+1} is kept with probability p and scaled by 1/p (inverted dropout: E[train input] = the test input), else 0.
Audits: zero | noH (errpath, H = I at inference) | uniform (o / errpath: a = 1 / #observed slots).
usage: python pat_fb/tf_timerxl.py --data E1|W --arm errpath --rank 8 --seed 2021"""
import argparse, json, os, sys, time
from math import sqrt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, HS, L, P, bb_forward, split_x, windows
from pat_fb.joint_timerxl import LR, build_backbone, forward_states


class Module(nn.Module):
    def __init__(self, arm, d_model, rank=8, nh=4, dk=8, keep=1.0):
        super().__init__()
        assert arm in ('ctrl', 't0r', 'o', 'errpath'), arm
        assert 0 < keep <= 1, keep
        self.arm, self.r, self.nh, self.dk, self.keep, self.audit = arm, rank, nh, dk, keep, None
        if arm == 't0r':
            self.W1 = nn.Linear(2 * P, rank, bias=False)
        if arm in ('o', 'errpath'):
            self.U = nn.Linear(P, rank, bias=False)
            self.Wq = nn.Linear(d_model, nh * dk, bias=False)
            self.Wk = nn.Linear(d_model, nh * dk, bias=False)
            self.log_tau = nn.Parameter(torch.zeros(nh))
            self.W1 = nn.Linear(nh * rank, rank, bias=False)
        if arm == 'errpath':
            self.Ww = nn.Linear(d_model, rank, bias=False)
            self.Wb = nn.Linear(d_model, 1)
        if arm != 'ctrl':
            self.W2 = nn.Linear(rank, P, bias=False)
            nn.init.zeros_(self.W2.weight)

    def householder(self, z, V):
        """V (b, C, M, r) <- H(z) V, H(z) = I - beta w w^T, z (b, C, D); identity for 'o' and under audit noH."""
        if self.arm != 'errpath' or self.audit == 'noH' or V.shape[2] == 0:
            return V
        z = z.detach()
        w = F.normalize(self.Ww(z), dim=-1)
        beta = 2 * torch.sigmoid(self.Wb(z))
        return V - (beta * torch.einsum('bcmr,bcr->bcm', V, w)).unsqueeze(-1) * w[:, :, None]

    def read(self, z_q, V, K):
        """delta (b, C, P) from the query state z_q (b, C, D) over the slots V (b, C, M, r), K (b, C, M, nh, dk)."""
        b, c, M = V.shape[:3]
        if self.audit == 'uniform':
            a = V.new_full((b, c, self.nh, M), 1.0 / M)
        else:
            q = self.Wq(z_q.detach()).unflatten(-1, (self.nh, self.dk))
            logit = torch.einsum('bchd,bcmhd->bchm', q, K) / (sqrt(self.dk) * self.log_tau.exp())[None, None, :, None]
            a = torch.softmax(logit, -1)
        return self.W2(torch.relu(self.W1(torch.einsum('bchm,bcmr->bchr', a, V).flatten(2))))

    def teacher_forced(self, E, Z):
        """Corrections of positions 1..N-1 of the teacher-forced pass: D (b, C, N-1, P) (D[:, :, i-1] corrects position i) and the
        rollout memory (the slots transported up to position N-1). E (b, C, N-1, P) = e_1..e_{N-1}, Z (b, C, N, D)."""
        b, c, n1 = E.shape[:3]
        if self.arm == 't0r':
            Ep = torch.cat([torch.zeros_like(E[:, :, :1]), E], 2)                          # Ep[:, :, i] = e_i (e_0 = 0)
            feat = torch.cat([Ep[:, :, :-1], Ep[:, :, 1:]], -1)                             # position i: [e_{i-1}, e_i], i = 1..N-1
            D = self.W2(torch.relu(self.W1(feat)))
            return D, dict(last=D[:, :, -1])
        V0 = self.U(E)                                                                      # v_j = U e_{j+1}, token j = 0..N-2
        K0 = self.Wk(Z[:, :, :-1].detach()).unflatten(-1, (self.nh, self.dk))               # k_j from token j
        Vt, out = V0[:, :, :0], []
        for i in range(1, n1 + 1):                                                          # v_{i-1} joins, then H_i acts on all
            Vt = self.householder(Z[:, :, i], torch.cat([Vt, V0[:, :, i - 1:i]], 2))        # Vt[:, :, j] = H_i ... H_{j+1} v_j
            out.append(self.read(Z[:, :, i], Vt, K0[:, :, :i]))
        return torch.stack(out, 2), dict(V=Vt, K=K0)

    def roll(self, mem, z_last):
        """Rollout step k >= 1: the lookback memory continues along the path through the new last token's state."""
        if self.arm == 't0r':
            return mem['last']
        mem['V'] = self.householder(z_last, mem['V'])
        return self.read(z_last, mem['V'], mem['K'])


def tf_pass(bb, mod, x):
    """Teacher-forced pass with the corrections: out (b, L, C) corrected, sd (b, 1, C), memory, Z."""
    b, _, c = x.shape
    out, sd, Z = forward_states(bb, x)
    sd0 = sd.detach()
    if mod.arm == 'ctrl' or mod.audit == 'zero':
        return out, sd0, None
    E = ((x[:, P:] - out[:, :-P]) / sd0).reshape(b, L // P - 1, P, c).permute(0, 3, 1, 2).detach()
    if mod.training and mod.keep < 1:                                                       # inverted slot dropout on the errors
        E = E * ((torch.rand(E.shape[:3], device=E.device) < mod.keep).to(E.dtype) / mod.keep).unsqueeze(-1)
    D, mem = mod.teacher_forced(E, Z)                                                       # (b, C, N-1, P)
    corr = torch.cat([torch.zeros_like(D[:, :, :1]), D], 2).permute(0, 2, 3, 1).reshape(b, L, c) * sd0   # position 0 uncorrected
    return out + corr, sd0, mem


def rollout(bb, mod, x, n_out):
    out, sd0, mem = tf_pass(bb, mod, x)
    preds, xw = [out[:, -P:]], x
    for k in range(1, n_out):
        xw = torch.cat([xw[:, P:], preds[-1]], 1)
        o_k, _, Z_k = forward_states(bb, xw)
        d = None if mem is None else mod.roll(mem, Z_k[:, :, -1])
        preds.append(o_k[:, -P:] if d is None else o_k[:, -P:] + d.transpose(1, 2) * sd0)
    return torch.cat(preds, 1)


@torch.no_grad()
def evaluate(bb, mod, xs, H, dev, bs=64):
    n = len(xs) - L - H + 1
    se, ae = 0.0, 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), H, dev)
        p = rollout(bb, mod, x, -(-H // P))[:, :H]
        se += (p - y).square().sum().item(); ae += (p - y).abs().sum().item()
    nel = n * H * xs.shape[1]
    return se / nel, ae / nel


@torch.no_grad()
def valid_last(bb, mod, xs, dev, bs=32):
    """The official early-stopping loss (exp_forecast.vali, valid_last): the last position's (corrected) next-patch MSE, the batch
    means averaged with batch-size weights, val windows with a P-step future, batch size = the training batch size."""
    n = len(xs) - L - P + 1
    tot = 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), P, dev)
        tot += (tf_pass(bb, mod, x)[0][:, -P:] - y).square().mean().item() * x.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E1', 'W']); ap.add_argument('--arm', default='errpath')
    ap.add_argument('--rank', type=int, default=8); ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--mod_lr_mult', type=float, default=10.0); ap.add_argument('--out', default=None)
    ap.add_argument('--keep', type=float, default=1.0, help='error (slot) keep probability in training, inverted dropout')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bb = build_backbone(a.data).to(dev)
    mod = Module(a.arm, CFG[a.data]['d_model'], rank=a.rank, keep=a.keep).to(dev)
    xs = {f: split_x(a.data, f) for f in ('train', 'val', 'test')}
    lr = LR[a.data]
    groups = [dict(params=list(bb.parameters()), lr=lr, base=lr)]
    if list(mod.parameters()):
        groups.append(dict(params=list(mod.parameters()), lr=lr * a.mod_lr_mult, base=lr * a.mod_lr_mult))
    opt = torch.optim.Adam(groups)
    ntr = len(xs['train']) - L - P + 1                                                      # the official train windows
    tag = a.arm + ('' if a.arm == 'ctrl' else f'_r{a.rank}') + (f'_k{a.keep:g}' if a.keep < 1 else '')
    npar = sum(p.numel() for p in mod.parameters())
    print(f'[tf] {a.data} arm {tag} | module params {npar} | train windows {ntr} | lr {lr} (module x{a.mod_lr_mult}) | official recipe', flush=True)
    path = os.path.join(ROOT, 'checkpoints', 'pat311_tf', f'{a.data}_{tag}_s{a.seed}.pth')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best, bad = float('inf'), 0
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        bb.train(); mod.train(); t0 = time.time(); tl = []
        perm = rng.permutation(ntr)
        for idx in (perm[i0:i0 + a.bs] for i0 in range(0, ntr, a.bs)):                     # as the official DataLoader: bs, last smaller
            x, y = windows(xs['train'], idx.tolist(), P, dev)
            out = tf_pass(bb, mod, x)[0]
            loss = (out - torch.cat([x[:, P:], y], 1)).square().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        bb.eval(); mod.eval()
        v = valid_last(bb, mod, xs['val'], dev, a.bs)
        print(f'[tf] epoch {ep + 1} {time.time() - t0:.0f}s | train {np.mean(tl):.5f} | valid_last {v:.5f}', flush=True)
        if v < best:
            best, bad = v, 0
            torch.save(dict(bb=bb.state_dict(), mod=mod.state_dict()), path)
        else:
            bad += 1
            if bad >= a.patience:
                break
        for g in opt.param_groups:                                                          # Timer-XL lradj type1
            g['lr'] = g['base'] * (0.1 ** (ep + 1))
    ck = torch.load(path, map_location=dev)
    bb.load_state_dict(ck['bb']); mod.load_state_dict(ck['mod']); bb.eval(); mod.eval()
    res = dict(data=a.data, arm=a.arm, rank=a.rank, keep=a.keep, seed=a.seed, params=npar, best_val=best, test={})
    audits = {'t0r': ('zero',), 'o': ('zero', 'uniform'), 'errpath': ('zero', 'noH', 'uniform')}.get(a.arm, ())
    for H in HS:
        mse, mae = evaluate(bb, mod, xs['test'], H, dev)
        r = dict(mse=mse, mae=mae)
        for au in audits:
            mod.audit = au; r[au] = evaluate(bb, mod, xs['test'], H, dev)[0]; mod.audit = None
        res['test'][H] = r
        print(f'[tf] test H{H} | mse {mse:.4f} mae {mae:.4f}' + ''.join(f' | {au} {r[au]:.4f}' for au in audits), flush=True)
    out = a.out or os.path.join(ROOT, 'pat_logs', 'tf', f'{a.data}_{tag}_s{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
