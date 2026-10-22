"""PAT-311 Timer-XL + in-context error feedback, JOINT training (TEFL-style: backbone and module trained together, from scratch).
Backbone: models/timer_xl.py unchanged (its use_norm done out of place by fb_timerxl.bb_forward, the same operations).

Per batch (window x = N = L/P patches, future y = Hmax patches):
  1 teacher-forced pass (the official Timer-XL objective): out = bb(x), token n predicts patch n+1;  L_tf = MSE(out, [x_1..x_{N-1}, y_0])
  2 in-context errors from the SAME pass (detached): e_{n+1} = (x_{n+1} - out_n) / sd, n = 0..N-2; token states z_n = head input
  3 rolling forecast of Hmax patches (the official test loop, the CORRECTED forecast fed back), step k: base_k + sd * delta_k
  L = L_tf + L_roll,  L_roll = mean over H in (96, 192, 336, 720) of the MSE of the first H forecast steps (= the evaluation's average;
  the early patches count in every term instead of being dominated by the large far-horizon errors)
Arms (per variable, module shared across variables, all deltas zero at init):
  ctrl     no module (same recipe: L_tf + L_roll)                                             -> isolates the effect of the errors
  t0r      delta_k = block k of W2 ReLU(W1 [e_{N-2}, e_{N-1}])   (W1: 2P -> r, W2: r -> Hmax*P)      (TEFL-style module, PAT-307 T0r)
  o        H = I ablation of errpath (retrieval, no path)
  errpath  (the PAT-307 O-ErrPath design, user 2026-10-21: path on the retrieved ERRORS, logits plain q.k; here in the low-rank space)
           value of lookback token j (j = 0..N-2): v_j = U e_{j+1} (U: P -> r), same-variable slots (as R1's slots)
           path H_s = I - beta_s w_s w_s^T in R^r, w_s = normalize(W_w z_s), beta_s = 2 sigmoid(W_b z_s), z_s detached
           at the origin v_j <- H_{N-1} ... H_{j+2} v_j (the states of the tokens after the error's own patch j+1, as R1Obs vpath);
           rollout step k >= 1: every value <- H(z of the new last token) v (the path continues through the rollout states)
           step k: m = sum_j softmax_j(q . k_j / tau) v_j per head (q = W_Q z_last(k), k_j = W_K z_{j+1}: the state of the token whose
           input is the error's patch, as R1Obs; nh = 4, d_k = 8, z detached),
           delta_k = W2 ReLU(W1 [m_1..m_nh]), W1: nh*r -> r, W2: r -> P (zero)
Audits: 'zero' (delta = 0) | 'noH' (errpath with H = I at inference) | 'uniform' (o / errpath: a = 1 / (N-1)).
Selection: early stop on the val L_roll (patience 3); lr: Timer-XL's lradj type1 (x0.1 per epoch) on both groups, module lr = 10x the
backbone lr at every epoch (a fixed ratio, fixed in advance). The official Timer-XL reproduction is an external reference; ctrl is the
causal control of the module (same recipe, windows, selection).
usage: python pat_fb/joint_timerxl.py --data E1|W --arm errpath --rank 8 --seed 2021"""
import argparse, json, os, sys, time
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models import timer_xl
from pat_fb.fb_timerxl import CFG, HMAX, HS, L, P, bb_forward, split_x, windows

LR = {'E1': 1e-4, 'W': 5e-4}


def build_backbone(D):
    c = CFG[D]
    cfg = SimpleNamespace(input_token_len=P, output_token_len=P, d_model=c['d_model'], d_ff=c['d_ff'], n_heads=8, e_layers=c['e_layers'],
                          dropout=0.1, activation='relu', output_attention=False, covariate=False, flash_attention=False, use_norm=False)
    return timer_xl.Model(cfg)                                                              # use_norm done by bb_forward


class Module(nn.Module):
    def __init__(self, arm, d_model, rank=8, hc=2, nh=4, dk=8, hmax=HMAX):
        super().__init__()
        assert arm in ('ctrl', 't0r', 'o', 'errpath'), arm
        self.arm, self.r, self.nh, self.dk, self.hc, self.hmax, self.audit = arm, rank, nh, dk, hc, hmax, None
        if arm == 't0r':
            self.W1 = nn.Linear(hc * P, rank, bias=False)
            self.W2 = nn.Linear(rank, hmax * P, bias=False)
        if arm in ('o', 'errpath'):
            self.U = nn.Linear(P, rank, bias=False)
            self.Wq = nn.Linear(d_model, nh * dk, bias=False)
            self.Wk = nn.Linear(d_model, nh * dk, bias=False)
            self.log_tau = nn.Parameter(torch.zeros(nh))
            self.W1 = nn.Linear(nh * rank, rank, bias=False)
            self.W2 = nn.Linear(rank, P, bias=False)
        if arm == 'errpath':
            self.Ww = nn.Linear(d_model, rank, bias=False)
            self.Wb = nn.Linear(d_model, 1)
        if arm != 'ctrl':
            nn.init.zeros_(self.W2.weight)

    def householder(self, z, V):
        """V (b, C, M, r) <- H(z) V with H(z) = I - beta w w^T, z (b, C, D) one state per variable; identity for 'o' / audit noH."""
        if self.arm != 'errpath' or self.audit == 'noH':
            return V
        z = z.detach()
        w = F.normalize(self.Ww(z), dim=-1)                                                 # (b, C, r)
        beta = 2 * torch.sigmoid(self.Wb(z))                                                # (b, C, 1)
        return V - (beta * torch.einsum('bcmr,bcr->bcm', V, w)).unsqueeze(-1) * w[:, :, None]

    def init_memory(self, E, Z):
        """E (b, C, N-1, P) errors e_1..e_{N-1}, Z (b, C, N, D) token states -> memory dict (values transported to the origin)."""
        if self.arm in ('ctrl', 't0r'):
            return dict(E=E)
        V = self.U(E)                                                                       # (b, C, N-1, r): token j -> e_{j+1}
        N = Z.shape[2]
        if self.arm == 'errpath':                                                           # v_j <- H_{N-1} ... H_{j+2} v_j
            Vt = V[:, :, :1]
            for s in range(2, N):                                                           # state of token s acts on the older values,
                Vt = torch.cat([self.householder(Z[:, :, s], Vt), V[:, :, s - 1:s]], 2)     # then the value of token s-1 (e_s) joins
            V = Vt
        # key of e_{j+1}: the state of token j+1 (the token whose input is the error's patch), as R1Obs keys slot t by bstates[:, t]
        return dict(E=E, V=V, K=self.Wk(Z[:, :, 1:].detach()).unflatten(-1, (self.nh, self.dk)))    # keys (b, C, N-1, nh, dk)

    def step(self, mem, k, z_last):
        """delta_k (b, C, P) for rollout step k; z_last (b, C, D) = the state of the window's last token at this step."""
        if self.arm == 'ctrl' or self.audit == 'zero':
            return None
        if self.arm == 't0r':
            if k == 0:
                b, c = mem['E'].shape[:2]
                mem['D'] = self.W2(torch.relu(self.W1(mem['E'][:, :, -self.hc:].flatten(2)))).reshape(b, c, self.hmax, P)
            return mem['D'][:, :, min(k, self.hmax - 1)]
        if k:                                                                               # the path continues through the rollout
            mem['V'] = self.householder(z_last, mem['V'])
        V, K = mem['V'], mem['K']
        b, c, M = V.shape[:3]
        if self.audit == 'uniform':
            a = V.new_full((b, c, self.nh, M), 1.0 / M)
        else:
            q = self.Wq(z_last.detach()).unflatten(-1, (self.nh, self.dk))                  # (b, C, nh, dk)
            logit = torch.einsum('bchd,bcmhd->bchm', q, K) / (sqrt(self.dk) * self.log_tau.exp())[None, None, :, None]
            a = torch.softmax(logit, -1)
        m = torch.einsum('bchm,bcmr->bchr', a, V).flatten(2)                                 # (b, C, nh*r)
        return self.W2(torch.relu(self.W1(m)))


def forward_states(bb, x):
    """bb_forward + the head input (token states) of the same pass: out (b, L, C), sd (b, 1, C), Z (b, C, N, D)."""
    cap = {}
    h = bb.head.register_forward_hook(lambda mod, inp, out: cap.__setitem__('z', inp[0]))
    try:
        out, sd = bb_forward(bb, x)
    finally:
        h.remove()
    b, _, c = x.shape
    return out, sd, cap['z'].reshape(b, c, x.shape[1] // P, -1)


def run(bb, mod, x, n_out):
    """Teacher-forced pass + rolling forecast. Returns out_tf (b, L, C) and the corrected rollout (b, n_out*P, C)."""
    b, _, c = x.shape
    out, sd, Z = forward_states(bb, x)
    sd0 = sd.detach()
    E = ((x[:, P:] - out[:, :-P]) / sd0).reshape(b, L // P - 1, P, c).permute(0, 3, 1, 2).detach()
    mem = mod.init_memory(E, Z)
    preds, xw, base, z_last = [], x, out[:, -P:], Z[:, :, -1]
    for k in range(n_out):
        if k:
            xw = torch.cat([xw[:, P:], preds[-1]], 1)
            o_k, _, Z_k = forward_states(bb, xw)
            base, z_last = o_k[:, -P:], Z_k[:, :, -1]
        d = mod.step(mem, k, z_last)
        preds.append(base if d is None else base + d.transpose(1, 2) * sd0)
    return out, torch.cat(preds, 1)


def roll_loss(pred, y):
    return sum((pred[:, :H] - y[:, :H]).square().mean() for H in HS) / len(HS)


@torch.no_grad()
def evaluate(bb, mod, xs, H, dev, bs=64):
    n = len(xs) - L - H + 1
    se, ae = 0.0, 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), H, dev)
        p = run(bb, mod, x, -(-H // P))[1][:, :H]
        se += (p - y).square().sum().item(); ae += (p - y).abs().sum().item()
    nel = n * H * xs.shape[1]
    return se / nel, ae / nel


@torch.no_grad()
def val_roll(bb, mod, xs, dev, bs=64):
    n = len(xs) - L - HMAX * P + 1
    tot = 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), HMAX * P, dev)
        tot += roll_loss(run(bb, mod, x, HMAX)[1], y).item() * x.shape[0]
    return tot / n


@torch.no_grad()
def val_last(bb, xs, dev, bs=64):
    """The official early-stopping loss (exp_forecast.vali with valid_last): MSE of the last token's next-patch forecast on every val
    window with a P-step future (= the H = 96 one-step forecast), averaged over batches weighted by batch size."""
    n = len(xs) - L - P + 1
    tot = 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), P, dev)
        tot += (bb_forward(bb, x)[0][:, -P:] - y).square().mean().item() * x.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E1', 'W']); ap.add_argument('--arm', default='errpath')
    ap.add_argument('--rank', type=int, default=8); ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--mod_lr_mult', type=float, default=10.0); ap.add_argument('--out', default=None)
    ap.add_argument('--no_roll', action='store_true', help='recipe bisect (ctrl only): L = L_tf only, the official train windows (96-step future)')
    ap.add_argument('--sel', default='roll', choices=['roll', 'last'], help="early stop on val L_roll, or on the official valid_last loss "
                    "(teacher-forced last-token MSE on the official val windows)")
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bb = build_backbone(a.data).to(dev)
    mod = Module(a.arm, CFG[a.data]['d_model'], rank=a.rank).to(dev)
    xs = {f: split_x(a.data, f) for f in ('train', 'val', 'test')}
    lr = LR[a.data]
    groups = [dict(params=list(bb.parameters()), lr=lr, base=lr)]
    if list(mod.parameters()):
        groups.append(dict(params=list(mod.parameters()), lr=lr * a.mod_lr_mult, base=lr * a.mod_lr_mult))
    opt = torch.optim.Adam(groups)
    assert not a.no_roll or a.arm == 'ctrl', '--no_roll is a ctrl-only recipe bisect'
    Hf = P if a.no_roll else HMAX * P
    ntr = len(xs['train']) - L - Hf + 1
    tag = a.arm + ('' if a.arm == 'ctrl' else f'_r{a.rank}') + a.tag
    npar = sum(p.numel() for p in mod.parameters())
    print(f'[joint] {a.data} arm {tag} | module params {npar} | backbone params {sum(p.numel() for p in bb.parameters())} | train windows {ntr} '
          f'| lr {lr} (module x{a.mod_lr_mult}) | loss = L_tf' + ('' if a.no_roll else ' + mean_H L_roll(H)') + f' | select {a.sel}', flush=True)
    path = os.path.join(ROOT, 'checkpoints', 'pat311_joint', f'{a.data}_{tag}_s{a.seed}.pth')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best, bad = float('inf'), 0
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        bb.train(); mod.train(); t0 = time.time(); ltf, lro = [], []
        for idx in np.array_split(rng.permutation(ntr), max(ntr // a.bs, 1)):
            x, y = windows(xs['train'], idx.tolist(), Hf, dev)
            if a.no_roll:
                out = forward_states(bb, x)[0]
                l_ro = torch.zeros((), device=dev)
            else:
                out, pred = run(bb, mod, x, HMAX)
                l_ro = roll_loss(pred, y)
            l_tf = (out - torch.cat([x[:, P:], y[:, :P]], 1)).square().mean()
            opt.zero_grad(); (l_tf + l_ro).backward(); opt.step()
            ltf.append(l_tf.item()); lro.append(l_ro.item())
        bb.eval(); mod.eval()
        v = val_roll(bb, mod, xs['val'], dev) if a.sel == 'roll' else val_last(bb, xs['val'], dev)
        print(f'[joint] epoch {ep + 1} {time.time() - t0:.0f}s | train L_tf {np.mean(ltf):.5f} L_roll {np.mean(lro):.5f} | val {a.sel} {v:.5f}', flush=True)
        if v < best:
            best, bad = v, 0
            torch.save(dict(bb=bb.state_dict(), mod=mod.state_dict()), path)
        else:
            bad += 1
            if bad >= a.patience:
                break
        for g in opt.param_groups:                                                          # Timer-XL lradj type1: lr * 0.1 ** epoch
            g['lr'] = g['base'] * (0.1 ** (ep + 1))
    ck = torch.load(path, map_location=dev)
    bb.load_state_dict(ck['bb']); mod.load_state_dict(ck['mod']); bb.eval(); mod.eval()
    res = dict(data=a.data, arm=a.arm, rank=a.rank, seed=a.seed, params=npar, best_val=best, test={})
    audits = {'t0r': ('zero',), 'o': ('zero', 'uniform'), 'errpath': ('zero', 'noH', 'uniform')}.get(a.arm, ())
    for H in HS:
        mse, mae = evaluate(bb, mod, xs['test'], H, dev)
        r = dict(mse=mse, mae=mae)
        for au in audits:
            mod.audit = au; r[au] = evaluate(bb, mod, xs['test'], H, dev)[0]; mod.audit = None
        res['test'][H] = r
        print(f'[joint] test H{H} | mse {mse:.4f} mae {mae:.4f}' + ''.join(f' | {au} {r[au]:.4f}' for au in audits), flush=True)
    out = a.out or os.path.join(ROOT, 'pat_logs', 'joint', f'{a.data}_{tag}_s{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
