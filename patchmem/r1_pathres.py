"""PAT-307 R1 + path-transported residual correction (R1PathRes). No attention: the lookback one-step errors (the R1Obs values) are
embedded and carried by a state-driven Householder path with a forget gate; the carried vector is read out as the correction.

  slot t (t = 1..T-1): u_t = W_e vec(e_t) (r dims), e_t = sg(x_t - yh_{t-1}) as in R1Obs (RevIN-normalised, original variable order)
  transition from state h:  T(h) c = lam(h) * (I - beta(h) k(h) k(h)^T) c      k = normalize(W_k sg h), beta = 2 sigmoid(.), lam = sigmoid(.)
  history:  c_t = T(h_t) c_{t-1} + u_t                    => c_{T-1} = sum_j (prod_{i>j} lam_i H_i) u_j  (errors transported along the path)
  rollout patch k: c <- T(h_k) c for k >= 1 (no new errors), delta_k = W_o c (C*P), W_o = 0 at init (R1PathRes == R1 exactly);
                   forecast = read(h_k) + delta_k, fed back (feed=True).
noH=True: H = I (lam-gated sum of errors only, i.e. learned exponential smoothing of the errors) -- the path ablation.
Audits (R1Obs interface): 'zero' (delta = 0) | 'uniform' (H = I at inference: path removed, same weights) | 'perm' (slot values permuted).
Slot keep p < 1: training-time per-sample dropout of slots (u_t = 0), as R1Obs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from patchmem.r1_obsmem import R1Obs
from patchmem.r1_model import R1


class PathNet(nn.Module):
    def __init__(self, d_inner, d_state, c, P, rank=32, noH=False, lam_bias=2.0):
        super().__init__()
        din = d_inner * d_state
        self.r, self.noH = rank, noH
        self.We = nn.Linear(c * P, rank, bias=False)
        self.Wk = nn.Linear(din, rank, bias=False)
        self.Wb = nn.Linear(din, 1)
        self.Wl = nn.Linear(din, 1)
        nn.init.constant_(self.Wl.bias, lam_bias)                                      # lam ~ 0.88 at init
        self.Wo = nn.Linear(rank, c * P, bias=False)
        nn.init.zeros_(self.Wo.weight)
        self.audit = None

    def coef(self, h):
        """h (..., D, N) -> k (..., r) unit, beta (..., 1), lam (..., 1)."""
        z = h.detach().flatten(-2)
        return F.normalize(self.Wk(z), dim=-1), 2 * torch.sigmoid(self.Wb(z)), torch.sigmoid(self.Wl(z))

    def step(self, c, k, beta, lam):
        if not (self.noH or self.audit == 'uniform'):
            c = c - beta * k * (k * c).sum(-1, keepdim=True)
        return lam * c


class R1PathRes(R1Obs):
    def __init__(self, cfg, readout_ln=False, rank=32, noH=False, slot_keep=1.0, feed=True):
        R1.__init__(self, cfg, readout_ln=readout_ln)
        self.obs_net = PathNet(self.D, self.N, self.c, self.P, rank=rank, noH=noH)
        self.slot_keep, self.feed = slot_keep, feed
        self.audit, self.audit_gen = None, None
        self.mem_stats = {}

    def forward(self, x, n_out, perm=None, trace=None, lens=None):
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        self.mem_stats = {}
        net = self.obs_net
        means, stdev, xn, t0, tm = self.revin(x, lens)
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xo = xp.permute(0, 2, 1, 3)                                                     # (b, T, C, P) original order
        xs = self.to_scan(xo, perm)
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)
        cv = xo.new_zeros(b, net.r)
        if T >= 2:
            V = (xo[:, 1:] - yh[:, :-1]).detach()                                       # (b, T-1, C, P) e_t, slots t = 1..T-1
            if self.audit == 'perm' and T - 1 >= 2:
                V = V[:, torch.randperm(T - 1, generator=self.audit_gen).to(V.device)]
            U = net.We(V.flatten(2))                                                    # (b, T-1, r)
            keep = (torch.rand(b, T - 1, device=x.device) < self.slot_keep) if self.training and self.slot_keep < 1 else None
            if t0 is not None:
                ok = torch.arange(1, T, device=x.device)[None, :] >= (t0 + 1)[:, None]
                keep = ok if keep is None else keep & ok
            if keep is not None:
                U = U * keep.unsqueeze(-1).to(U.dtype)
            K, B, Lm = net.coef(bstates[:, 1:])                                         # transitions by the slot states h_t
            for t in range(T - 1):
                cv = net.step(cv, K[:, t], B[:, t], Lm[:, t]) + U[:, t]
            st = self.mem_stats
            st.setdefault('lam', []).append(Lm.mean().detach()); st.setdefault('beta', []).append(B.mean().detach())
        base = yh[:, -1]
        outs = []
        for k in range(n_out):
            if k:
                fb = outs[-1] if self.feed else base
                states = self.step_time(self.embed(self.to_scan(fb.unsqueeze(1), perm)).squeeze(1), states)
                base = self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1)
                cv = net.step(cv, *net.coef(states[-1]))
            delta = net.Wo(cv).reshape(b, c, P)
            if self.audit == 'zero' or T < 2:
                delta = torch.zeros_like(delta)
            self.mem_stats.setdefault('delta_rms', []).append(delta.square().mean().sqrt().detach())
            outs.append(base + delta)
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        return dict(pred=pred, hist=hist, perm=perm)
