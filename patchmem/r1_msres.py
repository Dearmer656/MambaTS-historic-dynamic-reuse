"""PAT-307 multi-step (lead-k) residuals of the backbone's own free-running rollouts inside the lookback.
O and T0 store ONE-step errors (teacher-forced: state after patch j-1 -> patch j). The rollout's k-th forecast is a k-step-ahead
forecast, whose error is a different object (compounding, exposure bias). Here every lookback origin j (state after patch j,
j = 0..T-2) is rolled out K patches on its own predictions (plain backbone, uncorrected), giving
  E[:, j, k] = x_{j+1+k} - xhat_{j+1+k | j}        lead k+1, available iff j + 1 + k <= T - 1 (inside the lookback)
(RevIN-normalised with the current window's statistics, original variable order, detached). E[:, j, 0] is O's slot value V[:, j].

R1ObsMS (O-MS): O with lead-matched values. Rollout patch k (lead k+1) retrieves over the origins that have that lead:
  delta_k = g(h) * sum_h W_h * sum_j softmax_j(q(h_k) . k(h_j)) E[:, j, min(k, K-1)]
  key = the ORIGIN state h_j (the situation before the outcome; O keys the state after the error patch). Beyond K the last lead is
  reused. Slot keep p (origins), validity (lens), audits as R1Obs ('perm' permutes the origins of the values).
R1TeflMS (T0-MS): TEFL-style g on the multi-step errors of ONE origin j0 = T-1-K (TEFL: the forecast issued one horizon ago),
  eps = [E[:, j0, 0], ..., E[:, j0, K-1]] (the last K lookback patches) -> ReLU(eps W1) W2 -> K lead blocks, K = Hmax."""
import torch
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from patchmem.ref_scan import mamba_proj
from patchmem.r1_obsmem import R1Obs
from patchmem.r1_tefl import R1Tefl


@torch.no_grad()
def boundary_layer_states(m, tok, tm=None):
    """Teacher-forced pass (as R1.history) that keeps EVERY layer's SSM state after every time patch: list of (b, T, D, N)."""
    layers = m.enc.layers
    hidden, residual, caps = tok, None, []
    for i, layer in enumerate(layers):
        hn, residual = m._norm(layer, hidden, residual)
        mx = layer.mixer
        x, dt, A, B, C, D, z, bias = mamba_proj(mx, hn)
        delta = F.softplus(dt + bias[None, :, None])
        if tm is not None:
            delta = delta * tm[:, None, :]
        dA = torch.exp(delta.unsqueeze(-1) * A[None, :, None, :])
        dBu = (delta * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        h = x.new_zeros(x.shape[0], x.shape[1], A.shape[1])
        cap = []
        for s in range(x.shape[-1]):
            h = dA[:, :, s] * h + dBu[:, :, s]
            if (s + 1) % m.c == 0:
                cap.append(h)
        caps.append(torch.stack(cap, 1))
        if i < len(layers) - 1:
            y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=None, delta_softplus=False)
            hidden = mx.out_proj(y.transpose(1, 2))
    return caps


@torch.no_grad()
def multistep_residuals(m, tok, xo, perm, tm, t0, K):
    """E (b, T-1, K, C, P) lead-k residuals of origins j = 0..T-2, avail (b, T-1, K) bool (inside the lookback and valid origin)."""
    b, T, c, P = xo.shape
    M = T - 1
    caps = boundary_layer_states(m, tok, tm)
    states = [s[:, :M].reshape(b * M, *s.shape[2:]).clone() for s in caps]
    pm = perm.repeat_interleave(M, 0)
    preds = []
    base = m.to_orig(m.read(states[-1]).unsqueeze(1), pm).squeeze(1)                    # (b*M, C, P) = yh[:, j]
    for k in range(K):
        if k:
            states = m.step_time(m.embed(m.to_scan(preds[-1].unsqueeze(1), pm)).squeeze(1), states)
            base = m.to_orig(m.read(states[-1]).unsqueeze(1), pm).squeeze(1)
        preds.append(base)
    pr = torch.stack(preds, 1).reshape(b, M, K, c, P)
    j = torch.arange(M, device=xo.device)[:, None]
    tgt = j + 1 + torch.arange(K, device=xo.device)[None, :]                             # (M, K) target patch index
    ok = tgt <= T - 1
    E = xo[:, tgt.clamp(max=T - 1)] - pr                                                 # (b, M, K, C, P)
    avail = ok[None].expand(b, M, K)
    if t0 is not None:                                                                   # origin j valid iff patch j is valid
        avail = avail & (torch.arange(M, device=xo.device)[None, :] >= t0[:, None])[:, :, None]
    E = E * avail[..., None, None].to(E.dtype)
    return E.detach(), avail


class R1ObsMS(R1Obs):
    def __init__(self, cfg, readout_ln=False, mem_group=4, qk_dim=8, slot_keep=1.0, feed=True, kmax=7):
        super().__init__(cfg, readout_ln=readout_ln, mem_group=mem_group, qk_dim=qk_dim, slot_keep=slot_keep, feed=feed)
        self.kmax = kmax

    def forward(self, x, n_out, perm=None, trace=None, lens=None):
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        self.mem_stats = {}
        means, stdev, xn, t0, tm = self.revin(x, lens)
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xo = xp.permute(0, 2, 1, 3)
        xs = self.to_scan(xo, perm)
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)
        K = self.kmax
        if T >= 2:
            E, avail = multistep_residuals(self, tok, xo, perm, tm, t0, K)              # (b, M, K, C, P), (b, M, K)
            if self.audit == 'perm' and T - 1 >= 2:
                E = E[:, torch.randperm(T - 1, generator=self.audit_gen).to(E.device)]
            keys = self.obs_net.key(bstates[:, :T - 1].flatten(0, 1)).unflatten(0, (b, T - 1))   # origin states h_j
            keep = (torch.rand(b, T - 1, device=x.device) < self.slot_keep) if self.training and self.slot_keep < 1 else None
        base = yh[:, -1]
        outs = []
        for k in range(n_out):
            if k:
                fb = outs[-1] if self.feed else base
                states = self.step_time(self.embed(self.to_scan(fb.unsqueeze(1), perm)).squeeze(1), states)
                base = self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1)
            if T < 2:
                outs.append(base)
                continue
            kk = min(k, K - 1)
            mask = avail[:, :, kk] if keep is None else avail[:, :, kk] & keep
            h = bstates[:, -1] if k == 0 else states[-1]
            outs.append(base + self._correct(h, keys, E[:, :, kk], mask))
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        return dict(pred=pred, hist=hist, perm=perm)


class R1TeflMS(R1Tefl):
    """T0-MS: eps = the K lead errors of the single origin j0 = T-1-K (K = Hmax); din = K * P."""
    def __init__(self, cfg, readout_ln=False, rank=64, hmax=7, feed=True, tmax=30, extrap='last'):
        super().__init__(cfg, readout_ln=readout_ln, mode='native', hc=hmax, rank=rank, hmax=hmax, feed=feed, tmax=tmax, extrap=extrap)
        self.mode = 'ms'

    def error(self, x, xn, xo, yh, perm, means, stdev, lens=None):
        b, L, c = x.shape
        P, K = self.P, self.hmax
        T = L // P
        assert T - 1 - K >= 0, f'T0-MS needs T >= K + 1 ({T}, {K})'
        _, _, _, t0, tm = self.revin(x, lens)
        tok = self.embed(self.to_scan(xo, perm)).reshape(b, T * c, -1)
        E, avail = multistep_residuals(self, tok, xo, perm, tm, t0, K)
        j0 = T - 1 - K
        e = E[:, j0]                                                                     # (b, K, C, P) leads 1..K of origin j0
        return e.permute(0, 2, 1, 3).reshape(b, c, K * P).detach()
