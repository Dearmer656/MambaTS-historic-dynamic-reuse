"""PAT-307 R1TeflICL: TEFL (T0-MS, patchmem/r1_msres.py R1TeflMS) with its output map W2 replaced (or corrected) by the implicit
one-step-GD map of an attention over in-window demonstrations (TEFL "compiled into" ICL).

T0-MS: eps = the K = Hmax lead errors of the single lookback origin j0 = T-1-K (the forecast issued one horizon ago, compared with the
observed last K patches), delta = W2 ReLU(W1 eps). Every lookback origin j has the same object E[:, j] (leads 1..K of the backbone's
free-running rollout from j, current window's RevIN, multistep_residuals). Demonstration pairs inside the lookback, stride 1 patch:
  key k_j = E[:, j] (error of the forecast issued at j), value v_j = E[:, j+K] (error of the forecast issued one horizon later),
  j = 0..T-1-2K (both fully observed: the value's last lead is patch j+2K <= T-1). The query is eps = E[:, j0] and its "value" is the
  error of the forecast being made now (origin T-1), i.e. exactly what delta should predict. n = T-2K demos (L1440: 16, L720: 1,
  L384: 0 -> no ICL term). No data from before the window is used.
One GD step from W = 0 on the demos in feature space phi: W_impl = sum_j v_j phi(k_j)^T, delta_icl = W_impl phi(q) (= linear attention).
Step size normalised so the term is bounded by max |v_j| and scale free: delta_icl = eta_k sum_j v_j <phi(k_j), phi(q)> /
(|phi(q)| sum_j |phi(k_j)|) (one demo with k = q gives eta * v).
  variant 'id'  : phi = identity (no learned map; the offline zero-parameter arm, eta fixed 1). k_follow=True: K = the forecast's
                  H in patches (min(n_out, Hmax)) instead of Hmax (TEFL's error length = H; n = T-2K demos, e.g. L720 H96: 11)
  variant 'tied': phi = ReLU(W1 .), W2 frozen at 0 (TEFL with W2 replaced by W_impl; same W1 as TEFL)
  variant 'qk'  : q = Wq eps, k = Wk k_j (independent low-rank linear maps, rank = TEFL rank), W1 / W2 unused
  variant 'hyb' : delta = W2 phi(q) + eta_k sum_j (v_j - W2 phi(k_j)) <phi(k_j), phi(q)> / (...): one GD step on the demos starting
                  from TEFL's trained W2 (TEFL + test-time correction), phi = ReLU(W1 .)
Variables share the maps (as TEFL). eta (Hmax,) is learned (init eta0; eta0 = 0 -> the model is R1 / T0-MS at init) or fixed.
Errors are detached (stop-gradient, as T0-MS). If the query origin is invalid (padded sample), every demo origin (< query origin) is
too, so the ICL term is 0. Inference audits (on top of R1Tefl's zero / perm; 'perm' swaps only the query for a distant window's,
the demos stay the window's own): 'icloff' (ICL term removed; 'hyb' -> its TEFL part), 'iclperm' (values permuted
across the window's demos: pairing broken, the demo set and its mean kept)."""
import torch
import torch.nn as nn
from patchmem.r1_msres import R1TeflMS, multistep_residuals


def demos(E, avail, K):
    """E (b, T-1, K, C, P) lead errors of the lookback origins, avail (b, T-1, K). Returns the query eps = E[:, T-1-K] as (b, C, K*P)
    and the demos dict(k, v (b, n, C, K*P), mask (b, n)) with k = E[:, j], v = E[:, j+K], j = 0..T-1-2K (None when n = T-2K < 1)."""
    b, M, _, c, P = E.shape
    T = M + 1
    flat = lambda e: e.permute(0, 1, 3, 2, 4).reshape(b, e.shape[1], c, K * P)
    eps = flat(E[:, T - 1 - K:T - K])[:, 0].detach()
    n = T - 2 * K
    if n < 1:
        return eps, None
    j = torch.arange(n, device=E.device)
    return eps, dict(k=flat(E[:, j]).detach(), v=flat(E[:, j + K]).detach(), mask=avail[:, j, K - 1] & avail[:, j + K, K - 1])


class R1TeflICL(R1TeflMS):
    def __init__(self, cfg, variant='tied', eta0=0.0, eta_learn=True, k_follow=False, **kw):
        super().__init__(cfg, **kw)
        assert variant in ('id', 'tied', 'qk', 'hyb'), variant
        assert not k_follow or variant == 'id', 'K = H only for the map-free id arm (learned maps have a fixed input size)'
        self.variant, self.k_follow, self.n_out = variant, k_follow, None
        r = self.W1.out_features
        if variant == 'qk':
            self.Wq, self.Wk = nn.Linear(self.din, r, bias=False), nn.Linear(self.din, r, bias=False)
        if variant in ('id', 'tied', 'qk'):                                              # no trained W2 term
            self.W2.weight.requires_grad_(False)
            nn.init.zeros_(self.W2.weight)
        if variant in ('id', 'qk'):
            self.W1.weight.requires_grad_(False)
        self.eta = nn.Parameter(torch.full((self.hmax,), float(eta0)), requires_grad=eta_learn)
        self.demo, self.icl_gen = None, torch.Generator().manual_seed(0)

    def set_audit(self, mode=None, seed=0):
        if mode in ('icloff', 'iclperm'):
            self.audit = mode
            self.icl_gen = torch.Generator().manual_seed(seed)
            return
        super().set_audit(mode, seed)

    def forward(self, x, n_out, *a, **kw):
        self.n_out = n_out
        return super().forward(x, n_out, *a, **kw)

    def error(self, x, xn, xo, yh, perm, means, stdev, lens=None):
        b, L, c = x.shape
        P = self.P
        K = min(self.n_out, self.hmax) if self.k_follow else self.hmax                 # k_follow: error length = the forecast's H
        T = L // P
        assert T - 1 - K >= 0, f'T0-MS needs T >= K + 1 ({T}, {K})'
        _, _, _, t0, tm = self.revin(x, lens)
        tok = self.embed(self.to_scan(xo, perm)).reshape(b, T * c, -1)
        E, avail = multistep_residuals(self, tok, xo, perm, tm, t0, K)                 # (b, T-1, K, C, P), (b, T-1, K)
        eps, self.demo = demos(E, avail, K)
        if K < self.hmax:                                                                # zero-pad to din (W1 / W2 unused by 'id')
            eps = torch.cat([eps, eps.new_zeros(b, c, (self.hmax - K) * P)], -1)
        return eps

    def phi(self, e, role):
        if self.variant == 'id':
            return e
        if self.variant == 'qk':
            return (self.Wq if role == 'q' else self.Wk)(e)
        return torch.relu(self.W1(e))

    def extra_delta(self, delta, eps, stdev):
        D = self.demo
        if self.variant != 'hyb':                                                        # W2 term off (also after loading a T0-MS W2)
            delta = torch.zeros_like(delta)
        if D is None or self.audit == 'icloff':
            return delta
        b, c, hmax, P = delta.shape
        KP = D['k'].shape[-1]                                                            # K*P (< din when k_follow)
        q = self.phi(eps[..., :KP], 'q')                                                 # (b, C, r)
        k = self.phi(D['k'], 'k')                                                        # (b, n, C, r)
        v = D['v']                                                                       # (b, n, C, K*P)
        if self.variant == 'hyb':
            v = v - self.W2(k)                                                           # residual of the TEFL map on the demos
        if self.audit == 'iclperm':                                                      # eval windows are unpadded: every demo valid
            assert bool(D['mask'].all()), 'iclperm assumes no masked demos'
            v = v[:, torch.randperm(v.shape[1], generator=self.icl_gen).to(v.device)]
        m = D['mask'][:, :, None].to(q.dtype)                                            # (b, n, 1)
        s = (k * q[:, None]).sum(-1) * m                                                 # (b, n, C)
        den = q.norm(dim=-1) * (k.norm(dim=-1) * m).sum(1) + 1e-6                       # (b, C)
        icl = (s[..., None] * v).sum(1) / den[..., None]                                 # (b, C, K*P)
        if KP < hmax * P:
            icl = torch.cat([icl, icl.new_zeros(b, c, hmax * P - KP)], -1)
        return delta + icl.reshape(b, c, hmax, P) * self.eta[None, None, :, None]
