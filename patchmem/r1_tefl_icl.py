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
  variant 'id'  : phi = identity (no learned map; the offline zero-parameter arm, eta fixed 1)
  variant 'ridge': the same demos, closed-form ridge instead of one GD step: delta = q K^T (K K^T + lam I)^-1 V per variable
                  (lam = RIDGE_LAM = 1, EFC-CET's value; no learned map, eta fixed 1): the solver ablation of 'id' (same information)
k_follow=True (any variant): K = the forecast's H in patches (min(n_out, Hmax)) instead of Hmax (TEFL's error length = H; n = T-2K
demos, e.g. L720 H96: 11). eps and the demo errors are zero-padded on leads K+1..Hmax so W1 keeps its input size; training samples H,
so one model covers every H. 'hyb' with eta fixed at 0 is plain TEFL with this input (TEFL-KH).
  variant 'tied': phi = ReLU(W1 .), W2 frozen at 0 (TEFL with W2 replaced by W_impl; same W1 as TEFL)
  variant 'qk'  : q = Wq eps, k = Wk k_j (independent low-rank linear maps, rank = TEFL rank), W1 / W2 unused
  variant 'hyb' : delta = W2 phi(q) + eta_k sum_j (v_j - W2 phi(k_j)) <phi(k_j), phi(q)> / (...): one GD step on the demos starting
                  from TEFL's trained W2 (TEFL + test-time correction), phi = ReLU(W1 .)
key='ced' ('id' / 'ridge' only; the 2x2 vs EFC-CET: row vs anti-diagonal feature x window-local vs stream): the key of a demo
  origin o is EFC's causal error diagonal inside the lookback, D_o[l] = E[:, o-l, lead l] (l = 1..K: the K earlier origins' errors on
  the same patch o), value v_o = E[:, o] (leads 1..K, targets o+1..o+K), o = K..T-1-K (n = T-2K, as 'row'); query D_{T-1} (errors of
  origins T-1-l on the last observed patch T-1). The default key='row' is k_j = E[:, j] above.
Shrinkage (2026-10-25, fixed before results):
  ridge_rel = s: ridge lam = s * mean_j |k_j|^2 over the valid demos (per sample / variable) instead of the absolute RIDGE_LAM;
                 s in {1, 10, 100, 1000} traces closed-form ridge (small s) -> scaled one-step GD (large s); unset = old lam 1.
  variant 'hybridge': ridge anchored at TEFL's W2 in phi = ReLU(W1 .) space: W = argmin sum |v_j - W phi_j|^2 + lam |W - W2|^2, i.e.
                 delta = W2 phi(q) + phi(q) Phi^T (Phi Phi^T + lam I)^-1 (V - Phi W2^T) (lam -> inf: TEFL), lam as ridge_rel.
  nshrink = c (GD variants): eta_eff = eta * n / (n + c), n = valid demos of the sample (1 demo: 1/(1+c); c = 8).
Variables share the maps (as TEFL). eta (Hmax,) is learned (init eta0; eta0 = 0 -> the model is R1 / T0-MS at init) or fixed.
Errors are detached (stop-gradient, as T0-MS). If the query origin is invalid (padded sample), every demo origin (< query origin) is
too, so the ICL term is 0. Inference audits (on top of R1Tefl's zero / perm; 'perm' swaps only the query for a distant window's,
the demos stay the window's own): 'icloff' (ICL term removed; 'hyb' -> its TEFL part), 'iclperm' (values permuted
across the window's demos: pairing broken, the demo set and its mean kept)."""
import torch
import torch.nn as nn
RIDGE_LAM = 1.0                                                                          # EFC-CET's lambda (normalised errors)
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


def demos_ced(E, avail, K):
    """As demos() with EFC's anti-diagonal keys: query D_{T-1}, demos k = D_o, v = E[:, o], o = K..T-1-K (None when n < 1)."""
    b, M, _, c, P = E.shape
    T = M + 1
    l = torch.arange(1, K + 1, device=E.device)
    flat = lambda e: e.permute(0, 1, 3, 2, 4).reshape(b, e.shape[1], c, K * P)        # (b, n, K, C, P) -> (b, n, C, K*P)
    diag = lambda o: E[:, o[:, None] - l[None], l[None] - 1]                            # (b, n, K, C, P): E[:, o-l, l-1]
    eps = flat(diag(torch.tensor([T - 1], device=E.device)))[:, 0].detach()
    n = T - 2 * K
    if n < 1:
        return eps, None
    o = torch.arange(K, T - K, device=E.device)
    mask = avail[:, o[:, None] - l[None], l[None] - 1].all(-1) & avail[:, o, K - 1]
    return eps, dict(k=flat(diag(o)).detach(), v=flat(E[:, o]).detach(), mask=mask)


class R1TeflICL(R1TeflMS):
    def __init__(self, cfg, variant='tied', eta0=0.0, eta_learn=True, k_follow=False, key='row', ridge_rel=None, nshrink=0.0, **kw):
        super().__init__(cfg, **kw)
        assert variant in ('id', 'tied', 'qk', 'hyb', 'ridge', 'hybridge'), variant
        assert variant != 'hybridge' or ridge_rel is not None, 'hybridge needs ridge_rel'
        self.ridge_rel, self.nshrink = ridge_rel, nshrink
        assert key == 'row' or (key == 'ced' and variant in ('id', 'ridge')), (key, variant)   # ced changes eps: no TEFL map on it
        self.variant, self.k_follow, self.key, self.n_out = variant, k_follow, key, None
        r = self.W1.out_features
        if variant == 'qk':
            self.Wq, self.Wk = nn.Linear(self.din, r, bias=False), nn.Linear(self.din, r, bias=False)
        if variant in ('id', 'tied', 'qk', 'ridge'):                                     # no trained W2 term
            self.W2.weight.requires_grad_(False)
            nn.init.zeros_(self.W2.weight)
        if variant in ('id', 'qk', 'ridge'):
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
        eps, self.demo = (demos_ced if self.key == 'ced' else demos)(E, avail, K)
        if K < self.hmax:                                                                # zero-pad leads K+1..Hmax: W1 input size fixed
            pad = lambda e: torch.cat([e, e.new_zeros(*e.shape[:-1], (self.hmax - K) * P)], -1)
            eps = pad(eps)
            if self.demo is not None:
                self.demo.update(k=pad(self.demo['k']), v=pad(self.demo['v']))
        return eps

    def phi(self, e, role):
        if self.variant in ('id', 'ridge'):
            return e
        if self.variant == 'qk':
            return (self.Wq if role == 'q' else self.Wk)(e)
        return torch.relu(self.W1(e))

    def extra_delta(self, delta, eps, stdev):
        D = self.demo
        if self.variant not in ('hyb', 'hybridge'):                                      # W2 term off (also after loading a T0-MS W2)
            delta = torch.zeros_like(delta)
        if D is None or self.audit == 'icloff':
            return delta
        b, c, hmax, P = delta.shape
        q = self.phi(eps, 'q')                                                           # (b, C, r) (zero padding: same dot / norm for 'id')
        k = self.phi(D['k'], 'k')                                                        # (b, n, C, r)
        v = D['v']                                                                       # (b, n, C, K*P)
        if self.variant in ('hyb', 'hybridge'):
            v = v - self.W2(k)                                                           # residual of the TEFL map on the demos
        if self.audit == 'iclperm':                                                      # eval windows are unpadded: every demo valid
            assert bool(D['mask'].all()), 'iclperm assumes no masked demos'
            v = v[:, torch.randperm(v.shape[1], generator=self.icl_gen).to(v.device)]
        m = D['mask'][:, :, None].to(q.dtype)                                            # (b, n, 1)
        if self.variant in ('ridge', 'hybridge'):                                        # closed-form ridge on the same demos, per variable
            kk = (k * m[..., None]).permute(0, 2, 1, 3)                                  # (b, C, n, d), masked demos -> 0 rows
            vv = (v * m[..., None]).permute(0, 2, 1, 3)
            if self.ridge_rel is None:
                lam = torch.full(kk.shape[:2], RIDGE_LAM, device=kk.device, dtype=kk.dtype)
            else:                                                                        # s * mean |k_j|^2 over the valid demos
                nv = m[..., 0].sum(1)[:, None].clamp_min(1)                              # (b, 1)
                lam = (self.ridge_rel * kk.square().sum((-1, -2)) / nv).clamp_min(1e-6)  # (b, C)
            G = kk @ kk.transpose(-1, -2) + lam[..., None, None] * torch.eye(kk.shape[2], device=kk.device, dtype=kk.dtype)
            icl = ((q[:, :, None] @ kk.transpose(-1, -2)) @ torch.linalg.solve(G, vv)).squeeze(2)   # q K^T (K K^T + lam I)^-1 V
            return delta + icl.reshape(b, c, hmax, P) * self.eta[None, None, :, None]
        s = (k * q[:, None]).sum(-1) * m                                                 # (b, n, C)
        den = q.norm(dim=-1) * (k.norm(dim=-1) * m).sum(1) + 1e-6                       # (b, C)
        icl = (s[..., None] * v).sum(1) / den[..., None]                                 # (b, C, K*P)
        if self.nshrink:                                                                 # eta * n / (n + c), n = valid demos
            nv = m[..., 0].sum(1)[:, None, None]                                         # (b, 1, 1)
            icl = icl * nv / (nv + self.nshrink)
        return delta + icl.reshape(b, c, hmax, P) * self.eta[None, None, :, None]
