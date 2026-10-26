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
Training-time controls of 'hyb' (2026-10-26, fixed before results; the inference audits showed iclperm ~ +1% on Traffic / ~0 on ETTh2):
  variant 'hybmean': as 'hyb' with the query similarity removed: weights |phi(k_j)| / sum_j |phi(k_j)| instead of
                 <phi(k_j), phi(q)> / (|phi(q)| sum_j |phi(k_j)|), i.e. cos(q, k_j) -> 1 (a recent-residual mean; same scale, no selection).
  variant 'hybperm': as 'hyb' with the demo residuals v_j - W2 phi(k_j) permuted among the sample's valid demos at every forward
                 (training and evaluation; pairing broken, residual set kept). Its iclperm audit is not informative (already unpaired).
PAT-313 case analysis (2026-10-26, fixed before results; inference audits on a trained hyb, R1_ICL_XAUDIT in exp_r1.py): the normal
  pass banks every window's demos (fp32, CPU) and logs Delta_t (the ICL term); then iclconst (Delta_t -> its pass mean), iclswap<d> (demos of
  window (i - d) mod N, query and TEFL part kept), iclproj_{const,lin,daily,common} (Delta_t projected; daily period R1_ICL_PERIOD).
  Windows are indexed by win_ptr (eval order, reset by set_audit), so the audited split must be read in order (stride 1).
  Point 6 (select then adapt): iclsel / iclsel1 / iclselfix_{all,recent4,phase,topk8} (sel_correct; phase lags R1_ICL_SELPHASE);
  controls iclselavg / iclselavg1 (uniform 1/4 mix) and iclselfix1_<set> (one GD step on one set); per-window weights -> sel_wdump;
  iclselw_<tag>: one-step set fits mixed with given per-window weights from <R1_ICL_SELW>_<tag>.pt (S, N, C) (causal Hedge, offline).
Weight dump (2026-10-26, diagnostic): with R1_ICL_WDUMP=<prefix>, Exp_R1.audit records the GD-step demo weights
  <phi(k_j), phi(q)> / (|phi(q)| sum_j |phi(k_j)|) (mean over variables) of the normal pass per split -> <prefix>_<split>_L<L>H<H>.pt
  (demo j = in-window origin patch j; query origin T-1-K; lag in patches = T-1-K-j).
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
import os
import torch
import torch.nn as nn
RIDGE_LAM = 1.0                                                                          # EFC-CET's lambda (normalised errors)
from patchmem.r1_msres import R1TeflMS, multistep_residuals


HYB = ('hyb', 'hybridge', 'hybmean', 'hybperm')                                       # variants with TEFL's trained W2 term


def permute_valid(v, mask, gen):
    """v (b, n, ...), mask (b, n): per sample, a random permutation of the rows among the valid demos (invalid rows stay in place)."""
    b, n = mask.shape
    r = torch.rand(b, n, generator=gen).to(v.device)
    ar = (torch.arange(n, device=v.device, dtype=r.dtype)[None] / n).expand(b, n)
    src = torch.argsort(torch.where(mask, r, 10 + ar), dim=1)                            # valid rows in random order, then invalid in place
    dst = torch.argsort(torch.where(mask, ar, 10 + ar), dim=1)                           # valid rows in place order, then invalid in place
    out = v.clone()
    bi = torch.arange(b, device=v.device)[:, None]
    out[bi, dst] = v[bi, src]
    return out


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
        assert variant in ('id', 'tied', 'qk', 'hyb', 'ridge', 'hybridge', 'hybmean', 'hybperm'), variant
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
        self.perm_gen = torch.Generator().manual_seed(1)                                 # 'hybperm' training-time permutation
        self.wlog = None                                                                  # list, set by Exp_R1.audit under R1_ICL_WDUMP
        self.win_ptr, self.case_collect, self.case_period = 0, False, 0                    # PAT-313 case analysis (Exp_R1.audit, R1_ICL_XAUDIT)
        self._bank, self._icl_log, self.bank, self.icl_mean = [], [], None, None
        self.sel_phase, self.sel_log, self.K_last = [], [], self.hmax                   # PAT-313 point 6 (iclsel*)
        self.sel_wdump = None
        self.selw_cache = {}                                                             # iclselw_<tag>: R1_ICL_SELW_<tag>.pt

    def set_audit(self, mode=None, seed=0):
        self.win_ptr = 0                                                                  # PAT-313: window index within an eval pass
        if mode is not None and mode.startswith(('iclconst', 'iclswap', 'iclproj_', 'iclsel')):
            self.audit = mode
            return
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
        self.K_last = K
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
        if self.variant not in HYB:                                                      # W2 term off (also after loading a T0-MS W2)
            delta = torch.zeros_like(delta)
        if D is None or self.audit == 'icloff':
            assert D is not None or not (self.case_collect or (self.audit or '').startswith(('iclconst', 'iclswap', 'iclproj_'))), \
                'PAT-313 case audits need demos in every window (win_ptr alignment)'
            return delta
        b, c, hmax, P = delta.shape
        if self.case_collect and self.audit is None:                                     # normal pass: bank this window's demos (CPU fp16)
            self._bank.append((D['k'].float().cpu(), D['v'].float().cpu(), D['mask'].cpu()))
        if self.audit is not None and self.audit.startswith('iclswap'):                  # demos of window (i - d) mod N, query / TEFL kept
            dd = int(self.audit[len('iclswap'):])
            N = self.bank[0].shape[0]
            idx = (torch.arange(self.win_ptr, self.win_ptr + b) - dd) % N
            D = dict(k=self.bank[0][idx].to(D['k']), v=self.bank[1][idx].to(D['v']), mask=self.bank[2][idx].to(D['mask'].device))
        q = self.phi(eps, 'q')                                                           # (b, C, r) (zero padding: same dot / norm for 'id')
        k = self.phi(D['k'], 'k')                                                        # (b, n, C, r)
        v = D['v']                                                                       # (b, n, C, K*P)
        if self.variant in HYB:
            v = v - self.W2(k)                                                           # residual of the TEFL map on the demos
        if self.variant == 'hybperm':
            v = permute_valid(v, D['mask'], self.perm_gen)
        if self.audit == 'iclperm':                                                      # eval windows are unpadded: every demo valid
            assert bool(D['mask'].all()), 'iclperm assumes no masked demos'
            v = v[:, torch.randperm(v.shape[1], generator=self.icl_gen).to(v.device)]
        if self.audit is not None and self.audit.startswith('iclsel'):                   # PAT-313 point 6: select then adapt
            add = self.sel_correct(q, k, v, D['mask'], b, c, hmax, P)
            self.win_ptr += b
            return delta + add.reshape(b, c, hmax, P)
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
        if self.variant == 'hybmean':                                                    # cos(q, k_j) -> 1: no query dependence at all
            s = k.norm(dim=-1) * m
            den = (k.norm(dim=-1) * m).sum(1) + 1e-6
        else:
            s = (k * q[:, None]).sum(-1) * m                                             # (b, n, C)
            den = q.norm(dim=-1) * (k.norm(dim=-1) * m).sum(1) + 1e-6                   # (b, C)
        if self.wlog is not None and self.audit is None:                                 # R1_ICL_WDUMP: demo weights of the normal pass
            self.wlog.append((s / den[:, None, :]).mean(-1).detach().float().cpu())       # (b, n) mean over variables
        icl = (s[..., None] * v).sum(1) / den[..., None]                                 # (b, C, K*P)
        if self.nshrink:                                                                 # eta * n / (n + c), n = valid demos
            nv = m[..., 0].sum(1)[:, None, None]                                         # (b, 1, 1)
            icl = icl * nv / (nv + self.nshrink)
        add = (icl.reshape(b, c, hmax, P) * self.eta[None, None, :, None]).reshape(b, c, hmax * P)   # Delta_t, the ICL term
        if self.case_collect and self.audit is None:
            self._icl_log.append(add.detach().float().cpu())
        add = self.case_transform(add)
        self.win_ptr += b
        return delta + add.reshape(b, c, hmax, P)

    SEL_SETS = ('all', 'recent4', 'phase', 'topk8')

    def sel_masks(self, Phi, x, qpos, valid):
        """Demo sets (b, C, n) bool for a query at origin qpos with features x (b, C, r): lag = qpos - j, usable iff lag >= K."""
        b, C, n, _ = Phi.shape
        j = torch.arange(n, device=Phi.device)
        lag = qpos - j
        base = ((lag >= self.K_last)[None, None] & valid[:, None, :]).expand(b, C, n)
        ph = torch.zeros(n, dtype=torch.bool, device=Phi.device)
        for l_ in self.sel_phase:
            ph |= lag == l_
        sets = dict(all=base, recent4=base & (lag < self.K_last + 4)[None, None], phase=base & ph[None, None])
        sim = (Phi @ x[..., None]).squeeze(-1).masked_fill(~base, float('-inf'))
        kk = int(min(8, base[0, 0].sum()))
        tk = torch.zeros_like(base)
        tk.scatter_(-1, sim.topk(kk, -1).indices, True)
        sets['topk8'] = tk & base
        return {k_: torch.where(m_.any(-1, keepdim=True), m_, base) for k_, m_ in sets.items()}, base   # empty row -> all usable

    def sel_fit(self, Phi, R, x, M, base, onestep, hmax, P):
        """Delta for features x from the demos in M: converged ridge in dual form (lam = mean |phi|^2 over the usable demos) or hyb's
        normalised one GD step restricted to M (times eta). Phi (b, C, n, r), R (b, C, n, d), x (b, C, r) -> (b, C, d)."""
        Mf = M.to(Phi.dtype)
        Pm = Phi * Mf[..., None]
        if not onestep:
            Pm = Pm.double()
        if onestep:
            s_ = (Pm @ x[..., None]).squeeze(-1)
            den = x.norm(dim=-1) * (Phi.norm(dim=-1) * Mf).sum(-1) + 1e-6
            out = (s_[..., None] * R).sum(-2) / den[..., None]
            b, C, d = out.shape
            return (out.reshape(b, C, hmax, P) * self.eta[None, None, :, None]).reshape(b, C, d)
        bf = base.to(Phi.dtype)
        lam = ((Phi.square().sum(-1) * bf).sum(-1) / bf.sum(-1).clamp_min(1)).clamp_min(1e-6)        # (b, C)
        n = Phi.shape[2]
        G = Pm @ Pm.transpose(-1, -2) + lam.double()[..., None, None] * torch.eye(n, device=Phi.device, dtype=torch.float64)
        alpha = torch.linalg.solve(G, Pm @ x[..., None].double())                                       # (b, C, n, 1), float64
        return ((R * Mf[..., None]).transpose(-1, -2).double() @ alpha).squeeze(-1).to(Phi.dtype)

    def sel_correct(self, q, k, v, mask, b, c, hmax, P, m_val=4):
        """PAT-313 point 6. Stage 1: pick where to look per window and variable by validating each demo set on the m_val newest demos
        (pseudo-queries, each using only demos at lag >= K from it); stage 2: fit TEFL's map correction on the chosen demos
        (iclsel: ridge, softmax(-err / min err) over sets; iclsel1: one GD step; iclselfix_<set>: that set alone, ridge)."""
        assert bool(mask.all()), 'iclsel assumes unpadded eval windows (every demo valid), as iclperm'
        Phi, R = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)                          # (b, C, n, r), (b, C, n, d)
        n = Phi.shape[2]
        a = self.audit
        onestep = a in ('iclsel1', 'iclselavg1') or a.startswith(('iclselfix1_', 'iclselw_'))
        qpos = n + self.K_last - 1                                                       # the real query origin j0
        sets, base = self.sel_masks(Phi, q, qpos, mask)
        if a.startswith(('iclselfix_', 'iclselfix1_')):                                  # one fixed set (ridge | one GD step)
            name = a.split('_', 1)[1]
            return self.sel_fit(Phi, R, q, sets[name], base, onestep, hmax, P)
        outs = torch.stack([self.sel_fit(Phi, R, q, sets[name], base, onestep, hmax, P) for name in self.SEL_SETS])
        if a in ('iclselavg', 'iclselavg1'):                                             # control: fixed uniform combination
            return outs.mean(0)
        if a.startswith('iclselw_'):                                                     # given per-window weights (S, N, C), e.g. causal Hedge
            if a not in self.selw_cache:
                self.selw_cache[a] = torch.load(f"{os.environ['R1_ICL_SELW']}_{a[len('iclselw_'):]}.pt").float()
            W = self.selw_cache[a]
            idx = torch.arange(self.win_ptr, self.win_ptr + q.shape[0])
            assert int(idx[-1]) < W.shape[1], 'iclselw: fewer weights than windows'
            return (W[:, idx].to(outs)[..., None] * outs).sum(0)
        err = torch.zeros(len(self.SEL_SETS), b, c, device=Phi.device, dtype=torch.float64)
        for p in range(max(0, n - m_val), n):
            xp = Phi[:, :, p]
            sp, bp = self.sel_masks(Phi, xp, p, mask)
            for i, name in enumerate(self.SEL_SETS):
                err[i] += (R[:, :, p] - self.sel_fit(Phi, R, xp, sp[name], bp, onestep, hmax, P)).double().square().sum(-1)
        w = torch.softmax(-err / err.min(0).values.clamp_min(1e-12), dim=0).to(Phi.dtype)   # (S, b, C)
        self.sel_log.append(w.mean((1, 2)).detach().cpu())
        if self.sel_wdump is not None:                                                    # per-window weights (S, b, C)
            self.sel_wdump.append(w.detach().float().cpu())
        return (w[..., None] * outs).sum(0)

    def case_basis(self, kind, n, device):
        """Orthonormal (n, m) basis over the n lead steps: const | lin (const + linear) | daily (lin + sin/cos of the period, 2 harmonics)."""
        t = torch.arange(n, dtype=torch.float64)
        cols = [torch.ones(n)]
        if kind in ('lin', 'daily'):
            cols.append(t - t.mean())
        if kind == 'daily':
            assert self.case_period > 0, 'R1_ICL_PERIOD needed for the daily basis'
            for h in (1, 2):
                w = 2 * torch.pi * h * t / self.case_period
                cols += [torch.sin(w), torch.cos(w)]
        Q, _ = torch.linalg.qr(torch.stack([c_.double() for c_ in cols], 1))
        return Q.float().to(device)

    def case_transform(self, add):
        """PAT-313 audits on Delta_t (b, C, n): iclconst = the test-mean Delta; iclproj_{const,lin,daily} = projection on that basis per
        (window, variable); iclproj_common = the mean over variables. Anything else: unchanged."""
        a = self.audit
        if a == 'iclconst':
            return self.icl_mean.to(add)[None].expand_as(add)
        if a is not None and a.startswith('iclproj_'):
            kind = a[len('iclproj_'):]
            if kind == 'common':
                return add.mean(1, keepdim=True).expand_as(add)
            B = self.case_basis(kind, add.shape[-1], add.device).to(add.dtype)
            return (add @ B) @ B.T
        return add

    def finalize_case(self):
        """After the normal pass: stack the demo bank (N, n, C, d) and the Delta_t log; icl_mean = mean Delta over the pass's windows."""
        self.bank = tuple(torch.cat(z) for z in zip(*self._bank))
        icl = torch.cat(self._icl_log)
        self.icl_mean = icl.float().mean(0)
        self._bank, self._icl_log = [], []
        return icl
