"""PAT-304 R1 + observation-space error memory (R1Obs). Baseline = R1 (patchmem/r1_model.py, unchanged backbone); the memory stores the
backbone's own next-patch ERRORS in the observation space and adds a retrieved correction directly to the forecast (no state injection,
no CorrNet). Motivation (patchmem/mech_eval.py): R1Mem's gain is sample-specific error feedback, and a non-learned feedback of the
lookback errors (one val-fitted scalar) matches or beats it, so here the error itself is the value and the correction skips the state.

History (teacher forced, plain R1 recurrence): boundary states h_t, next-patch predictions yh_t = read(h_t) for patch t+1.
  slot t (t = 1..T-1): key  k_t = W_K sg h_t                          the state after patch t (the patch whose error is stored)
                       value e_t = sg(x_t - yh_{t-1})                  error of the backbone's prediction of patch t, (C, P), RevIN-normalised,
                                                                       ORIGINAL variable order (VPT draws a random scan order per sample)
Rollout, future patch k (k = 0..n_out-1), query state h = h_{T-1} for k = 0, else the last layer's state after the fed-back patch:
  a_h      = softmax_j(q_h . k_j,h / (sqrt(r) tau_h))  per head h (nh heads, low-rank q/k dim r), over ALL history slots (j <= T-1)
  m_h      = sum_j a_h,j e_j                            (C, P) per head
  delta_k  = g(h) * sum_h W[h] * m_h                    W (nh, C, P) per head, variable and position in the patch, ZERO at init;
                                                        g = sigmoid(W_g sg h + b_g) (C, 1) per variable, W_g = 0, b_g = 0 at init
  forecast = read(h) + delta_k                          (original order); fed back as the next patch (feed=True, default) or the backbone
                                                        prediction read(h) is fed back (feed=False: backbone rollout identical to R1)
At init W = 0 so delta = 0 and R1Obs == R1 exactly. No auxiliary loss: W, W_Q, W_K, tau, W_g are trained by the forecast loss only.
History predictions are not corrected (lambda_hist = 0 in every arm), so the stored errors are always the plain backbone's.
Training options: slot_keep p < 1 (per-sample slot dropout, as B2-drop), qk_dim r (default 8), uniform=True (a = 1/available slots,
no W_Q / W_K / tau), path=True (PAT-307 O-PaTH: PaTH-style logits q_t^T H_t ... H_{j+1} k_j, per-head Householder
H_s = I - beta_s w_s w_s^T in the q/k space with w_s = normalize(W_w sg h_s), beta_s = 2 sigmoid(W_b sg h_s); values untouched;
the path continues through the rollout states; audit 'noH' = H = I at inference), vpath=True (PAT-307 O-PathV, the intended
design: logits = q_t . k_j as O, VALUES transported along the path: m_h = sum_j a_hj H_t ... H_{j+1} e_j with one state-driven
Householder H_s = I - beta_s w_s w_s^T in the error space (C*P), w_s = normalize(W_wv sg h_s), beta_s = 2 sigmoid(W_bv sg h_s);
W / gate as O; continues through the rollout; audit 'noH' = H = I = O exactly), vshare=True (with vpath: the Householder acts in the
P-dim patch space and the SAME reflection is applied to every variable's error, w_s = normalize(W_wv sg h_s) in R^P; no mixing across
variables, W_wv d_inner*d_state -> P instead of -> C*P, as TEFL's variable-shared g), plain=True (PAT-307 O-plain, RATL-style: one head, delta = gamma * sum_j a_j e_j with a scalar gamma = 0 at init,
no per-head W, no gate). Per-sample lookback (lens, see r1_model.py): a slot is valid only from the sample's second valid patch on
(t >= t0 + 1), combined with the slot-keep mask. Inference audits: None | 'zero' (delta = 0) | 'uniform' (a = 1/M) | 'perm' (values of the M slots permuted:
key-error pairing broken)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from patchmem.r1_model import R1


class ObsMem(nn.Module):
    def __init__(self, d_inner, d_state, c, P, group=4, qk_dim=8, tau0=1.0, uniform=False, plain=False, path=False, vpath=False,
                 vshare=False):
        super().__init__()
        assert not vshare or vpath, 'vshare needs vpath'
        assert d_inner % group == 0, f'd_inner {d_inner} not divisible by group {group}'
        self.nh, self.dh, self.uniform, self.plain = 1 if plain else d_inner // group, qk_dim, uniform, plain
        assert not (path and uniform), 'path needs q/k'
        self.path, self.vpath, self.vshare = path, vpath, vshare
        if vpath:
            self.Wwv = nn.Linear(d_inner * d_state, P if vshare else c * P, bias=False)
            self.Wbv = nn.Linear(d_inner * d_state, 1)
        if path:
            self.Ww = nn.Linear(d_inner * d_state, self.nh * qk_dim, bias=False)
            self.Wb = nn.Linear(d_inner * d_state, self.nh)
        if not uniform:
            self.Wq = nn.Linear(d_inner * d_state, self.nh * qk_dim, bias=False)
            self.Wk = nn.Linear(d_inner * d_state, self.nh * qk_dim, bias=False)
            self.log_tau = nn.Parameter(torch.full((self.nh,), float(tau0)).log())
        if plain:                                                                       # scalar gamma (broadcast as W), no gate
            self.W = nn.Parameter(torch.zeros(1, 1, 1))
        else:
            self.W = nn.Parameter(torch.zeros(self.nh, c, P))                           # per-head output map, zero: R1 at init
            self.Wg = nn.Linear(d_inner * d_state, c)
            nn.init.zeros_(self.Wg.weight); nn.init.zeros_(self.Wg.bias)
        self.audit = None

    def transport(self, h, K):
        """Apply the per-head Householder H(h) to stored keys K (b, M, nh, dh); identity without path or under audit 'noH'."""
        if not self.path or self.audit == 'noH':
            return K
        z = h.detach().flatten(1)
        w = F.normalize(self.Ww(z).reshape(-1, self.nh, self.dh), dim=-1)              # (b, nh, dh)
        beta = 2 * torch.sigmoid(self.Wb(z))                                            # (b, nh)
        proj = torch.einsum('bmhd,bhd->bmh', K, w)
        return K - (beta[:, None, :] * proj).unsqueeze(-1) * w[:, None]

    def transport_v(self, h, V):
        """Apply the Householder H(h) in the error space to stored values V (b, M, C, P); identity without vpath or under 'noH'.
        vshare: H acts on the P-dim patch of every variable with the same w (b, P)."""
        if not self.vpath or self.audit == 'noH':
            return V
        z = h.detach().flatten(1)
        w = F.normalize(self.Wwv(z), dim=-1)                                            # (b, C*P), vshare (b, P)
        beta = 2 * torch.sigmoid(self.Wbv(z))                                           # (b, 1)
        if self.vshare:
            proj = torch.einsum('bmcp,bp->bmc', V, w)
            return V - (beta[:, :, None] * proj).unsqueeze(-1) * w[:, None, None, :]
        Vf = V.flatten(2)
        Vf = Vf - (beta * torch.einsum('bmd,bd->bm', Vf, w)).unsqueeze(-1) * w[:, None]
        return Vf.reshape(V.shape)

    def key(self, h):
        if self.uniform:
            return h.new_zeros(h.shape[0], self.nh, 0)
        return self.Wk(h.detach().flatten(1)).reshape(h.shape[0], self.nh, self.dh)

    def forward(self, h, keys, vals, mask=None):
        """h (b, D, N) query state; keys (b, M, nh, dh); vals (b, M, C, P) errors; mask (b, M) bool or None (False = slot dropped).
        Returns delta (b, C, P), a (b, nh, M), g (b, C, 1)."""
        b, M = keys.shape[:2]
        hq = h.detach().flatten(1)
        if self.uniform or self.audit == 'uniform':
            w = keys.new_ones(b, M) if mask is None else mask.to(keys.dtype)
            a = (w / w.sum(1, keepdim=True).clamp_min(1.0))[:, None, :].expand(b, self.nh, M)
        else:
            q = self.Wq(hq).reshape(b, self.nh, self.dh)
            logit = torch.einsum('bhd,bmhd->bhm', q, keys) / (self.dh ** 0.5 * self.log_tau.exp())[None, :, None]
            if mask is None:
                a = torch.softmax(logit, -1)
            else:                                                                       # all-dropped rows -> a = 0 (no correction)
                anyv = mask.any(1)
                mk = mask | ~anyv[:, None]
                a = torch.softmax(logit.masked_fill(~mk[:, None, :], float('-inf')), -1) * anyv[:, None, None]
        m = torch.einsum('bhm,bmcp->bhcp', a, vals)
        g = h.new_ones(b, 1, 1) if self.plain else torch.sigmoid(self.Wg(hq)).unsqueeze(-1)
        delta = g * (self.W[None] * m).sum(1)
        if self.audit == 'zero':
            delta = torch.zeros_like(delta)
        return delta, a, g


class R1Obs(R1):
    def __init__(self, cfg, readout_ln=False, mem_group=4, qk_dim=8, slot_keep=1.0, uniform=False, feed=True, plain=False, path=False,
                 vpath=False, vshare=False):
        super().__init__(cfg, readout_ln=readout_ln)
        self.obs_net = ObsMem(self.D, self.N, self.c, self.P, group=mem_group, qk_dim=qk_dim, uniform=uniform, plain=plain, path=path,
                              vpath=vpath, vshare=vshare)
        self.slot_keep, self.feed = slot_keep, feed
        self.audit, self.audit_gen = None, None
        self.mem_stats = {}

    def set_audit(self, mode=None, seed=0):
        assert mode in (None, 'zero', 'uniform', 'perm', 'noH'), mode
        self.audit = mode
        self.obs_net.audit = mode if mode in ('zero', 'uniform', 'noH') else None
        self.audit_gen = torch.Generator().manual_seed(seed) if mode == 'perm' else None

    def stats(self):
        return {k: torch.stack(v).mean().item() for k, v in self.mem_stats.items() if v}

    def _correct(self, h, K, V, mask):
        delta, a, g = self.obs_net(h, K, V, mask)
        st = self.mem_stats
        st.setdefault('g', []).append(g.mean().detach())
        st.setdefault('delta_rms', []).append(delta.square().mean().sqrt().detach())
        if K.shape[1] >= 2:
            st.setdefault('ent_norm', []).append((-(a * a.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(K.shape[1])))).mean().detach())
        return delta

    def forward(self, x, n_out, perm=None, trace=None, lens=None):
        """Same interface and pipeline as R1.forward; the rollout forecasts get the retrieved error correction."""
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        self.mem_stats = {}
        means, stdev, xn, t0, tm = self.revin(x, lens)
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xo = xp.permute(0, 2, 1, 3)                                                     # (b, T, C, P) original order
        xs = self.to_scan(xo, perm)
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)                                    # (b, T, C, P) original order
        if T >= 2:
            K = self.obs_net.key(bstates[:, 1:].flatten(0, 1)).unflatten(0, (b, T - 1))   # (b, T-1, nh, dh) slots t = 1..T-1
            if self.obs_net.path:                                                       # slot i (state bstates[:, i+1]): H of that state
                Kt = K[:, :1]                                                           # first acts on the older keys, then k_i joins
                for i in range(1, T - 1):
                    Kt = torch.cat([self.obs_net.transport(bstates[:, i + 1], Kt), K[:, i:i + 1]], 1)
                K = Kt                                                                  # K[:, j] = H_{T-1} ... H_{j+1} k_j
            V = (xo[:, 1:] - yh[:, :-1]).detach()                                       # (b, T-1, C, P) e_t
            mask = (torch.rand(b, T - 1, device=x.device) < self.slot_keep) if self.training and self.slot_keep < 1 else None
            if t0 is not None:                                                          # slot t = j + 1 valid iff t >= t0 + 1
                ok = torch.arange(1, T, device=x.device)[None, :] >= (t0 + 1)[:, None]
                mask = ok if mask is None else mask & ok
            if self.audit == 'perm' and T - 1 >= 2:
                V = V[:, torch.randperm(T - 1, generator=self.audit_gen).to(V.device)]
            if self.obs_net.vpath:                                                      # values along the path (after the perm audit)
                Vt = V[:, :1]
                for i in range(1, T - 1):
                    Vt = torch.cat([self.obs_net.transport_v(bstates[:, i + 1], Vt), V[:, i:i + 1]], 1)
                V = Vt                                                                  # V[:, j] = H_{T-1} ... H_{j+1} e_j
        base = yh[:, -1]
        outs = []
        for k in range(n_out):
            if k:
                fb = outs[-1] if self.feed else base
                states = self.step_time(self.embed(self.to_scan(fb.unsqueeze(1), perm)).squeeze(1), states)
                base = self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1)
                if T >= 2 and self.obs_net.path:                                        # the path continues through the rollout
                    K = self.obs_net.transport(states[-1], K)
                if T >= 2 and self.obs_net.vpath:
                    V = self.obs_net.transport_v(states[-1], V)
            h = bstates[:, -1] if k == 0 else states[-1]
            outs.append(base + self._correct(h, K, V, mask) if T >= 2 else base)
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        return dict(pred=pred, hist=hist, perm=perm)
