"""PAT-304 R1: recurrent MambaTS baseline (no memory). Spec: Linear comments 40890e06, 8835a58c, 722e5953, d4ad8aee.

Tokens are the official per-variable patch tokens (shared value embedding), scanned TIME-MAJOR: at every time patch t the C variable
tokens are scanned in the order pi, then t+1. After the C tokens of time t (the patch boundary) the real last-layer SSM state
h_t (b, d_inner, d_state) is read once: Linear(d_inner*d_state -> C*P) gives the next patch of every variable in SCAN-POSITION order,
restored to the original variable order with pi^-1. Forecasting is autoregressive: the predicted patch is fed back as the next C tokens
(scan order pi) through every layer, carrying each layer's SSM state; any lookback (multiple of P) and any number of future patches.

VPT keeps the original MambaTS policy, re-indexed for time-major tokens: training draws one random pi per sample, used for the whole
window and its rollout; the within-step transitions pi_k -> pi_k+1 (plus the start tuple) are exactly the original channel-major
transition set, so the original cost definition is kept (per-sample loss mean over (t, c), divided by the batch std, added to each
of the sample's C transitions). It is re-implemented here (update_vpt_cost) because the original encoder.batch_update_state broadcasts
the per-sample cost with cost.repeat(C), which interleaves samples ([c0, c1, c0, c1, ...]) against the sample-major transition
indices; the official MambaTS / A0 path is left untouched. Evaluation uses one fixed order from the original open-path ATSP on the
accumulated mean cost, with never-observed edges filled pessimistically (vpt_adjacency) instead of the original 0/0 = NaN. The cross-step wrap edge pi_C(t) -> pi_1(t+1) exists in the scan but
is not an ATSP cost term in this version (closed tour = later ablation).

Lower layers run the official selective-scan kernel over the teacher-forced history for their token outputs; their final state
(the rollout cache) is recomputed in closed form so the future loss back-propagates into the history scan (the kernel's last_state
has no gradient). The last layer runs an explicit recurrence so its state is captured at every boundary. Rollout steps use the conv-off one-token step. The last layer's token outputs, norm_f and
the official flatten head are unused (only the last layer's state is read).

Per-sample lookback (lens, PAT-304 mixed-length batches): x (b, Lb, C) is right-aligned (every window ends at the forecast origin) and
sample i uses only its last lens[i] steps. RevIN statistics are taken over the valid steps only, padded steps are zeroed, and every
padded token gets SSM step dt = 0 after the softplus (dA = 1, dBu = 0): the state stays exactly zero through the padding, so each
sample's states, history predictions and forecast equal those of the unpadded window (only valid tokens touch the state; the conv-off
mixer has no other cross-token path). lens=None is the original code path, bit for bit."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from models.MambaTS import Model
from patchmem.ref_scan import mamba_proj, mamba_step_noconv


def atsp_order(adj, solver, n_vars):
    """Original MambaTS evaluation order (layers/mamba_ssm/mixer2_seq_simple.py, MixerTSModel.forward, eval branch): open-path ATSP
    on the mean transition cost adj (C, C), start = argmin of the diagonal. Identity if adj is None (no transition recorded)."""
    if adj is None:
        return list(range(n_vars))
    if solver == 'GD':
        start = torch.argmin(torch.diag(adj)).item()
        adj = adj.clone()
        for j in range(n_vars):
            adj[j, j] = float('inf')
        cur, seen = start, [start]
        while len(seen) < n_vars:
            adj[:, cur] = float('inf')
            cur = torch.argmin(adj[cur]).item()
            seen.append(cur)
        return seen
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing, solve_tsp_lin_kernighan
    start = torch.argmin(torch.diag(adj)).item()
    adj = adj.clone()
    for j in range(n_vars):
        adj[j, j] = 0
    dm = adj.cpu().numpy()
    dm[:, start] = 0                                                                    # open path: returning to the start is free
    dm[:, [start, 0]] = dm[:, [0, start]]
    dm[[start, 0], :] = dm[[0, start], :]
    solve = dict(SA=solve_tsp_simulated_annealing, LS=solve_tsp_local_search, LK=solve_tsp_lin_kernighan)[solver]
    path, dist = solve(dm, x0=list(range(n_vars)))
    path[path.index(start)] = 0
    path[0] = start
    print(f'[R1 ATSP] order {path} cost {dist:.4f}', flush=True)
    return list(path)


class R1(nn.Module):
    def __init__(self, cfg, readout_ln=False):
        super().__init__()
        assert not cfg.use_casual_conv, 'R1 assumes the conv-off mixer (official MambaTS setting)'
        self.base = Model(cfg)
        self.base.head = nn.Identity()                                                # the official flatten head is not used
        self.enc = self.base.encoder
        self.emb = self.base.value_embedding[1]                                        # shared Linear(P -> d_model), no bias
        self.c, self.P = cfg.enc_in, cfg.patch_len
        m = self.enc.layers[-1].mixer
        self.D, self.N = m.d_inner, m.d_state
        self.ro_ln = nn.LayerNorm(self.D * self.N) if readout_ln else None              # readout input only, never the carried state
        self.readout = nn.Linear(self.D * self.N, self.c * self.P)
        self.vpt = cfg.VPT_mode == 1
        self.ids_shuffle = None                                                         # cached evaluation order (list of C ints)
        self.vpt_skipped = 0                                                            # cost updates skipped (batch std undefined / ~0)

    # ---------------------------------------------------------------- ordering
    def reset_ids_shuffle(self):
        self.ids_shuffle = None

    def scan_order(self, b, device):
        """(b, C) scan order pi: pi[:, k] = original variable scanned at position k."""
        if self.vpt and self.training and not getattr(self, 'fixed_scan', False):   # fixed_scan: frozen backbone, deployed order
            self.ids_shuffle = None                                                     # as the original: re-solve at the next eval
            perm = torch.rand(b, self.c, device=device).argsort(1)
            ids_ = torch.cat([perm[:, :1], perm], 1)                                    # start tuple (pi_1, pi_1), then pi_k -> pi_k+1
            self.enc.transition_tuple = torch.stack([ids_[:, :-1], ids_[:, 1:]], -1)
            return perm
        if not self.vpt:
            return torch.arange(self.c, device=device).expand(b, -1)
        if self.ids_shuffle is None:
            self.ids_shuffle = atsp_order(self.vpt_adjacency(), self.enc.ATSP_solver, self.c)
        return torch.as_tensor(self.ids_shuffle, device=device).expand(b, -1)

    def update_vpt_cost(self, loss_bhc, valid=None):
        """VPT cost accumulation from the per-element forecast loss (b, H, C) of the last training forward. Same definition as the
        original batch_update_state, with the per-sample cost broadcast sample-major (repeat_interleave) to match the transition
        indices. A batch whose std is undefined (b = 1) or ~0 carries no relative information and is skipped (counted in vpt_skipped)."""
        if not self.vpt:
            return
        enc = self.enc
        if valid is None:
            cost = loss_bhc.detach().float().mean((1, 2))                               # (b,) per-sample mean over (t, c)
        else:                                                                           # valid (b, H): the sample's own horizon only
            w = valid.to(torch.float32).unsqueeze(-1)
            cost = (loss_bhc.detach().float() * w).sum((1, 2)) / (w.sum((1, 2)) * loss_bhc.shape[-1])
        sd = cost.std() if cost.numel() > 1 else cost.new_tensor(float('nan'))
        if not torch.isfinite(sd) or sd <= 1e-6 * cost.abs().mean() + 1e-12:
            self.vpt_skipped += 1
            return
        cost = cost / sd
        tt = enc.transition_tuple                                                       # (b, C, 2) from this forward
        assert tt.shape[0] == cost.shape[0], (tt.shape, cost.shape)
        idx = (tt[..., 0] * self.c + tt[..., 1]).reshape(-1)                            # sample-major: s0's C edges, s1's C edges, ...
        enc.adjacency_matrix.view(-1).scatter_add_(0, idx, cost.repeat_interleave(tt.shape[1]))
        enc.count_matrix.view(-1).scatter_add_(0, idx, torch.ones_like(cost).repeat_interleave(tt.shape[1]))
        enc.ending_points.scatter_add_(0, tt[:, -1, 1], cost)

    def vpt_adjacency(self):
        """Mean transition cost (C, C) shifted as the original get_adjacency_matrix (+|min| + 1e-7). Never-observed edges (count 0;
        the original gives 0/0 = NaN, which also turns the min shift and so every entry NaN) take the largest observed mean, so an
        unknown edge is never preferred. None if nothing was recorded."""
        cnt, tot = self.enc.count_matrix, self.enc.adjacency_matrix
        seen = cnt > 0
        if not bool(seen.any()):
            return None
        mean = torch.where(seen, tot / cnt.clamp_min(1), torch.zeros_like(tot))
        adj = torch.where(seen, mean, mean[seen].max())
        return adj + adj.min().abs() + 1e-7

    # ---------------------------------------------------------------- layers
    def _norm(self, layer, hidden, residual):
        return layer_norm_fn(hidden, layer.norm.weight, layer.norm.bias, residual=residual, prenorm=True,
                             residual_in_fp32=layer.residual_in_fp32, eps=layer.norm.eps)

    @staticmethod
    def _final_state(x, dt, A, B, bias):
        """Differentiable final SSM state of a zero-initialised scan, closed form:
        h_l = sum_s exp(A * sum_{r>s} dt_r) * dt_s * x_s * B_s  (dt after softplus; every exponent <= 0)."""
        dt = F.softplus(dt + bias[None, :, None])                                      # (b, d, l)
        rc = dt.flip(-1).cumsum(-1).flip(-1)                                           # sum_{r>=s} dt_r
        after = torch.cat([rc[..., 1:], torch.zeros_like(rc[..., :1])], -1)            # sum_{r>s} dt_r
        return torch.einsum('bdln,bdl,bnl->bdn', torch.exp(after.unsqueeze(-1) * A[None, :, None, :]), dt * x, B)

    @staticmethod
    def _final_state_delta(x, delta, A, B):
        """_final_state with the step size already after softplus (and masking): delta (b, d, l)."""
        rc = delta.flip(-1).cumsum(-1).flip(-1)
        after = torch.cat([rc[..., 1:], torch.zeros_like(rc[..., :1])], -1)
        return torch.einsum('bdln,bdl,bnl->bdn', torch.exp(after.unsqueeze(-1) * A[None, :, None, :]), delta * x, B)

    @staticmethod
    def _mixer_full(m, hn, tm=None):
        """Official conv-off Mamba forward with the selective-scan kernel (token outputs for the next layer) plus the final SSM state.
        The kernel's return_last_state gets no gradient (mamba_ssm selective_scan_fn docstring), which would cut the
        history -> rollout-cache path, so the cache is recomputed differentiably from the same projections (same dropout sample)."""
        x, dt, A, B, C, D, z, bias = mamba_proj(m, hn)
        if tm is not None:                                                              # padded tokens: dt = 0, state untouched
            delta = F.softplus(dt + bias[None, :, None]) * tm[:, None, :]
            y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=None, delta_softplus=False)
            return m.out_proj(y.transpose(1, 2)), R1._final_state_delta(x, delta, A, B)
        y = selective_scan_fn(x, dt, A, B, C, D, z=z, delta_bias=bias, delta_softplus=True)
        return m.out_proj(y.transpose(1, 2)), R1._final_state(x, dt, A, B, bias)

    def _last_full(self, m, hn, tm=None):
        """Last layer over the history: explicit recurrence, state captured after every C tokens. Returns (b, T, D, N), final state."""
        x, dt, A, B, C, D, z, bias = mamba_proj(m, hn)
        dt = F.softplus(dt + bias[None, :, None])
        if tm is not None:
            dt = dt * tm[:, None, :]
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])
        dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        h = x.new_zeros(x.shape[0], self.D, self.N)
        caps = []
        for s in range(x.shape[-1]):
            h = dA[:, :, s] * h + dBu[:, :, s]
            if (s + 1) % self.c == 0:
                caps.append(h)
        return torch.stack(caps, 1), h

    @staticmethod
    def _last_step(m, hn, state):
        """Last layer, one token: state update only (its token output is never read)."""
        x, _ = m.in_proj(hn.squeeze(1)).chunk(2, dim=-1)
        dt, B, _ = torch.split(m.x_proj(x), [m.dt_rank, m.d_state, m.d_state], dim=-1)
        dt = F.softplus(F.linear(dt, m.dt_proj.weight) + m.dt_proj.bias.to(dt.dtype))
        A = -torch.exp(m.A_log.float())
        return state * torch.exp(torch.einsum('bd,dn->bdn', dt, A)) + x.unsqueeze(-1) * torch.einsum('bd,bn->bdn', dt, B)

    def step_time(self, tok, states):
        """Feed the C tokens of one time patch (b, C, d_model, scan order) through all layers, one token at a time.
        states: per-layer SSM states (updated, returned). The last layer's state after the C-th token is the boundary state."""
        layers = self.enc.layers
        for k in range(tok.shape[1]):
            hidden, residual = tok[:, k:k + 1], None
            for i, layer in enumerate(layers):
                hn, residual = self._norm(layer, hidden, residual)
                if i < len(layers) - 1:
                    hidden, states[i] = mamba_step_noconv(layer.mixer, hn, states[i])
                else:
                    states[i] = self._last_step(layer.mixer, hn, states[i])
        return states

    def history(self, tok, tm=None):
        """Teacher-forced pass over the time-major history tokens (b, T*C, d_model). Returns boundary states (b, T, D, N) and per-layer states.
        tm (b, T*C) token validity (1 = valid, 0 = padding) or None."""
        layers = self.enc.layers
        hidden, residual, states = tok, None, []
        for i, layer in enumerate(layers):
            hn, residual = self._norm(layer, hidden, residual)
            if i < len(layers) - 1:
                hidden, last = self._mixer_full(layer.mixer, hn, tm)
                states.append(last)
            else:
                bstates, last = self._last_full(layer.mixer, hn, tm)
                states.append(last)
        return bstates, states

    # ---------------------------------------------------------------- readout / io
    def read(self, h):
        """h (..., D, N) -> next patch (..., C, P) in SCAN-POSITION order."""
        f = h.flatten(-2)
        if self.ro_ln is not None:
            f = self.ro_ln(f)
        return self.readout(f).unflatten(-1, (self.c, self.P))

    @staticmethod
    def to_scan(xo, perm):
        """(b, ..., C, P) original order -> scan order."""
        idx = perm.view(perm.shape[0], *([1] * (xo.dim() - 3)), perm.shape[1], 1).expand_as(xo)
        return xo.gather(-2, idx)

    @staticmethod
    def to_orig(xs, perm):
        """(b, ..., C, P) scan order -> original order (pi^-1)."""
        return R1.to_scan(xs, perm.argsort(1))

    def embed(self, xs):
        return self.emb(xs)

    def revin(self, x, lens=None):
        """RevIN of the lookback x (b, L, C). lens (b,) valid lengths of right-aligned windows or None (all valid). Returns means, stdev
        (b, 1, C, detached), xn (b, L, C) normalised with padded steps zeroed, t0 (b,) first valid patch, tm (b, T*C) token validity
        (t0 / tm None when lens is None)."""
        b, L, c = x.shape
        if lens is None:
            means = x.mean(1, keepdim=True).detach()
            xc = x - means
            stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            return means, stdev, xc / stdev, None, None
        lens = lens.to(x.device)
        assert bool((lens % self.P == 0).all()) and bool((lens > 0).all()) and bool((lens <= L).all()), lens
        vt = (torch.arange(L, device=x.device)[None, :] >= (L - lens)[:, None]).to(x.dtype).unsqueeze(-1)   # (b, L, 1)
        n = lens.to(x.dtype).view(b, 1, 1)
        means = ((x * vt).sum(1, keepdim=True) / n).detach()
        xc = (x - means) * vt
        stdev = torch.sqrt(xc.square().sum(1, keepdim=True) / n + 1e-5).detach()
        T = L // self.P
        t0 = T - lens // self.P
        tm = (torch.arange(T, device=x.device)[None, :] >= t0[:, None]).to(x.dtype).repeat_interleave(c, 1)
        return means, stdev, xc / stdev, t0, tm

    def forward(self, x, n_out, perm=None, trace=None, lens=None):
        """x (b, L, C) dataset-scaled lookback, L a multiple of P. n_out future patches. lens (b,): per-sample lookback (right-aligned
        windows, see the module doc) or None.
        Returns dict(pred (b, n_out*P, C) de-normalised forecast, hist (b, T-1, P, C) de-normalised next-patch predictions inside the
        lookback (target x[:, P:]; with lens, entries j < t0 are padding), perm (b, C)). trace (dict) receives the boundary states and
        the final history state."""
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        means, stdev, xn, t0, tm = self.revin(x, lens)
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)                                    # original variable order
        xs = self.to_scan(xp.permute(0, 2, 1, 3), perm)                                 # (b, T, C, P) scan order
        tok = self.embed(xs).reshape(b, T * c, -1)                                     # time-major: t0 (C tokens), t1, ...
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)                                    # (b, T, C, P) original order
        outs = [yh[:, -1]]
        for _ in range(n_out - 1):
            states = self.step_time(self.embed(self.to_scan(outs[-1].unsqueeze(1), perm)).squeeze(1), states)
            outs.append(self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1))
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)                   # y (b, n, P, C)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        return dict(pred=pred, hist=hist, perm=perm)
