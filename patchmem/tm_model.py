"""PAT-304 A1: MambaTS with a time-major patch scan and a state-to-target readout G (memory OFF).
Weights are the official MambaTS modules (value_embedding, encoder.layers, conv-off Mamba-1 TMB); only the token order and the
readout differ. Token order: patch-major, variables (VPT/VAST order `perm`) inside each patch, so the SSM cache state after patch j
(`h_j^-`, one (d_inner, d_state) matrix per layer per sample) has seen exactly X_{<=j}. Normalisation is the official window instance norm
(means/stdev may be passed in to hold them fixed)."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.ref_scan import mamba_forward_capture, mamba_proj, _chunk


def _post_only(hook):
    def f(i, h):
        o = hook(i, h)
        assert o is None or o[0] is None, 'd_read is only allowed on the readout layer'
        return o
    return f


class CorrNet(nn.Module):
    """Residual-driven correction of the readout layer's SSM state, per SSM channel k: C_k = f(h_k, E(e)).
    mode: shared (one f for all channels) | shared_id (one f + learned channel embedding) | channel (independent f_k per channel).
    The output layer is zero-initialised, so C == 0 at init."""
    def __init__(self, mode, d_inner, d_state, res_dim, z_dim=32, hidden=64, id_dim=8):
        super().__init__()
        assert mode in ('shared', 'shared_id', 'channel')
        self.mode = mode
        self.E = nn.Sequential(nn.Linear(res_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim))
        inp = d_state + z_dim + (id_dim if mode == 'shared_id' else 0)
        if mode == 'shared_id':
            self.ch_emb = nn.Parameter(torch.randn(d_inner, id_dim) * 0.1)
        K = d_inner if mode == 'channel' else 1
        self.W1 = nn.Parameter(torch.randn(K, inp, hidden) * inp ** -0.5); self.b1 = nn.Parameter(torch.zeros(K, hidden))
        self.W2 = nn.Parameter(torch.zeros(K, hidden, d_state)); self.b2 = nn.Parameter(torch.zeros(K, d_state))

    def forward(self, h, e):
        """h (b, d_inner, d_state) prior state (caller passes it detached), e (b, res_dim) residual. Returns C (b, d_inner, d_state)."""
        b, d, _ = h.shape
        parts = [h, self.E(e)[:, None].expand(-1, d, -1)]
        if self.mode == 'shared_id':
            parts.append(self.ch_emb[None].expand(b, -1, -1))
        u = torch.cat(parts, -1)
        if self.mode != 'channel':
            return F.gelu(u @ self.W1[0] + self.b1[0]) @ self.W2[0] + self.b2[0]
        a = F.gelu(torch.einsum('bdi,dih->bdh', u, self.W1) + self.b1)
        return torch.einsum('bdh,dhn->bdn', a, self.W2) + self.b2


class TMMambaTS(nn.Module):
    def __init__(self, base, readout_layer=-1, head='state', detach_trunk=False, corr=None, corr_shuffle=False, corr_res='full'):
        super().__init__()
        assert head in ('state', 'official', 'causal')
        assert not detach_trunk or head == 'causal'
        assert corr is None or head == 'causal'
        self.head_mode = head
        self.detach_trunk = detach_trunk   # causal head only: stop-gradient the shared trunk (embed/encoder/causal_W) from every origin but the last;
                                            # only causal_gain[j]/causal_bias[j] (per-origin "correction") are trained from origins j < n_patch-1
        self.base = base                                   # official models.MambaTS.Model (A0 weights load into it unchanged)
        self.patch_len, self.stride, self.seq_len, self.pred_len = base.patch_len, base.stride, base.seq_len, base.pred_len
        self.n_vars = base.encoder.n_vars
        self.n_patch = self.seq_len // self.patch_len
        m0 = base.encoder.layers[0].mixer
        self.readout_layer = readout_layer % len(base.encoder.layers)
        self.G = nn.Linear(m0.d_inner * m0.d_state, self.pred_len * self.n_vars)
        if head == 'causal':                                # causal block-decomposed official head: reads only the per-token output stream o
            d = self.base.value_embedding[1].out_features    # (d_model), not the SSM state; at j = n_patch-1 this is exactly the official head
            self.causal_W = nn.Parameter(torch.randn(self.n_patch, d, self.pred_len) * d ** -0.5)
            self.causal_bias = nn.Parameter(torch.zeros(self.n_patch, self.pred_len))
            self.causal_gain = nn.Parameter(torch.ones(self.n_patch))
        # residual-driven state correction on the readout layer. Y_j (pred_len steps after patch j) is fully observed at the end of patch
        # j + corr_delay, corr_delay = ceil(pred_len / patch_len). At boundary t (corr_delay <= t <= n_patch-2; a correction at the last boundary
        # reaches no later token): e = Y_{t-d} - stopgrad(Yhat_{t-d}^-), C_t = CorrNet(stopgrad h_t^-, e), the scan continues from h_t^+ = h_t^- + C_t.
        # corr_res='next' (innovation): only the first patch_len steps of Y_{t-1}, i.e. patch t itself: e = X_t - Yhat_{t-1}[:patch_len], d = 1.
        assert corr_res in ('full', 'next') and (corr_res == 'full' or self.patch_len <= self.pred_len)
        self.corr_shuffle, self.corr_res = corr_shuffle, corr_res
        self.corr_res_len = self.pred_len if corr_res == 'full' else self.patch_len
        self.corr_delay = -(-self.pred_len // self.patch_len) if corr_res == 'full' else 1
        self.corr_jmax = self.n_patch - 2 - self.corr_delay              # last origin whose residual is used
        self.corr_net = None
        if corr is not None:
            assert self.corr_jmax >= 0
            ml = base.encoder.layers[self.readout_layer].mixer
            self.corr_net = CorrNet(corr, ml.d_inner, ml.d_state, self.corr_res_len * self.n_vars)

    def norm(self, x_enc, stats=None):
        if stats is None:
            means = x_enc.mean(1, keepdim=True).detach()
            xc = x_enc - means
            stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        else:
            means, stdev = stats
        return (x_enc - means) / stdev, (means, stdev)

    def embed(self, xn, perm):
        b, _, c = xn.shape
        x = xn.permute(0, 2, 1).unfold(-1, self.patch_len, self.stride)          # b c l P
        e = self.base.value_embedding(x)                                          # (b c) l d
        e = rearrange(e, '(b c) l d -> b l c d', c=c)
        if perm is not None:
            e = torch.gather(e, 2, perm[:, None, :, None].expand(-1, e.shape[1], -1, e.shape[3]))
        return rearrange(e, 'b l c d -> b (l c) d'), e.shape[1]

    def _corrected_layer(self, blk, res, xn, perm, nb, c):
        """Readout layer scanned patch by patch with residual-driven state correction (see corr_net in __init__). Returns the mixer output
        (b, L, dm), prior boundary states h_t^- (b, nb, d_inner, d_state) and the correction regulariser mean_t mean(C_t^2)."""
        m = blk.mixer
        x, dt, A, B, C, D, z, bias = mamba_proj(m, blk.norm(res))
        dt = F.softplus(dt + bias[None, :, None])
        b = x.shape[0]
        h = x.new_zeros(b, x.shape[1], A.shape[1])
        nrm = self.base.encoder.norm_f
        inv = torch.argsort(perm, dim=1) if perm is not None else None
        P, L, d = self.patch_len, self.corr_res_len, self.corr_delay
        ys, priors, regs, yhats, cum = [], [], [], [], 0
        for t in range(nb):
            s, e = c * t, c * (t + 1)
            h, y = _chunk(h, x[..., s:e], dt[..., s:e], B[..., s:e], C[..., s:e], A, D, z[..., s:e])
            ys.append(y); priors.append(h)
            if t <= self.corr_jmax:                                            # prior prediction at origin t, exactly as the causal head reads it
                with torch.no_grad():
                    tok = F.layer_norm(m.out_proj(y.transpose(1, 2)) + res[:, s:e], (res.shape[-1],), nrm.weight, nrm.bias, nrm.eps)
                    if inv is not None:
                        tok = torch.gather(tok, 1, inv[:, :, None].expand(-1, -1, tok.shape[-1]))
                    cum = cum + tok @ self.causal_W[t]                          # b c H
                    yhats.append((cum * self.causal_gain[t] + self.causal_bias[t]).transpose(1, 2))   # b H c, normalised scale
            j = t - d
            if j < 0 or j > self.corr_jmax:
                continue
            e_res = (xn[:, P * (j + 1): P * (j + 1) + L] - yhats[j][:, :L]).reshape(b, -1)   # ends at P(t+1): observed by the end of patch t
            if self.corr_shuffle:                                              # control: residual of another sample in the batch
                e_res = e_res.roll(1, 0)
            Cm = self.corr_net(h.detach(), e_res)
            regs.append(Cm.square().mean())
            h = h + Cm                                                         # not detached: the corrected state carries gradient forward
        out = m.out_proj(torch.cat(ys, -1).transpose(1, 2))
        return out, torch.stack(priors, 1), torch.stack(regs).mean()

    def forward(self, x_enc, perm=None, stats=None, hooks=None, ckpt=False):
        """hooks: {layer_idx: hook(i, h_minus) -> None | (d_read, d_post)}, see ref_scan. d_read is only allowed on the readout layer (a read-time
        correction on a lower layer cannot reach the current patch's upper layers). Returns dict:
        pred_prior (b, nb, pred_len, c) = G(h_j^-) denormalised (the prediction made before any correction: use it for e_j = Y_j - Yhat_j);
        pred = G(h_j^read), read-time-corrected (== pred_prior without d_read); states = prior h^- per layer, reads = h_read of the readout layer; stats."""
        xn, stats = self.norm(x_enc, stats)
        hs, nb = self.embed(xn, perm)
        c = self.n_vars
        bounds = tuple(c * (j + 1) for j in range(nb))
        res, states, corr_reg = None, [], None
        for li, blk in enumerate(self.base.encoder.layers):
            res = hs if res is None else hs + res
            if self.corr_net is not None and li == self.readout_layer:
                assert not hooks
                hs, st, corr_reg = self._corrected_layer(blk, res, xn, perm, nb, c)
                states.append(st); reads = st
                continue
            hk = (hooks or {}).get(li)
            if hk is not None and li != self.readout_layer:
                hk = _post_only(hk)
            hs, _, st, rd = mamba_forward_capture(blk.mixer, blk.norm(res), bounds, ckpt=ckpt, hook=hk, return_read=True)
            states.append(st)
            if li == self.readout_layer:
                reads = rd
        means, stdev = stats
        if self.head_mode in ('official', 'causal'):   # readout from the per-token output stream o (not the SSM state), unshuffled to original variable order
            nrm = self.base.encoder.norm_f
            tok = torch.nn.functional.layer_norm(hs + res, (hs.shape[-1],), nrm.weight, nrm.bias, nrm.eps)
            tok = rearrange(tok, 'b (l c) d -> b l c d', c=c)
            inv = torch.argsort(perm, dim=1) if perm is not None else None
            if inv is not None:
                tok = torch.gather(tok, 2, inv[:, None, :, None].expand(-1, nb, -1, tok.shape[-1]))
            if self.head_mode == 'official':       # diagnostic (E2): official flatten head needs the full window; last origin only
                dec = self.base.head(rearrange(tok, 'b l c d -> b (c l) d'), n_vars=c)          # b H c
                pred = (dec * stdev[:, [0], :] + means[:, [0], :])[:, None]
                return dict(pred=pred, pred_prior=pred, states=states, reads=reads, stats=stats, out=hs, res=res)
            # causal: block-decomposed official head, Yhat_j = gain_j * sum_{l<=j} o_{l,c} @ W_l + bias_j (shared across variables); at j=nb-1
            # this sums the same 15 (d_model, pred_len) blocks the official head's one (240, 96) Linear is just the concatenation of.
            contrib = torch.einsum('blcd,ldp->blcp', tok, self.causal_W[:nb])
            cum = contrib.cumsum(1)
            if self.detach_trunk:               # block gradient to embed/encoder/causal_W from every origin but the last; gain/bias stay trainable everywhere
                cum = torch.cat([cum[:, :-1].detach(), cum[:, -1:]], 1)
            cum = cum * self.causal_gain[:nb][None, :, None, None] + self.causal_bias[:nb][None, :, None, :]
            pred = cum.permute(0, 1, 3, 2) * stdev[:, None, [0]] + means[:, None, [0]]
            return dict(pred=pred, pred_prior=pred, states=states, reads=reads, stats=stats, out=hs, res=res, corr_reg=corr_reg)
        dn = lambda hr: self.G(hr.flatten(2)).reshape(hr.shape[0], nb, self.pred_len, c) * stdev[:, None, [0]] + means[:, None, [0]]
        return dict(pred=dn(reads), pred_prior=dn(states[self.readout_layer]), states=states, reads=reads, stats=stats, out=hs, res=res)


def patch_targets(x, y_future, patch_len, pred_len):
    """Y_j = the pred_len steps right after patch j, j=0..n_patch-1, from the window x (b,L,c) and the official future block y_future (b,pred_len,c).
    Returns (b, n_patch, pred_len, c); Y_{n_patch-1} == y_future (the official target)."""
    full = torch.cat([x, y_future], 1)
    n = x.shape[1] // patch_len
    return torch.stack([full[:, patch_len * (j + 1): patch_len * (j + 1) + pred_len] for j in range(n)], 1)


def a1_loss(pred, x, y_future, patch_len, pred_len):
    """Equal-weight MSE over every patch origin (official-style, in the data scale). Returns total, per-origin MSE (n_patch,)."""
    tgt = patch_targets(x, y_future, patch_len, pred_len)
    per = (pred - tgt).square().mean((0, 2, 3))
    return per.mean(), per.detach()
