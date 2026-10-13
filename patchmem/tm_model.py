"""PAT-304 A1: MambaTS with a time-major patch scan and a state-to-target readout G (memory OFF).
Weights are the official MambaTS modules (value_embedding, encoder.layers, conv-off Mamba-1 TMB); only the token order and the
readout differ. Token order: patch-major, variables (VPT/VAST order `perm`) inside each patch, so the SSM cache state after patch j
(`h_j^-`, one (d_inner, d_state) matrix per layer per sample) has seen exactly X_{<=j}. Normalisation is the official window instance norm
(means/stdev may be passed in to hold them fixed)."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.ref_scan import mamba_forward_capture


class TMMambaTS(nn.Module):
    def __init__(self, base, readout_layer=-1):
        super().__init__()
        self.base = base                                   # official models.MambaTS.Model (A0 weights load into it unchanged)
        self.patch_len, self.stride, self.seq_len, self.pred_len = base.patch_len, base.stride, base.seq_len, base.pred_len
        self.n_vars = base.encoder.n_vars
        m0 = base.encoder.layers[0].mixer
        self.readout_layer = readout_layer % len(base.encoder.layers)
        self.G = nn.Linear(m0.d_inner * m0.d_state, self.pred_len * self.n_vars)

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

    def forward(self, x_enc, perm=None, stats=None, hooks=None, ckpt=False, inject_layers=None):
        """hooks: {layer_idx: hook(i, h_minus) -> delta|None}, see ref_scan. Returns dict:
        pred (b, nb, pred_len, c) = G(h_j^-) denormalised; states (n_layers, b, nb, d_inner, d_state) as a list; stats."""
        xn, stats = self.norm(x_enc, stats)
        hs, nb = self.embed(xn, perm)
        c = self.n_vars
        bounds = tuple(c * (j + 1) for j in range(nb))
        res, states = None, []
        for li, blk in enumerate(self.base.encoder.layers):
            res = hs if res is None else hs + res
            hs, _, st = mamba_forward_capture(blk.mixer, blk.norm(res), bounds, ckpt=ckpt, hook=(hooks or {}).get(li))
            states.append(st)
        hr = states[self.readout_layer]                                           # b nb d n
        yh = self.G(hr.flatten(2)).reshape(hr.shape[0], nb, self.pred_len, c)
        means, stdev = stats
        pred = yh * stdev[:, None, [0]] + means[:, None, [0]]
        return dict(pred=pred, pred_norm=yh, states=states, stats=stats, out=hs, res=res)


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
