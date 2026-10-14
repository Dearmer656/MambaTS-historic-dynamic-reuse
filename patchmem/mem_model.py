"""PAT-304: retrieval memory on top of an unfrozen MambaTS backbone, fine-tuned jointly with the real forecast loss.

The official Model.forward already returns per-(variable,patch) token features in natural (unshuffled) order before the head
(models/MambaTS.py: MixerTSModel internally un-shuffles). This module hooks in right there: for every (k, t) token of the CURRENT
input window, it retrieves from same-variable matured pairs (key = historical token output o_j, value = the raw next patch x_{j+1})
-- in-window (any j, no causal restriction: the whole 720-step input window is already observed at inference time) and the R
preceding non-overlapping 720-step windows (always fully available, encoded with no_grad: only the current window's encoder pass
carries gradients) -- and adds a learned, softmax-attention-weighted correction to the token before the head reads it. The
correction starts near zero (gate initialised very negative, sigmoid-bounded) so fine-tuning begins at (approximately) the loaded
checkpoint's behaviour, to avoid the instability seen when an untrained correction enters training at full strength."""
import torch
import torch.nn as nn
from einops import rearrange


class Retrieval(nn.Module):
    def __init__(self, d_model, patch_len, dk=None, tau0=0.1, max_scale=1.0):
        super().__init__()
        dk = dk or d_model
        self.Wq = nn.Linear(d_model, dk, bias=False)
        self.Wk = nn.Linear(d_model, dk, bias=False)
        self.Wv = nn.Linear(patch_len, d_model, bias=False)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau0))))
        self.gate = nn.Parameter(torch.tensor(-4.0))
        self.max_scale = max_scale

    def forward(self, o_q, o_mem, v_mem, mask, shuffle=False):
        """o_q (b, Lq, dmodel): query token features (Lq = c*l). o_mem (b, Lm, dmodel), v_mem (b, Lm, patch_len): memory keys/raw
        values. mask (Lq, Lm) bool: same-variable gating. Returns the gated correction (b, Lq, dmodel)."""
        if shuffle:
            o_mem = o_mem.roll(1, 0); v_mem = v_mem.roll(1, 0)
        q = self.Wq(o_q); k = self.Wk(o_mem); v = self.Wv(v_mem)
        S = torch.einsum('bqd,bmd->bqm', q, k) * self.log_tau.exp().reciprocal()
        w = torch.softmax(S.masked_fill(~mask[None], float('-inf')), -1)
        corr = torch.einsum('bqm,bmd->bqd', w, v)
        return torch.sigmoid(self.gate) * self.max_scale * corr


class MambaTSMem(nn.Module):
    def __init__(self, base, R=2, dk=None, tau0=0.1, mode='mem'):
        super().__init__()
        assert mode in ('none', 'mem', 'shuf')
        self.base = base; self.mode = mode; self.R = R
        self.n_vars = base.encoder.n_vars; self.l = base.seq_len // base.patch_len
        d_model = base.value_embedding[1].out_features
        self.retr = Retrieval(d_model, base.patch_len, dk=dk, tau0=tau0) if mode != 'none' else None
        c, l = self.n_vars, self.l
        k_q = torch.arange(c).repeat_interleave(l)                                  # query index order: c outer, l inner (matches encoder output)
        k_m = torch.arange(c).repeat_interleave(l - 1).repeat(R + 1)                 # memory index order: (R+1) blocks of (c outer, l-1 inner)
        mask = k_q[:, None] == k_m[None]                                             # (c*l, (R+1)*c*(l-1)), same variable only
        self.register_buffer('mask', mask, persistent=False)

    def reset_ids_shuffle(self):                                                    # run.py resets this before its own test() call
        self.base.reset_ids_shuffle()

    def norm(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach(); xc = x_enc - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return xc / stdev, means, stdev

    def embed_encode(self, xn, grad):
        """Official embedding + encoder, natural (unshuffled) order. Returns tok (b, c*l, d) and the patches xp (b, c, l, P)."""
        n_vars = self.n_vars
        x = xn.permute(0, 2, 1).unfold(-1, self.base.patch_len, self.base.stride)    # b c l P
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            enc_in = self.base.value_embedding(x)
            enc_in = rearrange(enc_in, '(b c) l d -> b (c l) d', c=n_vars)
            enc_out, _ = self.base.encoder(enc_in)
        return enc_out, x

    def forward(self, x_enc, x_mem=None, *_):
        """x_enc (b, seq_len, c): current window. x_mem (b, R, seq_len, c) or None (mode == 'none'): the R preceding non-overlapping
        windows, same normalisation applied independently to each."""
        n_vars = self.n_vars
        xn, means, stdev = self.norm(x_enc)
        tok, xp = self.embed_encode(xn, grad=True)
        if self.mode != 'none':
            assert x_mem is not None
            b = x_enc.shape[0]
            o_win = rearrange(tok, 'b (c l) d -> b c l d', c=n_vars)[:, :, :-1]      # in-window keys, j = 0..l-2 (grad flows)
            v_win = xp[:, :, 1:]                                                     # in-window values x_{k,j+1}
            o_mem_list, v_mem_list = [o_win], [v_win]
            for r in range(self.R):
                xn_m, _, _ = self.norm(x_mem[:, r])
                tok_m, xp_m = self.embed_encode(xn_m, grad=False)
                o_m = rearrange(tok_m, 'b (c l) d -> b c l d', c=n_vars)[:, :, :-1].detach()
                o_mem_list.append(o_m); v_mem_list.append(xp_m[:, :, 1:].detach())
            # self.mask's memory layout is (R+1) blocks outer, (c, l-1) inner per block -- stack then flatten in that order
            o_mem = torch.stack(o_mem_list, 0).permute(1, 0, 2, 3, 4).reshape(b, -1, o_win.shape[-1])
            v_mem = torch.stack(v_mem_list, 0).permute(1, 0, 2, 3, 4).reshape(b, -1, v_win.shape[-1])
            corr = self.retr(tok, o_mem, v_mem, self.mask, shuffle=(self.mode == 'shuf'))
            tok = tok + corr
        dec_out = self.base.head(tok, n_vars=n_vars)
        dec_out = dec_out * stdev[:, [0], :] + means[:, [0], :]
        return dec_out
