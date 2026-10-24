"""PAT-307 R1TeflAtt: TEFL (R1Tefl, T0r / TF0) + a self-attention read of the model's OWN PAST FORECAST errors, trained jointly.
delta = W2 ReLU(W1 eps) + eta_k * sum_s a_s v_s / sigma, where the memory (set per batch in self.mem by the caller) holds, for past
origins s = t - d (d = H .. 8H step P, s + H <= t), the raw full-horizon error v_s (data units, H = Hmax * P) of the model's forecast at
s and the TEFL input eps_s it saw there. sigma = the current window's RevIN scale (delta lives in the normalised space).
  variant 'pos': a = softmax_s(b_d)                                  learned window over the look-back distance d (no content)
  variant 'qk' : a = softmax_s(b_d + Wq eps_t . Wk eps_s / (sqrt(r) T))   + content selection (low rank r, shared by variables)
eta = 0 at init (R1TeflAtt == R1Tefl at init). Values carry no gradient (memory bank); Wk gets gradient through the bank eps (stored
inputs), Wq through eps_t. mem_audit: None | 'off' (attention term removed) | 'perm' (values permuted against the demos)."""
import math
import torch
import torch.nn as nn
from patchmem.r1_tefl import R1Tefl


class R1TeflAtt(R1Tefl):
    def __init__(self, cfg, variant='qk', n_demo=50, r=8, **kw):
        super().__init__(cfg, **kw)
        assert variant in ('pos', 'qk'), variant
        self.variant, self.n_demo, self.r = variant, n_demo, r
        self.Wq, self.Wk = nn.Linear(self.din, r, bias=False), nn.Linear(self.din, r, bias=False)
        nn.init.normal_(self.Wq.weight, std=1 / math.sqrt(self.din)); nn.init.normal_(self.Wk.weight, std=1 / math.sqrt(self.din))
        self.pos = nn.Parameter(torch.zeros(n_demo))
        self.logt = nn.Parameter(torch.zeros(()))
        self.eta = nn.Parameter(torch.zeros(self.hmax))
        self.mem, self.mem_audit = None, None

    def extra_delta(self, delta, eps, stdev):
        M = self.mem
        if M is None or self.mem_audit == 'off':
            return delta
        b, c, hmax, P = delta.shape
        v = M['v'] / stdev.permute(0, 2, 1)[:, None]                                    # (b, n, C, H) data -> normalised units
        assert v.shape[1] == self.n_demo and v.shape[-1] == hmax * P, (v.shape, self.n_demo, hmax * P)
        s = self.pos[None, :, None].expand(b, -1, c)                                    # (b, n, C)
        if self.variant == 'qk':
            q, k = self.Wq(eps), self.Wk(M['eps'])                                      # (b, C, r), (b, n, C, r)
            s = s + (k * q[:, None]).sum(-1) / (math.sqrt(self.r) * torch.exp(self.logt))
        mask = M['mask'][:, :, None]                                                    # (b, n, 1) demo available
        a = torch.softmax(s.masked_fill(~mask, -1e9), 1)                                # finite fill: no NaN gradient
        a = a * mask.any(1, keepdim=True).to(a.dtype)                                    # no demo -> no term
        if self.mem_audit == 'perm':
            v = v[:, torch.randperm(v.shape[1], device=v.device)]
        att = (a[..., None] * v).sum(1).reshape(b, c, hmax, P)
        return delta + att * self.eta[None, None, :, None]
