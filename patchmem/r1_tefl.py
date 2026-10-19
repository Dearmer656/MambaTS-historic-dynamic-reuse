"""PAT-307 R1 + TEFL-style error feedback (R1Tefl): the same-information controls for R1Obs (O). Baseline = R1 (patchmem/r1_model.py,
unchanged backbone). TEFL (arXiv 2602.22520): y' = y + ReLU(eps W1) W2, eps = the error of the forecast made H steps ago on [t-H, t).
Only the correction module g is taken from TEFL (no spectral-flatness loss, no warm-up, same Adam / MSE / budget as every R1 arm).

The error eps covers the last Hc lookback patches [t-Hc*P, t) in both modes, RevIN-normalised with the CURRENT window's statistics,
ORIGINAL variable order, detached (sg), backbone predictions only (never corrected):
  mode 'native'   (T0) one-step errors e_s = x_s - yh_{s-1} of the teacher-forced history, s = T-Hc..T-1
                       (= R1Obs's last Hc slot values V[:, -Hc:])
  mode 'faithful' (T1) multi-step errors: R1 run on the context x[:, :L-Hc*P] alone (its own RevIN statistics, the same scan order),
                       rolled out Hc patches from its own predictions, compared with x on [t-Hc*P, t)
g, shared across variables: eps (C, Hc*P) -> W1 (Hc*P -> r) -> ReLU -> W2 (r -> Hmax*P), no biases, W2 = 0 at init (R1Tefl == R1).
Rollout patch k gets the k-th P-block of g(eps); the corrected forecast is fed back (feed=True, as R1Obs) or the backbone patch
(feed=False: backbone rollout identical to R1). Inference audits: None | 'zero' (delta = 0) | 'perm' (eps rolled by one sample across
the batch: sample pairing broken; needs batch >= 2)."""
import torch
import torch.nn as nn
from patchmem.r1_model import R1


class R1Tefl(R1):
    def __init__(self, cfg, readout_ln=False, mode='native', hc=2, rank=64, hmax=7, feed=True):
        super().__init__(cfg, readout_ln=readout_ln)
        assert mode in ('native', 'faithful'), mode
        self.mode, self.hc, self.hmax, self.feed = mode, hc, hmax, feed
        self.W1 = nn.Linear(hc * self.P, rank, bias=False)                             # default init (TEFL: low-rank adapter)
        self.W2 = nn.Linear(rank, hmax * self.P, bias=False)
        nn.init.zeros_(self.W2.weight)                                                  # zero: R1 at init
        self.audit, self.mem_stats = None, {}

    def set_audit(self, mode=None, seed=0):
        assert mode in (None, 'zero', 'perm'), mode
        self.audit = mode

    def stats(self):
        return {k: torch.stack(v).mean().item() for k, v in self.mem_stats.items() if v}

    def error(self, x, xn, xo, yh, perm, means, stdev, lens=None):
        """eps (b, C, Hc*P), detached, normalised with the current window's statistics, original variable order."""
        b, L, c = x.shape
        P, hc = self.P, self.hc
        T = L // P
        if self.mode == 'native':
            assert T - 1 >= hc, f'native TEFL needs T-1 >= Hc ({T}, {hc})'
            e = xo[:, T - hc:] - yh[:, T - hc - 1:T - 1]                                # (b, Hc, C, P)
            return e.permute(0, 2, 1, 3).reshape(b, c, hc * P).detach()
        assert L - hc * P >= P, f'faithful TEFL needs a context of at least one patch ({L}, {hc})'
        with torch.no_grad():
            pc = R1.forward(self, x[:, :L - hc * P], hc, perm=perm, lens=None if lens is None else lens - hc * P)['pred']   # ctx RevIN
            pn = (pc - means) / stdev                                                   # -> current window's normalisation
            return (xn[:, L - hc * P:] - pn).transpose(1, 2).contiguous()               # (b, C, Hc*P)

    def forward(self, x, n_out, perm=None, trace=None, ret_eps=False, lens=None):
        """Same interface and pipeline as R1.forward; the rollout forecasts get delta = g(eps)."""
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        assert n_out <= self.hmax, (n_out, self.hmax)
        T = L // P
        self.mem_stats = {}
        means, stdev, xn, t0, tm = self.revin(x, lens)
        if lens is not None:
            assert bool((lens.to(x.device) // P - 1 >= self.hc).all()) and bool((lens.to(x.device) - self.hc * P >= P).all()), lens
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xo = xp.permute(0, 2, 1, 3)                                                     # (b, T, C, P) original order
        xs = self.to_scan(xo, perm)
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)                                    # (b, T, C, P) original order
        eps = self.error(x, xn, xo, yh, perm, means, stdev, lens)
        if self.audit == 'perm':
            eps = eps.roll(1, 0)
        delta = self.W2(torch.relu(self.W1(eps))).reshape(b, c, self.hmax, P)          # (b, C, Hmax, P)
        if self.audit == 'zero':
            delta = torch.zeros_like(delta)
        self.mem_stats = dict(delta_rms=[delta[:, :, :n_out].square().mean().sqrt().detach()],
                              eps_rms=[eps.square().mean().sqrt().detach()])
        base = yh[:, -1]
        outs = []
        for k in range(n_out):
            if k:
                fb = outs[-1] if self.feed else base
                states = self.step_time(self.embed(self.to_scan(fb.unsqueeze(1), perm)).squeeze(1), states)
                base = self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1)
            outs.append(base + delta[:, :, k])
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        out = dict(pred=pred, hist=hist, perm=perm)
        if ret_eps:
            out['eps'] = eps
        return out
