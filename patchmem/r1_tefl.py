"""PAT-307 R1 + TEFL-style error feedback (R1Tefl): the same-information controls for R1Obs (O). Baseline = R1 (patchmem/r1_model.py,
unchanged backbone). TEFL (arXiv 2602.22520): y' = y + ReLU(eps W1) W2, eps = the error of the forecast made H steps ago on [t-H, t).
Only the correction module g is taken from TEFL (no spectral-flatness loss, no warm-up, same Adam / MSE / budget as every R1 arm).

The error eps covers the last Hc lookback patches [t-Hc*P, t), RevIN-normalised with the CURRENT window's statistics, ORIGINAL
variable order, detached (sg), backbone predictions only (never corrected):
  mode 'native'   (T0) one-step errors e_s = x_s - yh_{s-1} of the teacher-forced history, s = T-Hc..T-1
                       (= R1Obs's last Hc slot values V[:, -Hc:]). hc=0: ALL T-1 lookback errors (= every R1Obs slot value), right-
                       aligned (most recent last) and zero-padded on the left to Tmax-1 patches (Tmax = R1_LMAX / P); with lens the
                       padded samples' invalid slots (t < t0 + 1) are zero too. W1 has no bias, so zero entries contribute nothing.
  mode 'faithful' (T1) multi-step errors: R1 run on the context x[:, :L-Hc*P] alone (its own RevIN statistics, the same scan order),
                       rolled out Hc patches from its own predictions, compared with x on [t-Hc*P, t)
g, shared across variables: eps (C, Din) -> W1 (Din -> r) -> ReLU -> W2 (r -> Hmax*P), no biases, W2 = 0 at init (R1Tefl == R1).
Rollout patch k gets the k-th P-block of g(eps); beyond the trained horizon (k >= Hmax) extrap='last' repeats the last block, 'zero'
adds nothing. The corrected forecast is fed back (feed=True, as R1Obs) or the backbone patch (feed=False: backbone rollout identical
to R1). Inference audits: None | 'zero' (delta = 0) | 'perm' (eps of a DISTANT window passed as eps_override: sample pairing broken;
stride-1 batch neighbours are near-identical windows, so a within-batch roll would not break it)."""
import torch
import torch.nn as nn
from patchmem.r1_model import R1


class R1Tefl(R1):
    def __init__(self, cfg, readout_ln=False, mode='native', hc=2, rank=64, hmax=7, feed=True, tmax=30, extrap='last'):
        super().__init__(cfg, readout_ln=readout_ln)
        assert mode in ('native', 'faithful') and extrap in ('last', 'zero'), (mode, extrap)
        assert hc > 0 or mode == 'native', 'hc=0 (all lookback errors) is native only'
        self.mode, self.hc, self.hmax, self.feed, self.tmax, self.extrap = mode, hc, hmax, feed, tmax, extrap
        self.din = (hc if hc else tmax - 1) * self.P
        self.W1 = nn.Linear(self.din, rank, bias=False)                                 # default init (TEFL: low-rank adapter)
        self.W2 = nn.Linear(rank, hmax * self.P, bias=False)
        nn.init.zeros_(self.W2.weight)                                                  # zero: R1 at init
        self.audit, self.mem_stats = None, {}

    def set_audit(self, mode=None, seed=0):
        assert mode in (None, 'zero', 'perm'), mode
        self.audit = mode

    def stats(self):
        return {k: torch.stack(v).mean().item() for k, v in self.mem_stats.items() if v}

    def error(self, x, xn, xo, yh, perm, means, stdev, lens=None):
        """eps (b, C, Din), detached, normalised with the current window's statistics, original variable order."""
        b, L, c = x.shape
        P, hc = self.P, self.hc
        T = L // P
        if self.mode == 'native' and hc == 0:
            assert T <= self.tmax, (T, self.tmax)
            e = xo[:, 1:] - yh[:, :-1]                                                  # (b, T-1, C, P) slots t = 1..T-1
            if lens is not None:                                                        # slot t valid iff t >= t0 + 1
                t0 = T - lens.to(x.device) // P
                e = e * (torch.arange(1, T, device=x.device)[None, :] >= (t0 + 1)[:, None]).to(e.dtype)[:, :, None, None]
            e = torch.cat([e.new_zeros(b, self.tmax - T, c, P), e], 1)                  # right-aligned, zero-padded to Tmax-1
            return e.permute(0, 2, 1, 3).reshape(b, c, self.din).detach()
        if self.mode == 'native':
            assert T - 1 >= hc, f'native TEFL needs T-1 >= Hc ({T}, {hc})'
            e = xo[:, T - hc:] - yh[:, T - hc - 1:T - 1]                                # (b, Hc, C, P)
            return e.permute(0, 2, 1, 3).reshape(b, c, hc * P).detach()
        assert L - hc * P >= P, f'faithful TEFL needs a context of at least one patch ({L}, {hc})'
        with torch.no_grad():
            pc = R1.forward(self, x[:, :L - hc * P], hc, perm=perm, lens=None if lens is None else lens - hc * P)['pred']   # ctx RevIN
            pn = (pc - means) / stdev                                                   # -> current window's normalisation
            return (xn[:, L - hc * P:] - pn).transpose(1, 2).contiguous()               # (b, C, Hc*P)

    def forward(self, x, n_out, perm=None, trace=None, ret_eps=False, lens=None, eps_override=None):
        """Same interface and pipeline as R1.forward; the rollout forecasts get delta = g(eps). eps_override (b, C, Din): use these errors
        instead of the window's own (the 'perm' audit)."""
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        self.mem_stats = {}
        means, stdev, xn, t0, tm = self.revin(x, lens)
        if lens is not None:
            assert bool((lens.to(x.device) // P - 1 >= self.hc).all()) and bool((lens.to(x.device) - self.hc * P >= P).all()), lens   # hc=0: trivially
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xo = xp.permute(0, 2, 1, 3)                                                     # (b, T, C, P) original order
        xs = self.to_scan(xo, perm)
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, tm)
        yh = self.to_orig(self.read(bstates), perm)                                    # (b, T, C, P) original order
        eps = self.error(x, xn, xo, yh, perm, means, stdev, lens)
        if self.audit == 'perm':
            assert eps_override is not None and eps_override.shape == eps.shape, 'perm audit needs eps_override from distant windows'
        if eps_override is not None:
            eps = eps_override.detach()
        delta = self.W2(torch.relu(self.W1(eps))).reshape(b, c, self.hmax, P)          # (b, C, Hmax, P)
        if n_out > self.hmax:                                                           # beyond the trained horizon
            ext = delta[:, :, -1:] if self.extrap == 'last' else torch.zeros_like(delta[:, :, -1:])
            delta = torch.cat([delta, ext.expand(b, c, n_out - self.hmax, P)], 2)
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
