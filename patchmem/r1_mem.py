"""PAT-304 R1 + state-level retrieval memory (R1Mem). Baseline = R1 (patchmem/r1_model.py, unchanged); the only difference is the
memory on the last layer's (readout layer's) SSM state. Lower layers, readout, VPT, normalisation and the rollout are R1's.

At every history patch boundary t (after the C tokens of time patch t) of the last layer:
  h-_t     prior state: the plain SSM recurrence, started from the previous boundary's h+_{t-1}
  dh_t     = StateMem(query sg h-_t, slots j < t)                        retrieval over strictly earlier slots
  h+_t     = h-_t + dh_t                                                 carried to patch t+1 and read: pred(t+1) = read(h+_t)
  e_t      = sg(x_t - read(h+_{t-1}))   (t >= 1)                         error of the prediction the model actually made for patch t
  C_t      = CorrNet(sg h+_t, e_t)                                       what is still unresolved after the memory
  slot t   = (key W_K sg h-_t, value C_t)                                C_t is ONLY stored, never added to the carried state
C_t is computed on the memory-corrected state, so a slot holds only the information the memory could not supply at that time; it
reaches later states only through retrieval (value not detached: the later losses train CorrNet through the memory). No cycle: dh_t
depends on C_j (j < t), which depends on dh_j, strictly backwards in time. Keys use h- because the query h-_t must exist before h+_t.
Rollout (future patches, no ground truth): after each time step h- = the step's last-layer state, h+ = h- + StateMem(h-, all history
slots), read(h+) and carry h+; no slot is written. At init CorrNet's output layer is zero, so C == 0 and R1Mem == R1 exactly.
corr_shuffle: control, residual rolled across the batch (C from another sample's error)."""
import torch
import torch.nn.functional as F
from patchmem.r1_model import R1
from patchmem.ref_scan import mamba_proj
from patchmem.tm_model import CorrNet, GatedCorrNet
from patchmem.a0corr import StateMem


class R1Mem(R1):
    def __init__(self, cfg, readout_ln=False, corr='gated', mem_group=4, corr_shuffle=False):
        super().__init__(cfg, readout_ln=readout_ln)
        res_dim = self.c * self.P
        self.corr_net = GatedCorrNet(self.D, self.N, res_dim) if corr == 'gated' else CorrNet(corr, self.D, self.N, res_dim)
        self.mem_net = StateMem(self.D, self.N, group=mem_group)
        self.corr_shuffle = corr_shuffle
        self.bank = None                                                                # (keys, vals) of the current forward
        self.mem_stats = {}

    def _retrieve(self, h_minus, keys, vals):
        inj, a, g = self.mem_net(h_minus, torch.stack(keys, 1), torch.stack(vals, 1))
        st = self.mem_stats
        st.setdefault('g', []).append(g.mean().detach())
        st.setdefault('ratio', []).append((inj.flatten(1).norm(dim=1) / h_minus.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().detach())
        if a.shape[-1] >= 2:
            ent = -(a * a.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(a.shape[-1])))
            st.setdefault('ent_norm', []).append(ent.mean().detach())
        return h_minus + inj

    def _last_full_mem(self, m, hn, xs):
        """Last layer over the history with the memory at every boundary. xs (b, T, C, P) normalised patches in scan order.
        Returns boundary states h+ (b, T, D, N) and the final h+; leaves the slot bank in self.bank for the rollout."""
        x, dt, A, B, C, D, z, bias = mamba_proj(m, hn)
        dt = F.softplus(dt + bias[None, :, None])
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])
        dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        h = x.new_zeros(x.shape[0], self.D, self.N)
        caps, keys, vals, prev = [], [], [], None
        for s in range(x.shape[-1]):
            h = dA[:, :, s] * h + dBu[:, :, s]
            if (s + 1) % self.c:
                continue
            t = (s + 1) // self.c - 1
            h_minus = h
            if vals:                                                                    # slots j < t only
                h = self._retrieve(h_minus, keys, vals)
            if t >= 1:
                e = (xs[:, t] - self.read(prev)).flatten(1).detach()                    # error of the prediction made at t-1
                if self.corr_shuffle:
                    e = e.roll(1, 0)
                Cm = self.corr_net(h.detach(), e)
                self.mem_stats.setdefault('C', []).append(Cm.square().mean().detach())
                keys.append(self.mem_net.key(h_minus)); vals.append(Cm)
            caps.append(h); prev = h
        self.bank = (keys, vals)
        return torch.stack(caps, 1), h

    def history(self, tok, xs):
        layers = self.enc.layers
        hidden, residual, states = tok, None, []
        for i, layer in enumerate(layers):
            hn, residual = self._norm(layer, hidden, residual)
            if i < len(layers) - 1:
                hidden, last = self._mixer_full(layer.mixer, hn)
                states.append(last)
            else:
                bstates, last = self._last_full_mem(layer.mixer, hn, xs)
                states.append(last)
        return bstates, states

    def forward(self, x, n_out, perm=None, trace=None):
        """Same interface and pipeline as R1.forward; differs only by the memory in history() and in the rollout."""
        b, L, c = x.shape
        P = self.P
        assert L % P == 0 and c == self.c, (L, P, c, self.c)
        T = L // P
        self.mem_stats = {}
        means = x.mean(1, keepdim=True).detach()
        xc = x - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        xn = xc / stdev
        perm = self.scan_order(b, x.device) if perm is None else perm
        xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
        xs = self.to_scan(xp.permute(0, 2, 1, 3), perm)                                 # (b, T, C, P) scan order
        tok = self.embed(xs).reshape(b, T * c, -1)
        bstates, states = self.history(tok, xs)
        yh = self.to_orig(self.read(bstates), perm)
        outs = [yh[:, -1]]
        keys, vals = self.bank
        for _ in range(n_out - 1):
            states = self.step_time(self.embed(self.to_scan(outs[-1].unsqueeze(1), perm)).squeeze(1), states)
            if vals:
                states[-1] = self._retrieve(states[-1], keys, vals)
            outs.append(self.to_orig(self.read(states[-1]).unsqueeze(1), perm).squeeze(1))
        self.bank = None
        if trace is not None:
            trace.update(bstates=bstates, h_last=bstates[:, -1])
        den = lambda y: y * stdev.unsqueeze(1) + means.unsqueeze(1)
        pred = den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)
        hist = den(yh[:, :-1].transpose(-1, -2))
        return dict(pred=pred, hist=hist, perm=perm)

    def stats(self):
        return {k: torch.stack(v).mean().item() for k, v in self.mem_stats.items() if v}
