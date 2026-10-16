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
reaches later states only through retrieval. No cycle: dh_t depends on C_j (j < t), which depends on dh_j, strictly backwards in
time. Keys use h- because the query h-_t must exist before h+_t. The last history slot C_{T-1} is first used by the second future
patch: the first future patch is read(h+_{T-1}), retrieved over j < T-1 (a designed one-step delay, not a bug).
Rollout (future patches, no ground truth): after each time step h- = the step's last-layer state, h+ = h- + StateMem(h-, all history
slots), read(h+) and carry h+; no slot is written. At init CorrNet's output layer is zero, so C == 0 and R1Mem == R1 exactly.

Training variants (comment e18ad84b):
  roll_retrieve=False B2-H: retrieval at the history boundaries only; the rollout is R1's plain recurrence from h+_{T-1}
  slot_keep=p < 1     B2-drop: during training each sample writes each history slot with probability p (per-sample slot dropout;
                      the dropped slot is invisible to every later retrieval); evaluation uses all slots
  value_detach=False  Hybrid (B2): memory values keep their graph, the forecast losses train CorrNet through later retrievals
  value_detach=True   Pure-B (B1): values detached when written, CorrNet is trained by the writer loss only
  retrieve=False      B3: CorrNet / writer loss as usual but no retrieval injection (h+ = h-), the forecast equals R1's
corr_shuffle: control, residual rolled across the batch (C from another sample's error).
Writer loss (corr_aux, training only): L_corr = mean_{t=1..T-2} ||R_sg(sg h+_t + C_t) - x_{t+1}||^2 with the SAME readout R under
stop-gradient weights, so it trains CorrNet only; h+_t + C_t is a temporary state for this loss, never read for the forecast nor
carried. x_{t+1} is inside the lookback: in-window auxiliary supervision (whole-lookback RevIN), like L_hist; t = T-1 is excluded.
Inference audits (same checkpoint, eval only): audit = None | 'zero' (gate 0) | 'uniform' (a = 1/M) | 'perm' (each retrieval reads the
values of its M slots under a random permutation from audit_gen: same keys, same values and slot count, key-value pairing broken;
identity when M = 1).
diag=True: the next forwards also record detailed writer / retrieval statistics in diag_rec (floats; slower, for logging batches)."""
import math
import torch
import torch.nn.functional as F
from patchmem.r1_model import R1
from patchmem.ref_scan import mamba_proj
from patchmem.tm_model import CorrNet, GatedCorrNet
from patchmem.a0corr import StateMem


def _q(x, qs=(0.1, 0.5, 0.9)):
    x = x.detach().float().flatten()
    if x.numel() > 200_000:                                                             # quantile cost cap: fixed-stride subsample
        x = x[:: x.numel() // 200_000 + 1]
    return [v.item() for v in torch.quantile(x, torch.tensor(qs, device=x.device))]


class R1Mem(R1):
    def __init__(self, cfg, readout_ln=False, corr='gated', mem_group=4, corr_shuffle=False, value_detach=False, retrieve=True,
                 roll_retrieve=True, slot_keep=1.0):
        super().__init__(cfg, readout_ln=readout_ln)
        res_dim = self.c * self.P
        self.corr_net = GatedCorrNet(self.D, self.N, res_dim) if corr == 'gated' else CorrNet(corr, self.D, self.N, res_dim)
        self.mem_net = StateMem(self.D, self.N, group=mem_group)
        self.corr_shuffle = corr_shuffle
        self.value_detach = value_detach
        self.retrieve = retrieve
        self.roll_retrieve = roll_retrieve
        self.slot_keep = slot_keep
        self.corr_aux = False                                                           # compute the writer loss (set by the trainer)
        self.corr_loss = None                                                           # L_corr of the last training forward
        self.audit, self.audit_gen = None, None
        self.diag, self.diag_rec = False, {}
        self.bank = None                                                                # (keys, vals, slot times, slot masks) of the current forward
        self.mem_stats = {}

    def set_audit(self, mode=None, seed=0):
        assert mode in (None, 'zero', 'uniform', 'perm'), mode
        self.audit = mode
        self.mem_net.audit = mode if mode in ('zero', 'uniform') else None
        self.audit_gen = torch.Generator().manual_seed(seed) if mode == 'perm' else None

    def read_sg(self, h):
        """read(h) with stop-gradient readout parameters: the writer loss reaches CorrNet only."""
        f = h.flatten(-2)
        if self.ro_ln is not None:
            f = F.layer_norm(f, f.shape[-1:], self.ro_ln.weight.detach(), self.ro_ln.bias.detach(), self.ro_ln.eps)
        return F.linear(f, self.readout.weight.detach(), self.readout.bias.detach()).unflatten(-1, (self.c, self.P))

    def _rec(self, key, val):
        self.diag_rec.setdefault(key, []).append(float(val))

    def _retrieve(self, h_minus, keys, vals, times=None, t_now=None, phase='hist', masks=None):
        K, V = torch.stack(keys, 1), torch.stack(vals, 1)
        M = V.shape[1]
        if self.audit == 'perm' and M >= 2:
            V = V[:, torch.randperm(M, generator=self.audit_gen).to(V.device)]
        inj, a, g = self.mem_net(h_minus, K, V, None if masks is None else torch.stack(masks, 1))
        st = self.mem_stats
        st.setdefault('g', []).append(g.mean().detach())
        st.setdefault('ratio', []).append((inj.flatten(1).norm(dim=1) / h_minus.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().detach())
        if M >= 2:
            ent = -(a * a.clamp_min(1e-12).log()).sum(-1) / math.log(M)
            st.setdefault('ent_norm', []).append(ent.mean().detach())
        h_plus = h_minus + inj
        if self.diag:
            self._diag_retrieval(phase, h_minus, h_plus, inj, a, g, V, M, times, t_now)
        return h_plus

    @torch.no_grad()
    def _diag_retrieval(self, phase, h_minus, h_plus, inj, a, g, V, M, times, t_now):
        p, r = phase + '_', self._rec
        hm, hp = h_minus.flatten(1).norm(dim=1), h_plus.flatten(1).norm(dim=1)
        dn = inj.flatten(1).norm(dim=1)
        r(p + 'slots', M)
        r(p + 'h_minus_rms', (h_minus.flatten(1).square().mean(1).sqrt()).mean())
        r(p + 'h_plus_rms', (h_plus.flatten(1).square().mean(1).sqrt()).mean())
        r(p + 'delta_over_hminus', (dn / hm.clamp_min(1e-12)).mean())                  # per sample ||delta||_F / ||h-||_F, batch mean
        r(p + 'delta_over_Cslot', (dn / V.flatten(2).norm(dim=2).mean(1).clamp_min(1e-12)).mean())   # / mean_j ||C_j||_F
        gq = _q(g)
        r(p + 'gate_mean', g.mean()); r(p + 'gate_p10', gq[0]); r(p + 'gate_p50', gq[1]); r(p + 'gate_p90', gq[2])
        r(p + 'gate_frac_lt_0.05', (g < 0.05).float().mean()); r(p + 'gate_frac_gt_0.95', (g > 0.95).float().mean())
        mx = a.max(-1)
        r(p + 'attn_max', mx.values.mean())
        if M >= 2:                                                                       # entropy undefined (N/A) for M = 1: not recorded
            ent = -(a * a.clamp_min(1e-12).log()).sum(-1)                                 # (b, nh)
            r(p + 'attn_entropy', ent.mean()); r(p + 'attn_entropy_norm', (ent / math.log(M)).mean())
            r(p + 'attn_eff_slots', ent.exp().mean())
            r(p + 'unif_heads', (ent / math.log(M) > 0.95).float().mean())
        if times is not None:
            age = (t_now - torch.as_tensor(times, device=a.device, dtype=a.dtype))         # (M,) slot age in patches
            top1 = age[mx.indices]                                                       # (b, nh)
            r(p + 'top1_age', top1.mean()); r(p + 'mean_age', (a * age).sum(-1).mean())
            r(p + 'top1_age_head_std', top1.std(1, unbiased=False).mean())
        mn = self.mem_net
        hq = h_minus.detach().flatten(1)
        r(p + 'q_norm', mn.Wq(hq).norm(dim=1).mean()); r(p + 'k_norm', mn.Wk(hq).norm(dim=1).mean())
        r(p + 'tau_mean', mn.log_tau.exp().mean())

    @torch.no_grad()
    def _diag_writer(self, h_plus, Cm, e, target):
        r = self._rec
        Cf = Cm.flatten(1)
        cn = Cf.norm(dim=1)
        r('C_fro', cn.mean()); r('C_rms', Cf.square().mean().sqrt())
        r('C_over_hplus', (cn / h_plus.flatten(1).norm(dim=1).clamp_min(1e-12)).mean())
        cq = _q(Cf.abs(), (0.5, 0.9, 0.99))
        r('C_abs_p50', cq[0]); r('C_abs_p90', cq[1]); r('C_abs_p99', cq[2])
        r('C_frac_near0', (Cf.abs() < 1e-4).float().mean())                              # |C| < 1e-4 (normalised state units)
        r('e_norm', e.norm(dim=1).mean()); r('e_rms', e.square().mean().sqrt())
        if target is not None:                                                           # writer's immediate (oracle, labelled) effect
            lb = (self.read_sg(h_plus) - target).square().flatten(1).mean(1)
            la = (self.read_sg(h_plus + Cm) - target).square().flatten(1).mean(1)
            r('writer_L_before', lb.mean()); r('writer_L_after', la.mean()); r('writer_dL', (lb - la).mean())
            r('writer_frac_improved', (la < lb).float().mean())

    def _last_full_mem(self, m, hn, xs):
        """Last layer over the history with the memory at every boundary. xs (b, T, C, P) normalised patches in scan order.
        Returns boundary states h+ (b, T, D, N) and the final h+; leaves the slot bank in self.bank for the rollout."""
        x, dt, A, B, C, D, z, bias = mamba_proj(m, hn)
        dt = F.softplus(dt + bias[None, :, None])
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])
        dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        h = x.new_zeros(x.shape[0], self.D, self.N)
        caps, keys, vals, times, prev, aux = [], [], [], [], None, []
        drop = self.training and self.slot_keep < 1
        masks = [] if drop else None
        T = xs.shape[1]
        for s in range(x.shape[-1]):
            h = dA[:, :, s] * h + dBu[:, :, s]
            if (s + 1) % self.c:
                continue
            t = (s + 1) // self.c - 1
            h_minus = h
            if vals and self.retrieve:                                                  # slots j < t only
                h = self._retrieve(h_minus, keys, vals, times, t, masks=masks)
            if t >= 1:
                e = (xs[:, t] - self.read(prev)).flatten(1).detach()                    # error of the prediction made at t-1
                if self.corr_shuffle:
                    e = e.roll(1, 0)
                Cm = self.corr_net(h.detach(), e)
                self.mem_stats.setdefault('C', []).append(Cm.square().mean().detach())
                keys.append(self.mem_net.key(h_minus)); vals.append(Cm.detach() if self.value_detach else Cm); times.append(t)
                if drop:
                    masks.append(torch.rand(h.shape[0], device=h.device) < self.slot_keep)
                if self.corr_aux and self.training and t <= T - 2:                     # writer loss, temporary h+ + C
                    aux.append((self.read_sg(h.detach() + Cm) - xs[:, t + 1]).square().mean())
                if self.diag:
                    self._diag_writer(h.detach(), Cm, e, xs[:, t + 1] if t <= T - 2 else None)
            caps.append(h); prev = h
        self.corr_loss = torch.stack(aux).mean() if aux else None
        self.bank = (keys, vals, times, masks)
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
        keys, vals, times, masks = self.bank
        for k in range(n_out - 1):
            states = self.step_time(self.embed(self.to_scan(outs[-1].unsqueeze(1), perm)).squeeze(1), states)
            if vals and self.retrieve and self.roll_retrieve:
                states[-1] = self._retrieve(states[-1], keys, vals, times, T + k, phase='roll', masks=masks)
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

    def pop_diag(self):
        """Mean of every recorded diagnostic since the last pop; clears the record."""
        out = {k: sum(v) / len(v) for k, v in self.diag_rec.items() if v}
        self.diag_rec = {}
        return out
