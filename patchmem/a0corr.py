"""PAT-304 A0-frozen innovation correction. The official MambaTS checkpoint (A0) is loaded and frozen (eval mode: no dropout, one fixed
variable order). Only its last Mamba layer is re-scanned token by token, along A0's own scan order (variable-major, VPT order). Scan
token s = patch t of variable k. After the scan absorbs token s (t >= 1), the innovation e_s = x_s - aux(o_{s-1}) (x_s: the normalised
patch, o_{s-1}: the official last-layer token output of the previous patch of the same variable, aux: a frozen next-patch predictor
fitted on A0) drives a state correction h_s+ = h_s- + CorrNet(stopgrad h_s-, e_s); the scan continues from h_s+. The official head reads
the corrected token outputs. With CorrNet == 0 this is exactly A0."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from patchmem.ref_scan import mamba_proj
from patchmem.tm_model import CorrNet, GatedCorrNet, channel_decor


class StateMem(nn.Module):
    """PAT-304 state-level retrieval memory, a module separate from CorrNet (the writer). Multi-head attention over the history slots:
    q = W_Q flat(sg h_t-), k_j = W_K flat(sg h_j-), with FULL-state projections W_Q, W_K: D*N -> D*N (every q/k coordinate sees every
    channel), then q, k are split into nh = D / group heads of group*N dims (group=4: 64 dims; ETTh2 8 heads, Weather 16). Head h does
    its own softmax a_h = softmax(q_h . k_j,h / (sqrt(group*N) tau_h)) and weights only its own channel group of the values:
    m[group h channels] = sum_j a_h,j C_j[group h channels], so different channel groups can take corrections from different past sites.
    group=None: one head, a single softmax shared by all channels. value = C_j NOT detached (a later site's loss reaches the historical
    C_j and CorrNet). Injected through a per-channel scalar gate g = sigmoid(W_g flat(sg h_t-) + b_g) (b, D, 1), b_g = -4, W_g = 0 at init.
    No learned memory->state map W this round. audit: None | 'zero' (g = 0) | 'uniform' (a = 1/M) -- evaluation-only ablations.
    qk_dim: per-head q/k dimension, decoupled from the value channel grouping (default group*N, i.e. W_Q, W_K: D*N -> D*N; qk_dim = r
    gives W_Q, W_K: D*N -> nh*r, a low-rank score with the same heads and the same value groups).
    uniform=True: trained-without-selection control: no W_Q, W_K, tau; a = 1/(number of available slots) in every head (the slot mask,
    if any, removes dropped slots); the gate W_g and the per-head channel-group value aggregation are unchanged."""
    def __init__(self, d_inner, d_state, group=None, tau0=1.0, gate_bias=-4.0, qk_dim=None, uniform=False):
        super().__init__()
        group = group or d_inner
        assert d_inner % group == 0, f'd_inner {d_inner} not divisible by group {group}'
        self.group, self.nh, self.dh = group, d_inner // group, qk_dim or group * d_state   # dh: per-head q/k dim
        self.uniform = uniform
        if not uniform:
            self.Wq = nn.Linear(d_inner * d_state, self.nh * self.dh, bias=False)
            self.Wk = nn.Linear(d_inner * d_state, self.nh * self.dh, bias=False)
            self.log_tau = nn.Parameter(torch.full((self.nh,), float(tau0)).log())
        self.Wg = nn.Linear(d_inner * d_state, d_inner)
        nn.init.zeros_(self.Wg.weight); nn.init.constant_(self.Wg.bias, gate_bias)
        self.audit = None

    def key(self, h):
        if self.uniform:
            return h.new_zeros(h.shape[0], self.nh, 0)                                  # no key: selection-free
        return self.Wk(h.detach().flatten(1)).reshape(h.shape[0], self.nh, self.dh)   # (b, nh, dh)

    def forward(self, h, keys, vals, mask=None):
        """h (b, D, N) current pre-correction state; keys (b, M, nh, dh); vals (b, M, D, N); mask (b, M) bool or None: per-sample
        slots that exist (False = slot dropped; a sample with no slot gets a = 0, i.e. no injection).
        Returns the injection g*m (b, D, N), a (b, nh, M), g (b, D, 1)."""
        b, M = keys.shape[:2]
        hq = h.detach().flatten(1)
        if self.uniform:
            if mask is None:
                a = keys.new_full((b, self.nh, M), 1.0 / M)
            else:                                                                       # all-dropped rows -> a = 0 (no injection)
                w = mask.to(keys.dtype)
                a = (w / w.sum(1, keepdim=True).clamp_min(1.0))[:, None, :].expand(b, self.nh, M)
        elif self.audit == 'uniform':
            a = keys.new_full((b, self.nh, M), 1.0 / M)
        else:
            q = self.Wq(hq).reshape(b, self.nh, self.dh)
            logit = torch.einsum('bhd,bmhd->bhm', q, keys) / (self.dh ** 0.5 * self.log_tau.exp())[None, :, None]
            if mask is not None:
                anyv = mask.any(1)
                mk = mask | ~anyv[:, None]                                              # all-dropped rows: dummy full mask, zeroed below
                a = torch.softmax(logit.masked_fill(~mk[:, None, :], float('-inf')), -1) * anyv[:, None, None]
            else:
                a = torch.softmax(logit, -1)
        v = vals.reshape(b, M, self.nh, self.group, vals.shape[-1])                     # each head's own channel group of C_j
        m = torch.einsum('bhm,bmhgn->bhgn', a, v).reshape(b, -1, vals.shape[-1])
        g = torch.sigmoid(self.Wg(hq)).unsqueeze(-1)
        if self.audit == 'zero':
            g = torch.zeros_like(g)
        return g * m, a, g


class A0Corr(nn.Module):
    def __init__(self, base, order, corr='shared', shuffle=False, mode='self', freeze=True, step_sg=False, mem=None, mem_group=None):
        super().__init__()
        assert mem in (None, 'none', 'real', 'shuf')
        self.mem = mem                                                                 # cross mode. None: legacy (no step_mem term). 'none': M0, step_mem
        self.cut_value = self.cut_carry = False                                       # on h- (m = 0); 'real': M1; 'shuf': M2, a deterministic time
                                                                                       # permutation (values rolled by one slot along time within the
                                                                                       # sequence: same values, wrong pairing -- NOT an unrelated-memory
                                                                                       # null; that is run_cross(mem_bank=donor_bank(other window))).
                                                                                       # Test-only gradient surgeries:
                                                                                       # cut_value detaches memory values (no retrieval credit to C_j),
                                                                                       # cut_carry adds sg(C) to the carried state (no recurrence credit)
        self.step_sg = step_sg                                                         # cross mode: per-step loss on sg(h-) + C (no backbone gradient)
        self.freeze = freeze                                                          # False: backbone + ro train from scratch alongside corr_net
        self.base = base.eval() if freeze else base
        for p in self.base.parameters():
            p.requires_grad_(not freeze)
        self.patch_len, self.pred_len = base.patch_len, base.pred_len
        self.n_vars = base.encoder.n_vars
        self.n_patch = base.seq_len // base.patch_len
        self.register_buffer('order', torch.as_tensor(order, dtype=torch.long))       # scan position k -> variable index
        d_model = base.value_embedding[1].out_features
        m = base.encoder.layers[-1].mixer
        assert not m.use_casual_conv
        self.aux = nn.Linear(d_model, self.patch_len)                                  # next-patch predictor, fitted then frozen
        self.aux.requires_grad_(False)
        self.shuffle = shuffle
        assert mode in ('self', 'cross')
        self.mode = mode                                                               # self: F1/F2 (own innovation); cross: F3 (run_cross)
        self.ro = nn.Linear(m.d_inner * m.d_state, self.patch_len)                     # state readout h_{k,t} -> x_{k,t+1}; ridge-fitted then
        self.ro.requires_grad_(not freeze)                                             # frozen when freeze, else trained from scratch like any other layer
        res_dim = self.patch_len * (self.n_vars if mode == 'cross' else 1)
        if corr == 'gated':                                                            # C = gate(h, z) * (z @ Wc), C(h, e=0) = 0
            self.corr_net = GatedCorrNet(m.d_inner, m.d_state, res_dim)
        else:
            self.corr_net = CorrNet(corr, m.d_inner, m.d_state, res_dim) if corr else None
        self.mem_net = StateMem(m.d_inner, m.d_state, group=mem_group) if mem in ('real', 'shuf') else None
        self.use_corr = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.base.eval()                                                           # the backbone never leaves eval mode
        return self

    def norm(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        xc = x_enc - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return xc / stdev, means, stdev

    def lower(self, xn):
        """Embedding + all layers but the last, official computation in scan order. Returns the last layer's input residual stream
        res (b, c*l, d), the scan-ordered patches xp (b, c, l, P) and the official A0 forward pieces."""
        b = xn.shape[0]
        xp = xn.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)            # b c l P (stride == patch_len)
        xp = xp[:, self.order]
        e = self.base.value_embedding(xp)                                              # (b c) l d
        hid = rearrange(e, '(b c) l d -> b (c l) d', b=b)
        resid = None
        for blk in self.base.encoder.layers[:-1]:
            hid, resid = blk(hid, resid)
        return hid + resid, xp

    def tokens_out(self, out, res):
        nrm = self.base.encoder.norm_f
        return F.layer_norm(out + res, (out.shape[-1],), nrm.weight, nrm.bias, nrm.eps)

    def head(self, tok, means, stdev):
        """tok (b, c*l, d) in scan order -> official head on the original variable order, de-normalised (b, H, c)."""
        tok = rearrange(tok, 'b (c l) d -> b c (l d)', c=self.n_vars)
        tok = tok[:, torch.argsort(self.order)]
        dec = self.base.head(rearrange(tok, 'b c (l d) -> b (c l) d', l=self.n_patch), n_vars=self.n_vars)
        return dec * stdev[:, [0], :] + means[:, [0], :]

    def forward(self, x_enc, *_):                                                     # official vali/test signature
        return (self.run_cross if self.mode == 'cross' else self.run)(x_enc)['pred']

    def reset_ids_shuffle(self):                                                        # run.py calls it before test; the order is fixed
        pass

    def run(self, x_enc, record=None):
        """Returns dict(pred (b,H,c) de-normalised, reg = mean_s mean(C_s^2), step = mean_s mean((x_{s+1} - aux(o_s+))^2) or None).
        o_s+ is the token output read from the corrected state h_s+ (== o_s where no correction is applied); the next innovation is measured
        against this posterior prediction, e_{s+1} = x_{s+1} - aux(o_s+). The official head still reads o_s (prior). record receives (s, e_s)."""
        xn, means, stdev = self.norm(x_enc)
        res, xp = self.lower(xn)
        blk = self.base.encoder.layers[-1]
        m = blk.mixer
        xin = blk.norm(res)
        if self.corr_net is None or not self.use_corr:
            return dict(pred=self.head(self.tokens_out(m(xin), res), means, stdev), reg=None, step=None)
        x, dt, A, B, C, D, z, bias = mamba_proj(m, xin)
        dt = F.softplus(dt + bias[None, :, None])
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])                          # b d L n
        dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        b, L, l = x.shape[0], x.shape[-1], self.n_patch
        xpf = xp.reshape(b, L, self.patch_len)                                          # scan-ordered patches, token s = k*l + t
        h = x.new_zeros(b, x.shape[1], A.shape[1])
        ys, regs, steps, xhat = [], [], [], None
        read = lambda hh, s: (torch.einsum('bdn,bn->bd', hh, C[:, :, s]) + D * x[:, :, s]) * F.silu(z[:, :, s])
        for s in range(L):
            h = dA[:, :, s] * h + dBu[:, :, s]
            y = read(h, s)
            ys.append(y)
            t = s % l
            corrected = t >= 1 and s < L - 1                                            # nothing follows the last token
            if corrected:
                e = xpf[:, s] - xhat
                if record is not None:                                                  # the true innovation (before any shuffle control)
                    record.append((s, e))
                if self.shuffle:
                    e = e.roll(1, 0)
                Cm = self.corr_net(h.detach(), e)
                regs.append(Cm.square().mean())
                h = h + Cm
            if t < l - 1:                                                               # predict the next patch of the same variable from h_s+
                pnext = self.aux(self.tokens_out(m.out_proj(read(h, s) if corrected else y), res[:, s]))
                if corrected:
                    steps.append((xpf[:, s + 1] - pnext).square().mean())
                xhat = pnext.detach()
        tok = self.tokens_out(m.out_proj(torch.stack(ys, 1)), res)
        return dict(pred=self.head(tok, means, stdev), reg=torch.stack(regs).mean(), step=torch.stack(steps).mean())

    @torch.no_grad()
    def donor_bank(self, x_donor):
        """Memory slots (key, C) of every corrected site, scan order, written while scanning ANOTHER window: run_cross(x, mem_bank=...) then
        reads this unrelated history instead of its own (same variable, same t' < t). Evaluation-only null control."""
        tr = {}
        self.run_cross(x_donor, trace=tr)
        return tr['slots']

    def run_cross(self, x_enc, record=None, trace=None, mem_bank=None):
        """F3. State-orthogonal innovations of the uncorrected A0: e*_{k,t} = x_{k,t} - ro(h0_{k,t-1}) (t >= 1; 0 at t = 0). At scan token (k, t),
        t <= l-2, after the scan absorbs it: input u = [e*_{k,t}, e*_{j,t+1} for scan positions j < k (zeros for j >= k)] (c*P), all already
        scanned; h+ = h- + CorrNet(stopgrad h-, u); per-step loss (x_{k,t+1} - ro(h+))^2. The official head reads the prior token outputs.
        Returns dict(pred, reg, step, step0 = the same per-step MSE of the uncorrected A0). record receives (s, u).
        trace (dict) receives per-sample next-patch MSE (b, c, l-1): e0 = uncorrected A0, pri = ro(h-) before the site's own correction (carry-over of earlier
        corrections only), post = ro(h+); also dA, h0 (b c l n), xp and the corrections C (one (b, d, N) per corrected site, scan order)."""
        xn, means, stdev = self.norm(x_enc)
        res, xp = self.lower(xn)
        blk = self.base.encoder.layers[-1]; m = blk.mixer
        xin = blk.norm(res)
        if self.corr_net is None or not self.use_corr:
            return dict(pred=self.head(self.tokens_out(m(xin), res), means, stdev), reg=None, step=None, step0=None)
        x, dt, A, B, C, D, z, bias = mamba_proj(m, xin)
        dt = F.softplus(dt + bias[None, :, None])
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :]); dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        b, L, l, c, P = x.shape[0], x.shape[-1], self.n_patch, self.n_vars, self.patch_len
        with torch.no_grad():
            h, h0 = x.new_zeros(b, x.shape[1], A.shape[1]), []
            for s in range(L):
                h = dA[:, :, s] * h + dBu[:, :, s]; h0.append(h.flatten(1))
            h0 = rearrange(torch.stack(h0, 1), 'b (c l) n -> b c l n', c=c)
            pred0 = self.ro(h0[:, :, :-1])                                              # predicts x_{k,t+1}, t = 0..l-2
            es = torch.cat([torch.zeros_like(xp[:, :, :1]), xp[:, :, 1:] - pred0], 2)  # e*_{k,t}, b c l P
            step0 = (xp[:, :, 1:] - pred0).square().mean()
            if trace is not None:
                trace['e0'] = (xp[:, :, 1:] - pred0).square().mean(-1); trace['pri'] = []; trace['post'] = []
                trace.update(dA=dA, h0=h0, xp=xp, C=[])
        h = x.new_zeros(b, x.shape[1], A.shape[1])
        ys, regs, steps, hs, cs = [], [], [], [], []
        steps_mem, mstats = [], dict(ent=[], ent_norm=[], unif_heads=[], g=[], ratio=[])
        for s in range(L):
            h = dA[:, :, s] * h + dBu[:, :, s]
            ys.append((torch.einsum('bdn,bn->bd', h, C[:, :, s]) + D * x[:, :, s]) * F.silu(z[:, :, s]))
            k, t = divmod(s, l)
            if t == 0:
                mkeys, mvals = [], []                                                  # memory scope: own variable, earlier patches (t' < t), in-window
            if t <= l - 2:
                cross = torch.cat([es[:, :k, t + 1], es.new_zeros(b, c - 1 - k, P)], 1).reshape(b, -1)
                u = torch.cat([es[:, k, t], cross], 1)
                if record is not None:
                    record.append((s, u))
                if self.shuffle:
                    u = u.roll(1, 0)
                if trace is not None:
                    trace['pri'].append((xp[:, k, t + 1] - self.ro(h.flatten(1))).square().mean(1))
                h_minus = h
                Cm = self.corr_net(h_minus.detach(), u)
                regs.append(Cm.square().mean()); hs.append(h_minus.detach()); cs.append(Cm)
                if trace is not None:
                    trace['C'].append(Cm)
                if self.mem is not None:                                                # h_mem = h- + g * sum_j a_j C_j, re-predict x_{k,t+1} from it
                    if mem_bank is not None:                                            # unrelated-history control: another window's slots
                        rk = [mem_bank[k * (l - 1) + j][0] for j in range(t)]; rv = [mem_bank[k * (l - 1) + j][1] for j in range(t)]
                    else:
                        rk, rv = mkeys, mvals
                    if self.mem_net is not None and rv:
                        V = torch.stack(rv, 1)
                        if self.mem == 'shuf':
                            V = V.roll(1, 1)                                            # key_j <-> value_{j-1}, earliest key <-> latest value
                        if self.cut_value:
                            V = V.detach()
                        inj, a, g = self.mem_net(h_minus, torch.stack(rk, 1), V)
                        h = h_minus + inj
                        ent = -(a * a.clamp_min(1e-12).log()).sum(-1)                  # (b, nh)
                        mstats['ent'].append(ent.mean().detach())
                        if a.shape[-1] >= 2:                                            # entropy / log M: 1 = uniform; share of heads > 0.95
                            en = ent / math.log(a.shape[-1])
                            mstats['ent_norm'].append(en.mean().detach()); mstats['unif_heads'].append((en > 0.95).float().mean().detach())
                        mstats['g'].append(g.mean().detach())
                        mstats['ratio'].append((inj.flatten(1).norm(dim=1) / Cm.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().detach())
                    yhat_mem = self.ro(h.flatten(1))
                    steps_mem.append((xp[:, k, t + 1] - yhat_mem).square().mean())
                    if trace is not None:
                        trace.setdefault('mem', []).append(dict(k=k, t=t, yhat=yhat_mem, a=a if self.mem_net is not None and rv else None,
                                                                inj=inj if self.mem_net is not None and rv else None,
                                                                keys=torch.stack(rk, 1) if rk else None))
                    if self.mem_net is not None:
                        mkeys.append(self.mem_net.key(h_minus)); mvals.append(Cm)
                        if trace is not None:
                            trace.setdefault('slots', []).append((mkeys[-1], Cm))
                h_pre = h                                                               # h- (or h_mem) before the site's own correction
                h = h + (Cm.detach() if self.cut_carry else Cm)
                h_step = (h_pre.detach() + Cm) if self.step_sg else h                   # step_sg: the step loss reaches CorrNet and ro only, never the backbone
                steps.append((xp[:, k, t + 1] - self.ro(h_step.flatten(1))).square().mean())
                if trace is not None:
                    trace['post'].append((xp[:, k, t + 1] - self.ro(h.flatten(1))).square().mean(1))
        tok = self.tokens_out(m.out_proj(torch.stack(ys, 1)), res)
        decor = channel_decor(torch.cat(hs, 0), torch.cat(cs, 0))                      # per-channel corr(h-, C)^2 over batch x sites
        out = dict(pred=self.head(tok, means, stdev), reg=torch.stack(regs).mean(), step=torch.stack(steps).mean(), step0=step0, decor=decor)
        if self.mem is not None:
            out.update(step_mem=torch.stack(steps_mem).mean(), steps_mem=steps_mem,
                       mem_stats={n: torch.stack(v).mean().item() for n, v in mstats.items() if v})
        return out

    @torch.no_grad()
    def scan_states(self, x_enc):
        """Uncorrected A0, last layer: state h after every scan token (b, c, l, d_inner*d_state), token outputs o (b, c, l, d), patches xp (b, c, l, P)."""
        xn, _, _ = self.norm(x_enc)
        res, xp = self.lower(xn)
        blk = self.base.encoder.layers[-1]; m = blk.mixer
        xin = blk.norm(res)
        x, dt, A, B, C, D, z, bias = mamba_proj(m, xin)
        dt = F.softplus(dt + bias[None, :, None])
        dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :]); dBu = (dt * x).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)
        h = x.new_zeros(x.shape[0], x.shape[1], A.shape[1]); hs = []
        for s in range(x.shape[-1]):
            h = dA[:, :, s] * h + dBu[:, :, s]; hs.append(h.flatten(1))
        o = self.tokens_out(m(xin), res)
        r = lambda u: rearrange(u, 'b (c l) d -> b c l d', c=self.n_vars)
        return r(torch.stack(hs, 1)), r(o), xp

    @torch.no_grad()
    def aux_pairs(self, x_enc):
        """Uncorrected A0: features o_s (official last-layer token output) and targets x_{s+1} for every within-variable next patch."""
        xn, _, _ = self.norm(x_enc)
        res, xp = self.lower(xn)
        blk = self.base.encoder.layers[-1]
        tok = rearrange(self.tokens_out(blk.mixer(blk.norm(res)), res), 'b (c l) d -> b c l d', c=self.n_vars)
        return tok[:, :, :-1].reshape(-1, tok.shape[-1]), xp[:, :, 1:].reshape(-1, self.patch_len)
