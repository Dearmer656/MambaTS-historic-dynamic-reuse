"""PAT-304 A0-frozen innovation correction: official run.py / data / seed / early stopping / vali / test, model = A0Corr (frozen A0 checkpoint).
Env: A0C_MODE (self = F1/F2 | cross = F3), A0C_STEP (lambda of the per-step innovation loss, default 0), A0C_CKPT (A0 checkpoint.pth, required), A0C_CORR (shared | gated = GatedCorrNet, C(h, e=0) = 0 | none = A0 baseline in the fixed order), A0C_DECOR (weight of the per-channel corr(h-, C)^2 penalty, default 0), A0C_STEP_SG=1 (cross mode: step loss on sg(h-) + C, no backbone gradient), A0C_SCRATCH_LR (from scratch: backbone+ro lr factor), A0C_SHUFFLE=1 (residual rolled in batch),
A0C_MEM (cross mode, state-level retrieval memory: unset = legacy | none = M0, step_mem on h- | real = M1 | shuf = M2, deterministic time
permutation of the key-value pairing within the sequence; the unrelated-history null is evaluated by mem_audit), A0C_STEP_MEM (lambda of the per-site re-prediction loss from h_mem, default 0), A0C_MEM_DK (per-head key dim, default 64),
A0C_MEM_GROUP (channels per retrieval head, 0 = one head over the whole state; 4 = 4 channels x d_state 16 = 64 features per head),
A0C_GAMMA (default 1e-4). Stage 1: ridge fit of the next-patch predictor aux on the frozen A0. Stage 2: train CorrNet only (official MSE on the
last origin + gamma * mean C^2). Vali/test are the official ones (run.py calls test after train)."""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.MambaTS import Model
from patchmem.a0corr import A0Corr
from utils.tools import EarlyStopping, adjust_learning_rate, set_seed


class Exp_A0Corr(Exp_Long_Term_Forecast):
    def _build_model(self):
        a = self.args
        base = Model(a).float()
        ckpt = os.environ.get('A0C_CKPT', '')
        freeze = ckpt not in ('', 'none')                                  # no checkpoint -> train backbone + ro from scratch alongside corr_net
        if freeze:
            base.load_state_dict(torch.load(ckpt, map_location='cpu'))
        base.to(self.device)
        enc = base.encoder.eval(); enc.ids_shuffle = None
        set_seed(a.seed)                                                   # same as run.py before the official test: the first SA solve fixes the order
        with torch.no_grad():
            enc(torch.zeros(1, enc.n_vars * (a.seq_len // a.patch_len), a.d_model, device=self.device))
        order = enc.ids_shuffle[0].tolist()
        corr = os.environ.get('A0C_CORR', 'shared'); corr = None if corr == 'none' else corr
        step_sg = os.environ.get('A0C_STEP_SG', '0') == '1'
        mem = os.environ.get('A0C_MEM') or None
        print(f'A0C: ckpt {ckpt or "(none, from scratch)"} | freeze {freeze} | scan order {order} | corr {corr} shuffle {os.environ.get("A0C_SHUFFLE", "0")} | step_sg {step_sg} | mem {mem} group {os.environ.get("A0C_MEM_GROUP", "0")}', flush=True)
        return A0Corr(base, order, corr=corr, shuffle=os.environ.get('A0C_SHUFFLE', '0') == '1', mode=os.environ.get('A0C_MODE', 'self'), freeze=freeze, step_sg=step_sg,
                      mem=mem, mem_dk=int(os.environ.get('A0C_MEM_DK', '64')), mem_group=int(os.environ.get('A0C_MEM_GROUP', '0')) or None).float()

    @torch.no_grad()
    def fit_ro(self, train_loader, lam=1e-4):
        """Ridge fit of the state readout ro: last-layer state h_{k,t} (uncorrected A0) -> x_{k,t+1}, t = 0..l-2, on train."""
        m = self.model; m.eval(); XtX = XtY = None
        for bx, _, _, _ in train_loader:
            h, _, xp = m.scan_states(bx.float().to(self.device))
            X = h[:, :, :-1].reshape(-1, h.shape[-1]).double(); X = torch.cat([X, torch.ones_like(X[:, :1])], 1); Y = xp[:, :, 1:].reshape(-1, m.patch_len).double()
            XtX = X.T @ X if XtX is None else XtX + X.T @ X; XtY = X.T @ Y if XtY is None else XtY + X.T @ Y
        W = torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device), XtY).float()
        m.ro.weight.copy_(W[:-1].T); m.ro.bias.copy_(W[-1])

    @torch.no_grad()
    def step_eval(self, loader):
        m = self.model; m.eval(); a = b = n = 0
        for bx, _, _, _ in loader:
            out = m.run_cross(bx.float().to(self.device)); a += out['step'].item() * bx.shape[0]; b += out['step0'].item() * bx.shape[0]; n += bx.shape[0]
        return a / n, b / n

    @torch.no_grad()
    def fit_aux(self, train_loader, vali_loader, lam=1e-3):
        m = self.model; m.eval(); d = m.aux.in_features
        XtX = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=self.device); XtY = torch.zeros(d + 1, m.patch_len, dtype=torch.float64, device=self.device)
        for bx, _, _, _ in train_loader:
            f, y = m.aux_pairs(bx.float().to(self.device))
            f = torch.cat([f, torch.ones_like(f[:, :1])], 1).double()
            XtX += f.T @ f; XtY += f.T @ y.double()
        W = torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(d + 1, device=self.device, dtype=torch.float64), XtY).float()
        m.aux.weight.copy_(W[:d].T); m.aux.bias.copy_(W[d])
        se = np.zeros(3); n = 0
        for bx, _, _, _ in vali_loader:
            f, y = m.aux_pairs(bx.float().to(self.device))
            prev = m.lower(m.norm(bx.float().to(self.device))[0])[1][:, :, :-1].reshape(-1, m.patch_len)
            for i, p in enumerate((m.aux(f), torch.zeros_like(y), prev[:, -1:].expand_as(y))):
                se[i] += (p - y).square().sum().item()
            n += y.numel()
        print(f'aux next-patch val MSE (normalised): aux {se[0] / n:.4f} | zero {se[1] / n:.4f} | last-value {se[2] / n:.4f}', flush=True)

    @torch.no_grad()
    def diag(self, ckpt, vali_loader, test_loader):
        """Correction magnitude of a trained checkpoint: per site |C_s|, |h_s-|, |e_s| (Frobenius per sample), and the MSE with / without it."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval()
        cn, fwd = m.corr_net, m.corr_net.forward
        for name, loader in (('val', vali_loader), ('test', test_loader)):
            st = []
            def rec(h, e):
                C = fwd(h, e)
                st.append(torch.stack([torch.full_like(e[:, 0], len(st) % 97),C.flatten(1).norm(dim=1), h.flatten(1).norm(dim=1), e.norm(dim=1) / e.shape[1] ** 0.5,
                                       C.square().mean((1, 2)), h.square().mean((1, 2))], 1))
                return C
            cn.forward = rec; se = np.zeros(2); n = 0
            for bx, by, _, _ in loader:
                bx = bx.float().to(self.device); y = by.float().to(self.device)[:, -self.args.pred_len:]
                m.use_corr = True; p1 = m(bx); m.use_corr = False; p0 = m(bx)
                se += [(p1 - y).square().sum().item(), (p0 - y).square().sum().item()]; n += y.numel()
            cn.forward = fwd; m.use_corr = True
            S = torch.cat(st, 0)                                                      # rows: site, |C|, |h-|, rms e, mean C^2, mean h^2
            r = S[:, 1] / S[:, 2]
            print(f'[{name}] MSE with corr {se[0] / n:.4f} | A0 (no corr) {se[1] / n:.4f}')
            print(f'[{name}] |C|/|h-| mean {r.mean():.3f} median {r.median():.3f} p90 {r.quantile(0.9):.3f} max {r.max():.2f} | '
                  f'mean C^2 {S[:, 4].mean():.3e} mean h^2 {S[:, 5].mean():.3e} | rms e {S[:, 3].mean():.3f}')
            rs = [r[S[:, 0] == k].mean().item() for k in range(0, 97, 7)]
            print(f'[{name}] |C|/|h-| by scan site 0,7,..,91: {np.round(rs, 3).tolist()}', flush=True)

    @torch.no_grad()
    def diag_carry(self, ckpt, vali_loader, test_loader):
        """F3 per-t split of the next-patch gain. prior = ro(h-_t) before the site's own correction (carry-over of earlier corrections), post = ro(h+_t)."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval(); m.use_corr = True
        for name, loader in (('val', vali_loader), ('test', test_loader)):
            S = None; n = 0
            for bx, _, _, _ in loader:
                tr = {}; m.run_cross(bx.float().to(self.device), trace=tr)
                b, c, l1 = tr['e0'].shape
                cur = torch.stack([tr['e0'], torch.stack(tr['pri'], 1).view(b, c, l1), torch.stack(tr['post'], 1).view(b, c, l1)]).sum(1)
                S = cur if S is None else S + cur; n += b
            S = (S / n).cpu().numpy()                                                    # (3, c, l-1): e0, prior, post
            for lab, sl in (('k=0 only (no carry from other variables)', slice(0, 1)), ('all k', slice(None))):
                e0, pri, post = (S[i][sl].mean(0) for i in range(3))
                print(f'[{name}] {lab}: A0 MSE by t=0..13     {np.round(e0, 3).tolist()}')
                print(f'[{name}] {lab}: carry (prior) rel by t {np.round(pri / e0 - 1, 3).tolist()}  mean {pri.mean() / e0.mean() - 1:+.2%}')
                print(f'[{name}] {lab}: carry+fresh (post) rel {np.round(post / e0 - 1, 3).tolist()}  mean {post.mean() / e0.mean() - 1:+.2%}', flush=True)

    @torch.no_grad()
    def carry_relearn(self, ckpt, train_loader, vali_loader, test_loader, lam=1e-3):
        """Does the carry-over conclusion survive giving the readout a fresh chance? carry_decomp reused the frozen ro (ridge-fit on ordinary,
        uncorrected A0 states) to read both the decayed and the undecayed correction at a later site -- ro was never fit to decode C's content,
        only rescaled by a scalar. Here a brand-new ridge map W' (D x P, D = d_inner*d_state) is fit ON TRAIN from the correction itself (decayed
        delta_{s->s'} or the undecayed C_s) to the actual residual r_{s'} = x_{s'+1} - ro(h0_{s'}) of the uncorrected A0 at the target site, per
        lag n and group (same variable / next variable), then evaluated (R^2 = 1 - SSE / sum(r^2)) on held-out val/test. This is the best any
        linear readout -- however it were re-fit or adapted -- could ever extract from that specific correction at that lag."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval(); m.use_corr = True
        c, l = m.n_vars, m.n_patch; L = c * l
        src = torch.tensor([s for s in range(L) if s % l <= l - 2], device=self.device); ksrc = src // l
        D = m.corr_net.Wc.shape[0] * m.corr_net.Wc.shape[-1] if hasattr(m.corr_net, 'Wc') else m.ro.weight.shape[1]
        acc = {}                                                                         # (lag, grp, kind) -> [XtX, XtY, YtY, n]
        for split, loader in (('train', train_loader), ('val', vali_loader), ('test', test_loader)):
            for bx, _, _, _ in loader:
                tr = {}; m.run_cross(bx.float().to(self.device), trace=tr)
                dA, h0, xp = tr['dA'].double(), tr['h0'].double(), tr['xp'].double()
                W = m.ro.weight.double()
                rf = torch.zeros_like(xp); rf[:, :, :-1] = xp[:, :, 1:] - (h0[:, :, :-1] @ W.T + m.ro.bias.double())
                rf = rf.flatten(1, 2)                                                    # b L P
                Cs = torch.stack(tr['C'], 1).double().flatten(2)                         # b S D
                Dl = Cs.clone()
                for n in range(0, 2 * l):
                    tgt = src + n
                    if n:
                        Dl = Dl * dA[:, :, tgt.clamp(max=L - 1)].permute(0, 2, 1, 3).flatten(2)
                    ok = (tgt < L) & (tgt < (ksrc + 2) * l) & (tgt % l <= l - 2)
                    if not ok.any():
                        continue
                    r = rf[:, tgt.clamp(max=L - 1)]
                    same = (tgt // l == ksrc)
                    for grp, msk in ((('same', n), ok & same), (('next', 0), ok & ~same)):
                        if not msk.any():
                            continue
                        for kind, X in (('dec', Dl), ('und', Cs)):
                            Xm = X[:, msk]; Rm = r[:, msk]
                            key = grp + (kind,)
                            stat = [Xm.reshape(-1, D).T @ Xm.reshape(-1, D), Xm.reshape(-1, D).T @ Rm.reshape(-1, Rm.shape[-1]),
                                    Rm.square().sum().item(), Rm.shape[0] * Rm.shape[1]]
                            if split not in acc:
                                acc[split] = {}
                            if key not in acc[split]:
                                acc[split][key] = stat
                            else:
                                a0 = acc[split][key]; acc[split][key] = [a0[0] + stat[0], a0[1] + stat[1], a0[2] + stat[2], a0[3] + stat[3]]
        out = {}
        for key in acc['train']:
            XtX, XtY, _, _ = acc['train'][key]
            Wp = torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(D, dtype=XtX.dtype, device=XtX.device), XtY)
            r2 = {}
            for split in ('train', 'val', 'test'):
                if key not in acc[split]:
                    continue
                XtX_s, XtY_s, yty, n = acc[split][key]
                sse = yty - 2 * (Wp * XtY_s).sum().item() + (Wp * (XtX_s @ Wp)).sum().item()
                r2[split] = 1 - sse / yty
            out[key] = r2
        for kind, lab in (('und', 'refit readout, undecayed C'), ('dec', 'refit readout, decayed delta')):
            for lag in sorted({k[1] for k in out if k[2] == kind and k[0] == 'same'}):
                k = ('same', lag, kind); r2 = out.get(k)
                if r2:
                    print(f'[relearn] {lab} same lag {lag:2d}: train {r2.get("train", float("nan")):+.4f} | val {r2.get("val", float("nan")):+.4f} | '
                          f'test {r2.get("test", float("nan")):+.4f}', flush=True)
            k = ('next', 0, kind); r2 = out.get(k)
            if r2:
                print(f'[relearn] {lab} next var   : train {r2.get("train", float("nan")):+.4f} | val {r2.get("val", float("nan")):+.4f} | '
                      f'test {r2.get("test", float("nan")):+.4f}', flush=True)

    def path_lag1(self, ckpt, train_loader, vali_loader, test_loader, lr=1e-3, epochs=15, patience=3):
        """Minimal content-dependent (PaTH-style) carry test, lag 1 only. w, beta = Linear(h0_{s+1}) (per channel, shared weights), a single
        Householder H = I - sigmoid(beta)*w*w^T (w normalised) built from the NEXT (uncorrected) patch's own state -- the one piece of "path"
        content available between s and s+1. Applied to the source correction: C' = H C_s. Both C' and raw C_s are read by the SAME frozen ro;
        trained (only the small Linear) to minimise MSE(ro(C'), r_{s+1}) on train, early-stopped on val, reported as R^2 on held-out test,
        against the untransformed C_s baseline (same ro, no path content)."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval(); m.use_corr = True
        c, l = m.n_vars, m.n_patch; L = c * l
        d_inner, d_state = m.corr_net.Wc.shape[0], m.corr_net.Wc.shape[-1]
        src = torch.tensor([s for s in range(L) if s % l <= l - 2 and (s + 1) % l != 0], device=self.device)  # need a within-variable s+1
        gate = torch.nn.Linear(d_state, d_state + 1).to(self.device)
        opt = torch.optim.Adam(gate.parameters(), lr=lr)

        def batch_stats(bx):
            tr = {}; m.run_cross(bx.float().to(self.device), trace=tr)
            dA, h0, xp = tr['dA'].double(), tr['h0'].double(), tr['xp'].double()         # h0: b c l D (D = d_inner*d_state, flattened)
            b = h0.shape[0]
            W = m.ro.weight.double()
            rf = torch.zeros_like(xp); rf[:, :, :-1] = xp[:, :, 1:] - (h0[:, :, :-1] @ W.T + m.ro.bias.double())
            rf = rf.flatten(1, 2)[:, src + 1]                                            # b S P, target residual at s+1
            h0f = h0.flatten(1, 2)                                                       # b L D, scan order (c outer, l inner), matches src/dA
            Cs = torch.stack(tr['C'], 1).double()                                        # b S' d n (S' = all corrected sites, t <= l-2)
            csite = torch.tensor([ss for ss in range(L) if ss % l <= l - 2], device=self.device)
            Cs = Cs[:, torch.searchsorted(csite, src)]                                   # b S d n, aligned to src
            hn = h0f[:, src + 1].reshape(b, -1, d_inner, d_state)                        # b S d n, uncorrected state at s+1
            return Cs.float(), hn.float(), rf.float(), W.float(), m.ro.bias.float()

        def transform(Cs, hn):
            out = gate(hn)
            w = torch.nn.functional.normalize(out[..., :-1], dim=-1); beta = torch.sigmoid(out[..., -1:])
            return Cs - beta * w * (w * Cs).sum(-1, keepdim=True)

        def readout(X, W, bias):
            return torch.einsum('bsdn,pdn->bsp', X, W.reshape(-1, d_inner, d_state)) + bias

        def run_epoch(loader, train):
            tot_t = tot_b = n = 0.0
            for bx, _, _, _ in loader:
                Cs, hn, rf, W, bias = batch_stats(bx)
                if train:
                    opt.zero_grad()
                pred_t = readout(transform(Cs, hn), W, bias)
                loss = (pred_t - rf).square().mean()
                if train:
                    loss.backward(); opt.step()
                with torch.no_grad():
                    pred_b = readout(Cs, W, bias)
                    tot_t += (pred_t - rf).square().sum().item(); tot_b += (pred_b - rf).square().sum().item(); n += rf.numel()
            return tot_t / n, tot_b / n

        best_val = 1e9; best_state = None; bad = 0
        for ep in range(epochs):
            tl, _ = run_epoch(train_loader, True)
            with torch.no_grad():
                vt, vb = run_epoch(vali_loader, False)
            print(f'[path1] epoch {ep + 1} train_mse(transform) {tl:.4f} | val mse transform {vt:.4f} vs raw {vb:.4f}', flush=True)
            if vt < best_val:
                best_val = vt; best_state = {k: v.clone() for k, v in gate.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print('early stop', flush=True); break
        gate.load_state_dict(best_state)
        with torch.no_grad():
            for name, loader in (('train', train_loader), ('val', vali_loader), ('test', test_loader)):
                sse_t = sse_b = sst = 0.0
                for bx, _, _, _ in loader:
                    Cs, hn, rf, W, bias = batch_stats(bx)
                    pt = readout(transform(Cs, hn), W, bias); pb = readout(Cs, W, bias)
                    sse_t += (pt - rf).square().sum().item(); sse_b += (pb - rf).square().sum().item(); sst += rf.square().sum().item()
                print(f'[path1 FINAL] {name}: R2 raw-C (ro only) {1 - sse_b / sst:+.4f} | R2 path-transformed C {1 - sse_t / sst:+.4f}', flush=True)

    def path_lag1_full(self, ckpt, train_loader, vali_loader, test_loader, lr=1e-3, epochs=15, patience=3):
        """path_lag1 with the readout ALSO freed (not the frozen ro): the most generous combined test of "content-dependent transport +
        adapted readout". Two arms trained jointly, same data/epochs/optimizer for a fair comparison: plain = a trainable linear readout W'
        applied directly to raw C_s (no path content -- the SGD analogue of carry_relearn's closed-form ridge); path = the same Householder
        transform as path_lag1 (w, beta from the next, uncorrected patch's state) followed by its OWN trainable readout. R^2 on held-out
        val/test for both arms, vs predicting zero (same scale as every other carry-over number in this file)."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval(); m.use_corr = True
        c, l = m.n_vars, m.n_patch; L = c * l; P = m.patch_len
        d_inner, d_state = m.corr_net.Wc.shape[0], m.corr_net.Wc.shape[-1]
        src = torch.tensor([s for s in range(L) if s % l <= l - 2 and (s + 1) % l != 0], device=self.device)
        gate = torch.nn.Linear(d_state, d_state + 1).to(self.device)
        ro_plain = torch.nn.Linear(d_inner * d_state, P).to(self.device); ro_path = torch.nn.Linear(d_inner * d_state, P).to(self.device)
        opt = torch.optim.Adam(list(gate.parameters()) + list(ro_plain.parameters()) + list(ro_path.parameters()), lr=lr)

        def batch_stats(bx):
            tr = {}; m.run_cross(bx.float().to(self.device), trace=tr)
            h0, xp = tr['h0'].double(), tr['xp'].double()
            b = h0.shape[0]
            Wro = m.ro.weight.double()
            rf = torch.zeros_like(xp); rf[:, :, :-1] = xp[:, :, 1:] - (h0[:, :, :-1] @ Wro.T + m.ro.bias.double())
            rf = rf.flatten(1, 2)[:, src + 1]                                            # b S P
            h0f = h0.flatten(1, 2)
            Cs = torch.stack(tr['C'], 1).double()
            csite = torch.tensor([ss for ss in range(L) if ss % l <= l - 2], device=self.device)
            Cs = Cs[:, torch.searchsorted(csite, src)].flatten(2)                        # b S D
            hn = h0f[:, src + 1].reshape(b, -1, d_inner, d_state)                         # b S d n
            return Cs.float(), hn.float(), rf.float()

        def predict(Cs, hn):
            out = gate(hn)
            w = torch.nn.functional.normalize(out[..., :-1], dim=-1); beta = torch.sigmoid(out[..., -1:])
            Cs4 = Cs.reshape(*Cs.shape[:-1], d_inner, d_state)
            Cp = (Cs4 - beta * w * (w * Cs4).sum(-1, keepdim=True)).flatten(2)
            return ro_plain(Cs), ro_path(Cp)

        best_val = 1e9; best_state = None; bad = 0
        for ep in range(epochs):
            tl = n = 0.0
            for bx, _, _, _ in train_loader:
                Cs, hn, rf = batch_stats(bx); opt.zero_grad()
                pp, pt = predict(Cs, hn); loss = (pp - rf).square().mean() + (pt - rf).square().mean()
                loss.backward(); opt.step(); tl += loss.item() * rf.numel(); n += rf.numel()
            with torch.no_grad():
                sse_p = sse_t = sst = vn = 0.0
                for bx, _, _, _ in vali_loader:
                    Cs, hn, rf = batch_stats(bx); pp, pt = predict(Cs, hn)
                    sse_p += (pp - rf).square().sum().item(); sse_t += (pt - rf).square().sum().item(); sst += rf.square().sum().item(); vn += rf.numel()
            vp, vt = sse_p / vn, sse_t / vn
            print(f'[path1f] epoch {ep + 1} train_mse {tl / n:.4f} | val mse plain {vp:.4f} path {vt:.4f} | val R2 plain {1 - sse_p / sst:+.4f} path {1 - sse_t / sst:+.4f}', flush=True)
            if vt < best_val:
                best_val = vt; best_state = (gate.state_dict(), ro_plain.state_dict(), ro_path.state_dict()); bad = 0
                best_state = tuple({k: v.clone() for k, v in d.items()} for d in best_state)
            else:
                bad += 1
                if bad >= patience:
                    print('early stop', flush=True); break
        gate.load_state_dict(best_state[0]); ro_plain.load_state_dict(best_state[1]); ro_path.load_state_dict(best_state[2])
        with torch.no_grad():
            for name, loader in (('train', train_loader), ('val', vali_loader), ('test', test_loader)):
                sse_p = sse_t = sst = 0.0
                for bx, _, _, _ in loader:
                    Cs, hn, rf = batch_stats(bx); pp, pt = predict(Cs, hn)
                    sse_p += (pp - rf).square().sum().item(); sse_t += (pt - rf).square().sum().item(); sst += rf.square().sum().item()
                print(f'[path1f FINAL] {name}: R2 plain (free readout, no path) {1 - sse_p / sst:+.4f} | R2 path+free readout {1 - sse_t / sst:+.4f}', flush=True)

    @torch.no_grad()
    def carry_decomp(self, ckpt, vali_loader, test_loader):
        """F3 carry decomposition. The last-layer recurrence is linear in h and dA does not depend on the corrections, so the state at a later
        site s' holds delta_{s->s'} = (prod_{u=s+1..s'} dA_u) * C_s from every earlier correction C_s, exactly. For each source site s and lag n
        (same variable; 'next' = carried into the next scan variable) with residual r_{s'} = x_{s'+1} - ro(h0_{s'}) of the uncorrected A0:
        g = W_ro delta (the readout effect of the carried correction alone), gC = W_ro C_s (the same correction if it had not decayed).
        Reported per lag: energy retention |delta|^2/|C|^2, gain at scale 1 = 1 - |r-g|^2/|r|^2, best-scale gain (<g,r>^2/(|g|^2|r|^2), direction
        only) for delta and for the undecayed C_s, and the share of correction energy in slow state dims (dA > 0.9 at the next step)."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval(); m.use_corr = True
        c, l = m.n_vars, m.n_patch; L = c * l; W = m.ro.weight.double()
        src = torch.tensor([s for s in range(L) if s % l <= l - 2], device=self.device); ksrc = src // l
        names = ['rr', 'gr', 'gg', 'cr', 'cc', 'dd', 'C2']
        for name, loader in (('val', vali_loader), ('test', test_loader)):
            acc = {}; slow = torch.zeros(2, dtype=torch.float64, device=self.device)
            for bx, _, _, _ in loader:
                tr = {}; m.run_cross(bx.float().to(self.device), trace=tr)
                dA, h0, xp = tr['dA'].double(), tr['h0'].double(), tr['xp'].double()
                b = xp.shape[0]
                rf = torch.zeros_like(xp); rf[:, :, :-1] = xp[:, :, 1:] - (h0[:, :, :-1] @ W.T + m.ro.bias.double())
                rf = rf.flatten(1, 2)                                                    # b L P, residual at each site (0 at t = l-1)
                Cs = torch.stack(tr['C'], 1).double()                                    # b S d N
                C2 = Cs.square().sum((2, 3)); gC = Cs.flatten(2) @ W.T
                nxt = dA[:, :, (src + 1).clamp(max=L - 1)].permute(0, 2, 1, 3)
                slow += torch.stack([(Cs.square() * (nxt > 0.9)).sum(), Cs.square().sum()])
                Dl = Cs.clone()
                for n in range(0, 2 * l):
                    tgt = src + n
                    if n:
                        Dl = Dl * dA[:, :, tgt.clamp(max=L - 1)].permute(0, 2, 1, 3)
                    ok = (tgt < L) & (tgt < (ksrc + 2) * l) & (tgt % l <= l - 2)
                    if not ok.any():
                        continue
                    r = rf[:, tgt.clamp(max=L - 1)]; g = Dl.flatten(2) @ W.T
                    vals = [r.square().sum(-1), (g * r).sum(-1), g.square().sum(-1), (gC * r).sum(-1), gC.square().sum(-1), Dl.square().sum((2, 3)), C2]
                    same = (tgt // l == ksrc)
                    for grp, msk in ((('same', n), ok & same), (('next', 0), ok & ~same)):
                        if msk.any():
                            v = torch.stack([u[:, msk].sum() for u in vals])
                            acc[grp] = v if grp not in acc else acc[grp] + v
            print(f'[decomp {name}] correction energy share in slow dims (dA > 0.9 at the next step): {(slow[0] / slow[1]).item():.3f}')
            for grp in sorted(acc, key=lambda g: (g[0] != 'same', g[1])):
                rr, gr, gg, cr, cc, dd, c2 = acc[grp].tolist()
                lab = f'lag {grp[1]:2d}' if grp[0] == 'same' else 'next var'
                print(f'[decomp {name}] {lab}: retention {dd / c2:.3f} | gain@1 {(2 * gr - gg) / rr:+.4f} | best-scale {gr * gr / (gg * rr + 1e-30):.4f} (sign {"+" if gr > 0 else "-"}) | '
                      f'undecayed gain@1 {(2 * cr - cc) / rr:+.4f} best-scale {cr * cr / (cc * rr + 1e-30):.4f} (sign {"+" if cr > 0 else "-"})', flush=True)

    @torch.no_grad()
    def diag_step(self, ckpt, vali_loader, test_loader):
        """Per-step accuracy: next-patch prediction MSE (innovation e_s^2) along the scan with the correction vs the uncorrected A0, by patch t."""
        m = self.model; m.load_state_dict(torch.load(ckpt, map_location=self.device)); m.eval()
        c, l = m.n_vars, m.n_patch
        for name, loader in (('val', vali_loader), ('test', test_loader)):
            s1 = torch.zeros(c, l, device=self.device); s0 = torch.zeros(c, l, device=self.device); n = 0; se = np.zeros(2); ny = 0
            for bx, by, _, _ in loader:
                bx = bx.float().to(self.device); y = by.float().to(self.device)[:, -self.args.pred_len:]
                rec = []; m.use_corr = True; p1 = m.run(bx, record=rec)['pred']
                for s, e in rec:
                    s1[s // l, s % l] += e.square().mean(1).sum()
                f, yt = m.aux_pairs(bx)
                e0 = (yt - m.aux(f)).reshape(bx.shape[0], c, l - 1, -1).square().mean(-1).sum(0)
                s0[:, 1:] += e0; n += bx.shape[0]
                m.use_corr = False; p0 = m(bx); m.use_corr = True
                se += [(p1 - y).square().sum().item(), (p0 - y).square().sum().item()]; ny += y.numel()
            s0[c - 1, l - 1] = 0                                                        # the last scan token is never corrected
            mask = torch.zeros(c, l, dtype=torch.bool, device=self.device); mask[:, 1:] = True; mask[c - 1, l - 1] = False
            a1, a0 = (s1[mask].sum() / (n * mask.sum())).item(), (s0[mask].sum() / (n * mask.sum())).item()
            print(f'[{name}] final forecast MSE: corr {se[0] / ny:.4f} | A0 {se[1] / ny:.4f}')
            print(f'[{name}] next-patch MSE (all corrected sites): corr {a1:.4f} | A0 {a0:.4f} | rel {a1 / a0 - 1:+.2%}')
            t1 = (s1[:, 1:].sum(0) / (n * c)).cpu().numpy(); t0 = (s0[:, 1:].sum(0) / (n * c)).cpu().numpy()
            print(f'[{name}] by patch t=1..14, rel change: {np.round(t1 / t0 - 1, 3).tolist()}')
            v1 = torch.stack([s1[k][mask[k]].sum() / (n * mask[k].sum()) for k in range(c)]); v0 = torch.stack([s0[k][mask[k]].sum() / (n * mask[k].sum()) for k in range(c)])
            print(f'[{name}] by scan variable position k=0..{c - 1}, rel change: {np.round((v1 / v0 - 1).cpu().numpy(), 3).tolist()}', flush=True)

    @torch.no_grad()
    def innov_diag(self, train_loader, vali_loader, test_loader, lam=1e-4, E_fn=None, tag='aux(o)', feats_fn=None):
        """Linear predictability of the next innovation (upper-bound proxy for any innovation-driven correction). Innovations of the frozen A0:
        E[k, t] = x_{k,t} - aux(o_{k,t-1}), k = scan position, t = 1..14. Ridge fitted on train, R^2 = 1 - SSE/SST on each split.
        lag1: E[k,t] ~ E[k,t-1]; lag2: ~ E[k,t-1], E[k,t-2]; cross: ~ E[k,t-1] + E[j,t] for every j < k (already scanned; k >= 1)."""
        m = self.model; m.eval(); c = m.n_vars
        def feats(E):                                                               # E (b, c, 14, P) -> {name: [(X, Y)]}
            b, _, T, P = E.shape
            one = lambda X: torch.cat([X, torch.ones_like(X[..., :1])], -1)
            out = {'lag1': [(one(E[:, :, :-1].reshape(-1, P)), E[:, :, 1:].reshape(-1, P))],
                   'lag2': [(one(torch.cat([E[:, :, 1:-1], E[:, :, :-2]], -1).reshape(-1, 2 * P)), E[:, :, 2:].reshape(-1, P))]}
            out['cross'] = [(one(torch.cat([E[:, k, :-1], E[:, :k, 1:].permute(0, 2, 1, 3).reshape(b, T - 1, k * P)], -1).reshape(-1, (k + 1) * P)),
                             E[:, k, 1:].reshape(-1, P)) for k in range(1, c)]
            out['lag1_k>=1'] = [(one(E[:, k, :-1].reshape(-1, P)), E[:, k, 1:].reshape(-1, P)) for k in range(1, c)]
            return out
        if feats_fn is not None:
            feats = feats_fn
        def stats(loader):
            acc = {}
            for bx, _, _, _ in loader:
                if E_fn is None:
                    f, y = m.aux_pairs(bx.float().to(self.device))
                    E = (y - m.aux(f)).reshape(bx.shape[0], c, m.n_patch - 1, m.patch_len).double()
                else:
                    E = E_fn(bx.float().to(self.device))
                for name, pairs in feats(E).items():
                    lst = acc.setdefault(name, [None] * len(pairs))
                    for i, (X, Y) in enumerate(pairs):
                        st = (X.T @ X, X.T @ Y, (Y * Y).sum(0), Y.sum(0), Y.shape[0])
                        lst[i] = st if lst[i] is None else tuple(u + v for u, v in zip(lst[i], st))
            return acc
        S = {n: stats(l) for n, l in (('train', train_loader), ('val', vali_loader), ('test', test_loader))}
        for name in S['train']:
            Ws = []
            for XtX, XtY, *_ in S['train'][name]:
                Ws.append(torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device), XtY))
            line = []
            for split in ('train', 'val', 'test'):
                sse = sst = 0.0
                for W, (XtX, XtY, YY, Ys, n) in zip(Ws, S[split][name]):
                    sse += (YY.sum() - 2 * (W * XtY).sum() + (W * (XtX @ W)).sum()).item()
                    sst += (YY.sum() - (Ys * Ys).sum() / n).item()
                line.append(f'{split} R2 {1 - sse / sst:+.4f} (innov var {sst / sum(st[4] for st in S[split][name]) / m.patch_len:.4f})')
            print(f'[innov {tag}] {name:10s} ' + ' | '.join(line), flush=True)

    @torch.no_grad()
    def carry_diag(self, train_loader, vali_loader, test_loader, lam=1e-4):
        """Carry-over headroom. E[k, i] = x_{k,i+1} - ro(h0_{k,i}) (state-orthogonal innovation of the frozen A0, i = 0..13). The prior state h-_{t+1}
        (what the head reads) predicts x_{t+2} with error E[t+1]; corrections carried into it used E[<= t-1]. Ridge R^2 of E[i+m] on stale
        innovations: m = 1 is the fresh (same-step) reference, m = 2 is carry over one step, m = 3 over two steps. own: same variable; all: every
        variable at the stale time (lagged cross, also available at forecast time); per target variable k."""
        m = self.model; c = m.n_vars
        def E_fn(bx):
            h, _, xp = m.scan_states(bx)
            return (xp[:, :, 1:] - m.ro(h[:, :, :-1])).double()
        def feats(E):
            b, _, T, P = E.shape
            one = lambda X: torch.cat([X, torch.ones_like(X[..., :1])], -1)
            out = {}
            for mm in (1, 2, 3):
                out[f'm{mm} own'] = [(one(E[:, :, :T - mm].reshape(-1, P)), E[:, :, mm:].reshape(-1, P))]
                out[f'm{mm} all'] = [(one(E[:, :, :T - mm].permute(0, 2, 1, 3).reshape(-1, c * P)), E[:, k, mm:].reshape(-1, P)) for k in range(c)]
            out['m2 own+lag'] = [(one(torch.cat([E[:, :, 1:T - 2], E[:, :, :T - 3]], -1).reshape(-1, 2 * P)), E[:, :, 3:].reshape(-1, P))]
            return out
        self.innov_diag(train_loader, vali_loader, test_loader, lam=lam, E_fn=E_fn, tag='carry', feats_fn=feats)

    def retrieval_learn(self, train_data, vali_data, test_data, R=2, dk=None, lr=1e-3, epochs=15, patience=3, bs=8, trstride=12, evstride=2,
                         tau0=0.1, steps_cap=150, key_type='o', group=1):
        """Trainable Q/K retrieval on top of the fixed-cosine headroom test (retrieval_diag): frozen A0, memory = same variable, in-window
        matured positions + the R preceding non-overlapping windows (the 'own-ext' scope, the best fixed-cosine setting on Weather).
        key_type='o': key/query = token output (d_model-dim), single softmax, Wq/Wk (linear, no bias) + a learnable temperature + output
        scale beta. key_type='h32': key/query = the last-layer SSM state, grouped into d_inner/group heads of group*d_state dims each (group=1:
        one head per channel, the original narrow-slice version; group>1: combine `group` channels per head so each head sees a wider slice
        of the situation, closer to what the token output o already integrates). Shared Wq/Wk per head, ONE SOFTMAX PER HEAD over the same
        matured values, the per-head weighted-average predictions linearly mixed (trainable, per-head scalar weights) into the final
        prediction. Either way: regresses the next unobserved error E[k,i] (i >= 1) from matured E[k,j]; reports R2 (1 - SSE / sum(E^2)) on
        train/val/test, directly comparable to retrieval_diag's fixed-cosine numbers."""
        assert key_type in ('o', 'h32')
        m = self.model; m.eval(); c, l, P, Lw = m.n_vars, m.n_patch, m.patch_len, self.args.seq_len; T = l - 1
        mixl = m.base.encoder.layers[-1].mixer; d_inner, d_state = mixl.d_inner, mixl.d_state
        if key_type == 'o':
            dmodel = m.base.value_embedding[1].out_features; dk = dk or dmodel; nh = 1
        else:
            assert d_inner % group == 0
            dmodel = group * d_state; dk = dk or dmodel; nh = d_inner // group
        Wq = torch.nn.Linear(dmodel, dk, bias=False).to(self.device); Wk = torch.nn.Linear(dmodel, dk, bias=False).to(self.device)
        log_tau = torch.nn.Parameter(torch.tensor(float(np.log(tau0)), device=self.device))
        if key_type == 'o':
            beta = torch.nn.Parameter(torch.tensor(1.0, device=self.device)); mix = None
            params = list(Wq.parameters()) + list(Wk.parameters()) + [log_tau, beta]
        else:
            mix = torch.nn.Parameter(torch.full((nh,), 1.0 / nh, device=self.device)); beta = None
            params = list(Wq.parameters()) + list(Wk.parameters()) + [log_tau, mix]
        opt = torch.optim.Adam(params, lr=lr)
        k_q = torch.arange(c, device=self.device).repeat_interleave(T); i_q = torch.arange(T, device=self.device).repeat(c)
        k_m = k_q.repeat(R + 1); j_m = i_q.repeat(R + 1); r_m = torch.arange(R + 1, device=self.device).repeat_interleave(c * T)
        mask = (k_q[:, None] == k_m[None]) & (((j_m[None] <= i_q[:, None] - 1)) & (r_m == 0)[None] | (r_m != 0)[None])
        qsel = (i_q >= 1)

        def get_batch(ds, idx_pool, bsz, rng):
            ib = rng.choice(idx_pool, size=bsz, replace=len(idx_pool) < bsz)
            data = torch.as_tensor(ds.data_x, dtype=torch.float32)
            wins = torch.stack([data[i - r * Lw: i - r * Lw + Lw] for r in range(R + 1) for i in ib]).to(self.device)
            with torch.no_grad():
                h, o, xp = m.scan_states(wins)
                E = (xp[:, :, 1:] - m.ro(h[:, :, :-1])).float()
                rep = o if key_type == 'o' else h.reshape(*h.shape[:-1], nh, group * d_state)
            b = bsz
            trail = tuple(rep.shape[3:])                                             # (dmodel,) for 'o', (d_inner, d_state) for 'h32'
            Rq = rep[:b, :, :-1].flatten(1, 2).float()
            perm = (1, 0, 2) + tuple(range(3, 3 + len(trail)))
            Rm = rep[:, :, :-1].reshape(R + 1, b, c * T, *trail).permute(*perm).reshape(b, -1, *trail).float()
            Eq = E[:b].flatten(1, 2); Em = E.reshape(R + 1, b, c * T, P).permute(1, 0, 2, 3).reshape(b, -1, P)
            return Rq, Rm, Eq, Em

        def step(Rq, Rm, Eq, Em):
            q = Wq(Rq); k = Wk(Rm); tau = log_tau.exp().reciprocal()
            if key_type == 'o':
                S = torch.einsum('bqd,bmd->bqm', q, k) * tau
                w = torch.softmax(S.masked_fill(~mask[None], float('-inf')), -1)
                pred = beta * torch.einsum('bqm,bmp->bqp', w, Em)
            else:
                S = torch.einsum('bqhd,bmhd->bhqm', q, k) * tau                       # one softmax per channel/head
                w = torch.softmax(S.masked_fill(~mask[None, None], float('-inf')), -1)
                per_head = torch.einsum('bhqm,bmp->bhqp', w, Em)                     # b h q P, same value, head-specific weights
                pred = torch.einsum('h,bhqp->bqp', mix, per_head)                    # trainable linear mix over heads
            tgt = Eq[:, qsel]; pr = pred[:, qsel]
            return (pr - tgt).square().sum(), tgt.square().sum()

        def evaluate(ds, idx_pool, n_batches):
            rng = np.random.default_rng(0); sse = sst = 0.0
            with torch.no_grad():
                for _ in range(n_batches):
                    s, t = step(*get_batch(ds, idx_pool, bs, rng)); sse += s.item(); sst += t.item()
            return 1 - sse / sst

        tr_idx = np.arange(R * Lw, len(train_data))[::trstride]; va_idx = np.arange(R * Lw, len(vali_data))[::evstride]; te_idx = np.arange(R * Lw, len(test_data))[::evstride]
        rng = np.random.default_rng(1); best_val = -1e9; best_state = None; bad = 0
        steps_per_epoch = min(max(1, len(tr_idx) // bs), steps_cap)
        for ep in range(epochs):
            for _ in range(steps_per_epoch):
                opt.zero_grad(); s, t = step(*get_batch(train_data, tr_idx, bs, rng)); (s / t).backward(); opt.step()
            val_r2 = evaluate(vali_data, va_idx, max(1, len(va_idx) // bs))
            extra = f'beta {beta.item():.3f}' if beta is not None else f'mix {np.round(mix.detach().cpu().numpy(), 3).tolist()}'
            print(f'[retr-learn] epoch {ep + 1} val R2 {val_r2:+.4f} tau {log_tau.exp().item():.3f} {extra}', flush=True)
            if val_r2 > best_val:
                scale = (beta if beta is not None else mix).detach().clone()
                best_val = val_r2; best_state = ({k: v.clone() for k, v in Wq.state_dict().items()}, {k: v.clone() for k, v in Wk.state_dict().items()}, log_tau.item(), scale); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print('early stop', flush=True); break
        Wq.load_state_dict(best_state[0]); Wk.load_state_dict(best_state[1]); log_tau.data.fill_(best_state[2])
        (beta if beta is not None else mix).data.copy_(best_state[3])
        for name, ds, idx in (('train', train_data, tr_idx), ('val', vali_data, va_idx), ('test', test_data, te_idx)):
            print(f'[retr-learn FINAL] {name} R2 {evaluate(ds, idx, max(1, len(idx) // bs)):+.4f}', flush=True)

        dump_path = os.environ.get('A0C_RDUMP', '')
        if dump_path:
            import json
            with torch.no_grad():
                Rq, Rm, Eq, Em = get_batch(vali_data, va_idx, bs, np.random.default_rng(0))
                q = Wq(Rq); k = Wk(Rm); tau = log_tau.exp().reciprocal()
                if key_type == 'h32':
                    S = torch.einsum('bqhd,bmhd->bhqm', q, k) * tau
                    w = torch.softmax(S.masked_fill(~mask[None, None], float('-inf')), -1)
                else:
                    w = torch.softmax(torch.einsum('bqd,bmd->bqm', q, k).masked_fill(~mask[None], float('-inf')) * tau, -1)[:, None]
                valid_q = qsel.nonzero().flatten()
                pick = []
                for kk in range(c):
                    cand = valid_q[k_q[valid_q] == kk]
                    if len(cand):
                        pick.append(cand[0].item())
                    if len(pick) >= 6:
                        break
                sample = 0
                data = {'key_type': key_type, 'c': c, 'T': T, 'R': R, 'nh': nh,
                        'mix': (mix.detach().cpu().numpy().tolist() if mix is not None else None),
                        'mask_row': {str(qi): mask[qi].cpu().numpy().tolist() for qi in pick},
                        'k_q_of_memory': k_m.cpu().numpy().tolist(), 'j_m_of_memory': j_m.cpu().numpy().tolist(), 'r_m_of_memory': r_m.cpu().numpy().tolist(),
                        'picked_queries': {str(qi): {'k': int(k_q[qi]), 't': int(i_q[qi]), 'weights': w[sample, :, qi].cpu().numpy().tolist()} for qi in pick}}
            with open(dump_path, 'w') as f:
                json.dump(data, f)
            print(f'[retr-learn] dumped attention weights for {len(pick)} queries to {dump_path}', flush=True)

    @torch.no_grad()
    def retrieval_diag(self, sets, R=2, taus=(0.01, 0.03, 0.1, 0.3, 1.0, float('inf')), bs=48):
        """Training-free in-context retrieval headroom on the frozen A0. E[k,i] = x_{k,i+1} - ro(h0_{k,i}) (i = 0..13), key at (k,i) describes the
        situation when predicting x_{k,i+1}. Query (k,i), i >= 1, predicts E[k,i] (the next unobserved error, the analogue of forecast patch 15) by a
        softmax(cos/tau)-weighted average of matured memory values E[k',j]: in-window j <= i-1 and every pair of the R preceding non-overlapping windows.
        Keys: h (last-layer state, 512-d), h32 (per channel = 32 heads x 16-d state, one softmax per head, head outputs mixed by least squares fitted on
        train), o (token output), x (the patch). Scope: own (same variable) / all variables, in-window only / + R earlier windows. shuf: memory taken
        from another random window (control for sequence specificity). The output scale (and head mix) is fitted on train; tau picked by train SSE.
        Reported: MSE reduction vs A0 (predicting zero error) on val / test. Reference: global ridge E[k,i-1] -> E[k,i] on the same queries."""
        m = self.model; m.eval(); c, l, P, Lw = m.n_vars, m.n_patch, m.patch_len, self.args.seq_len
        mix = m.base.encoder.layers[-1].mixer; nh = mix.d_inner; T = l - 1
        k_q = torch.arange(c, device=self.device).repeat_interleave(T); i_q = torch.arange(T, device=self.device).repeat(c)
        k_m = k_q.repeat(R + 1); j_m = i_q.repeat(R + 1); r_m = torch.arange(R + 1, device=self.device).repeat_interleave(c * T)
        same, past, cur = k_q[:, None] == k_m[None], j_m[None] <= i_q[:, None] - 1, (r_m == 0)[None]
        masks = {'own-in': same & past & cur, 'all-in': past & cur, 'own-ext': same & ((past & cur) | ~cur), 'all-ext': (past & cur) | ~cur}
        qsel = (i_q >= 1)
        def window_feats(x):
            h, o, xp = m.scan_states(x)
            E = (xp[:, :, 1:] - m.ro(h[:, :, :-1])).double()
            keys = {'h': h[:, :, :-1], 'o': o[:, :, :-1], 'x': xp[:, :, :-1]}
            keys = {n: F.normalize(v.flatten(1, 2).float(), dim=-1) for n, v in keys.items()}
            keys['h32'] = F.normalize(h[:, :, :-1].flatten(1, 2).reshape(x.shape[0], c * T, nh, -1).float(), dim=-1)
            return E.flatten(1, 2), keys
        acc = {}
        def add(key, st):
            acc[key] = st if key not in acc else tuple(u + v for u, v in zip(acc[key], st))
        rng = np.random.default_rng(0)
        for split, ds in sets:
            data = torch.as_tensor(ds.data_x, dtype=torch.float32)
            stride = int(os.environ.get('A0C_TRSTRIDE', '2')) if split == 'train' else int(os.environ.get('A0C_EVSTRIDE', '1'))
            idx = rng.permutation(np.arange(R * Lw, len(ds))[::stride])
            for st in range(0, len(idx), bs):
                ib = idx[st:st + bs]; b = len(ib)
                wins = torch.stack([data[i - r * Lw: i - r * Lw + Lw] for r in range(R + 1) for i in ib]).to(self.device)   # (R+1)*b, Lw, c; r-major
                E, keys = window_feats(wins)
                Eq = E[:b]                                                              # b, c*T, P (query windows)
                Em = E.reshape(R + 1, b, c * T, P).permute(1, 0, 2, 3).reshape(b, -1, P)   # memory values, order (r, k, i)
                tgt = Eq[:, qsel]; EE = tgt.square().sum().item()
                Xr = torch.cat([Eq.reshape(b, c, T, P)[:, :, :-1], torch.ones_like(Eq.reshape(b, c, T, P)[:, :, :-1, :1])], -1).reshape(-1, P + 1)
                Yr = Eq.reshape(b, c, T, P)[:, :, 1:].reshape(-1, P)
                add((split, 'ridge'), (Xr.T @ Xr, Xr.T @ Yr, Yr.square().sum().item()))
                for kn, K in keys.items():
                    multi = kn == 'h32'
                    Kq = K[:b]
                    Km = K.reshape(R + 1, b, *K.shape[1:]).transpose(0, 1).reshape(b, -1, *K.shape[2:])
                    for ctl in ('', 'shuf'):
                        Kmm, Emm = (Km, Em) if not ctl else (Km.roll(1, 0), Em.roll(1, 0))
                        S = torch.einsum('bqhd,bmhd->bhqm', Kq, Kmm) if multi else torch.einsum('bqd,bmd->bqm', Kq, Kmm)[:, None]
                        for sc, M in masks.items():
                            if ctl and sc.endswith('-in'):
                                continue
                            for tau in taus:
                                logits = (S / tau if tau != float('inf') else torch.zeros_like(S)).masked_fill(~M, float('-inf'))
                                w = torch.softmax(logits, -1).nan_to_num(0.0).double()
                                pr = torch.einsum('bhqm,bmp->bhqp', w, Emm)[:, :, qsel]          # b, H, Q, P
                                G = torch.einsum('bhqp,bgqp->hg', pr, pr); g = torch.einsum('bhqp,bqp->h', pr, tgt)
                                add((split, kn, sc + ('-' + ctl if ctl else ''), tau), (G, g, EE))
        out = []
        for key in [k for k in acc if k[0] == 'train']:
            if key[1] == 'ridge':
                XtX, XtY, _ = acc[key]
                W = torch.linalg.solve(XtX + 1e-4 * XtX.diagonal().mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device), XtY)
                red = {}
                for split in ('train', 'val', 'test'):
                    A, B, YY = acc[(split, 'ridge')]
                    red[split] = 1 - (YY - 2 * (W * B).sum().item() + (W * (A @ W)).sum().item()) / YY
                print(f'[retr] global ridge E[i-1]->E[i] own        train {red["train"]:+.4f} | val {red["val"]:+.4f} | test {red["test"]:+.4f}', flush=True)
                continue
            G, g, EE = acc[key]
            beta = torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), g)
            red = {}
            for split in ('train', 'val', 'test'):
                Gs, gs, EEs = acc[(split,) + key[1:]]
                red[split] = 1 - (EEs - 2 * (beta @ gs).item() + (beta @ Gs @ beta).item()) / EEs
            out.append((key[1], key[2], key[3], red))
        for kn in ('h', 'h32', 'o', 'x'):
            for sc in ('own-in', 'all-in', 'own-ext', 'all-ext', 'own-ext-shuf', 'all-ext-shuf'):
                rows = [r for r in out if r[0] == kn and r[1] == sc]
                if not rows:
                    continue
                best = max(rows, key=lambda r: r[3]['train'])
                print(f'[retr] key {kn:4s} {sc:13s} tau {best[2]:<5} train {best[3]["train"]:+.4f} | val {best[3]["val"]:+.4f} | test {best[3]["test"]:+.4f} | '
                      f'uniform val {[r for r in rows if r[2] == float("inf")][0][3]["val"]:+.4f}', flush=True)

    @torch.no_grad()
    def innov_state_diag(self, train_loader, vali_loader, test_loader, lam=1e-4):
        """Innovation relative to the full last-layer state: e*_{k,t} = x_{k,t} - ridge(h_{k,t-1}), t = 1..14 (h: 512-d, fitted on train).
        Prints the next-patch MSE of three readouts (aux(o) 16-d, ridge(h), ridge([h, o])) and reruns the lag / cross R^2 diagnostic on e*."""
        m = self.model; m.eval(); c, P = m.n_vars, m.patch_len
        def pairs(bx):
            h, o, xp = m.scan_states(bx)
            one = lambda X: torch.cat([X, torch.ones_like(X[..., :1])], -1)
            fh = one(h[:, :, :-1]).double(); fho = one(torch.cat([h[:, :, :-1], o[:, :, :-1]], -1)).double()
            return fh, fho, o[:, :, :-1], xp[:, :, 1:].double()
        acc = {}
        for bx, _, _, _ in train_loader:
            fh, fho, _, y = pairs(bx.float().to(self.device))
            for k, X in (('h', fh), ('ho', fho)):
                X2, Y2 = X.reshape(-1, X.shape[-1]), y.reshape(-1, P)
                st = (X2.T @ X2, X2.T @ Y2)
                acc[k] = st if k not in acc else (acc[k][0] + st[0], acc[k][1] + st[1])
        W = {k: torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device), XtY)
             for k, (XtX, XtY) in acc.items()}
        for name, loader in (('train', train_loader), ('val', vali_loader), ('test', test_loader)):
            se = np.zeros(3); n = 0
            for bx, _, _, _ in loader:
                fh, fho, o, y = pairs(bx.float().to(self.device))
                for i, p in enumerate((m.aux(o).double(), fh @ W['h'], fho @ W['ho'])):
                    se[i] += (p - y).square().sum().item()
                n += y.numel()
            print(f'[readout] {name:5s} next-patch MSE: aux(o) 16-d {se[0] / n:.4f} | ridge(h) 512-d {se[1] / n:.4f} | ridge(h,o) {se[2] / n:.4f}', flush=True)
        def E_fn(bx):
            fh, _, _, y = pairs(bx)
            return y - fh @ W['h']
        self.innov_diag(train_loader, vali_loader, test_loader, lam=lam, E_fn=E_fn, tag='state h')

    @torch.no_grad()
    def final_resid_diag(self, train_loader, vali_loader, test_loader, lam=1e-3):
        """Headroom on the actual objective: R^2 of the A0 final forecast residual r_k = y_k - yhat_k (96-d, per variable)
        (data scale) regressed (ridge, per scan variable k, fitted on train) on (e* also in the data scale): own = e*_{k,13..14}; all = e*_{j,13..14} of every variable;
        state = final last-layer state h_{k,14} (512-d, reference: what the frozen state still holds). e*_{k,t} = x_{k,t} - ro(h_{k,t-1})."""
        m = self.model; m.eval(); c, l, P = m.n_vars, m.n_patch, m.patch_len; H = self.args.pred_len
        def batch(bx, by):
            h, _, xp = m.scan_states(bx)
            es = xp[:, :, 1:] - m.ro(h[:, :, :-1])                                        # e*_{k,t}, t = 1..l-1
            _, _, stdev = m.norm(bx)
            m.use_corr = False; pred = m(bx); m.use_corr = True
            r = (by - pred)[:, :, m.order].permute(0, 2, 1).double()                    # b c H, scan order, data scale (the official metric's scale)
            sd = stdev[:, 0, m.order][:, :, None, None]                                  # back to the data scale: per-window instance stdev
            last = (es[:, :, -2:] * sd).reshape(bx.shape[0], c, 2 * P).double()
            one = lambda X: torch.cat([X, torch.ones_like(X[..., :1])], -1)
            return {'own': [(one(last[:, k]), r[:, k]) for k in range(c)],
                    'all': [(one(last.reshape(bx.shape[0], -1)), r[:, k]) for k in range(c)],
                    'state': [(one(h[:, k, -1].double()), r[:, k]) for k in range(c)]}
        S = {}
        for split, loader in (('train', train_loader), ('val', vali_loader), ('test', test_loader)):
            acc = {}
            for bx, by, _, _ in loader:
                for name, pairs in batch(bx.float().to(self.device), by.float().to(self.device)[:, -H:]).items():
                    lst = acc.setdefault(name, [None] * len(pairs))
                    for i, (X, Y) in enumerate(pairs):
                        st = (X.T @ X, X.T @ Y, (Y * Y).sum(0), Y.sum(0), Y.shape[0])
                        lst[i] = st if lst[i] is None else tuple(u + v for u, v in zip(lst[i], st))
            S[split] = acc
        for name in S['train']:
            Ws = [torch.linalg.solve(XtX + lam * XtX.diagonal().mean() * torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device), XtY)
                  for XtX, XtY, *_ in S['train'][name]]
            line = []
            for split in ('train', 'val', 'test'):
                sse = sst = 0.0
                for W, (XtX, XtY, YY, Ys, n) in zip(Ws, S[split][name]):
                    sse += (YY.sum() - 2 * (W * XtY).sum() + (W * (XtX @ W)).sum()).item(); sst += (YY.sum() - (Ys * Ys).sum() / n).item()
                line.append(f'{split} R2 {1 - sse / sst:+.4f}')
            print(f'[final resid] {name:6s} ' + ' | '.join(line), flush=True)

    @torch.no_grad()
    def mem_audit(self, vali_data, vali_loader, test_data, bs=32):
        """Same checkpoint, evaluation-only interventions on the retrieval memory, fixed sample order, official target (seq_y[-H:]),
        dataset-scaled MSE/MAE like the official vali/test:
          real      -- own history, true key-value pairing
          perm      -- own history, deterministic time permutation (M2's roll): same values, wrong pairing
          unrelated -- another window's history (keys + values written while scanning window (i + N//2) mod N, which shares no time
                       step with window i or its target since N//2 >= seq_len + pred_len): the null for 'memory content matters'
          zero      -- g = 0 (memory removed)      uniform -- a = 1/M (no selection)
        Plus mean attention entropy, gate and |g m| / |C| on val."""
        m = self.model; a = self.args; m.eval(); saved = m.mem; H = a.pred_len
        for name, ds in (('val', vali_data), ('test', test_data)):
            N = len(ds)
            assert N // 2 >= a.seq_len + a.pred_len, f'{name}: {N} windows too few for a non-overlapping donor'
            get = lambda ids, j: torch.as_tensor(np.stack([ds[i][j] for i in ids]), dtype=torch.float32, device=self.device)
            row = []
            for mode in ('real', 'perm', 'unrelated', 'zero', 'uniform'):
                m.mem = 'shuf' if mode == 'perm' else 'real'
                m.mem_net.audit = mode if mode in ('zero', 'uniform') else None
                se = ae = cnt = 0.0
                for s0 in range(0, N, bs):
                    ids = list(range(s0, min(N, s0 + bs)))
                    bank = m.donor_bank(get([(i + N // 2) % N for i in ids], 0)) if mode == 'unrelated' else None
                    err = m.run_cross(get(ids, 0), mem_bank=bank)['pred'] - get(ids, 1)[:, -H:]
                    se += err.square().sum().item(); ae += err.abs().sum().item(); cnt += err.numel()
                row.append(f'{mode} {se / cnt:.6f}/{ae / cnt:.6f}')
            print(f'[mem audit] trained as {saved} | {name} MSE/MAE: ' + ' | '.join(row), flush=True)
        m.mem = saved; m.mem_net.audit = None; st = {}
        for i, (bx, _, _, _) in enumerate(vali_loader):
            for n, v in m.run_cross(bx.float().to(self.device))['mem_stats'].items():
                st.setdefault(n, []).append(v)
            if i >= 49:
                break
        print(f'[mem stats val] ' + ' | '.join(f'{n} {np.mean(v):.4f}' for n, v in st.items()), flush=True)

    def train(self, setting):
        a = self.args
        train_data, train_loader = self._get_data('train'); vali_data, vali_loader = self._get_data('val'); test_data, test_loader = self._get_data('test')
        crit = self._select_criterion(); m = self.model
        assert not a.use_amp and a.features == 'M'
        m.use_corr = False
        t0 = time.time()
        if m.freeze:
            print(f'A0 frozen baseline (fixed order): Vali Loss: {self.vali(vali_data, vali_loader, crit):.7f} Test Loss: {self.vali(test_data, test_loader, crit):.7f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        else:
            print('from scratch: skipping the (meaningless, randomly-initialised) pre-training baseline eval', flush=True)
        self.fit_aux(train_loader, vali_loader)
        if m.mode == 'cross':
            self.fit_ro(train_loader)                                      # ridge warm-start for ro; it then keeps training by gradient if not frozen
        if os.environ.get('A0C_INNOV') == '6':
            self.fit_ro(train_loader); self.retrieval_learn(train_data, vali_data, test_data, dk=int(os.environ.get('A0C_RKDIM', '0')) or None, lr=float(os.environ.get('A0C_RLR', '1e-3')), epochs=int(os.environ.get('A0C_REPOCHS', '15')), patience=int(os.environ.get('A0C_RPATIENCE', '3')), bs=int(os.environ.get('A0C_RBS', '8')), trstride=int(os.environ.get('A0C_TRSTRIDE', '12')), evstride=int(os.environ.get('A0C_EVSTRIDE', '2')), tau0=float(os.environ.get('A0C_RTAU0', '0.1')), steps_cap=int(os.environ.get('A0C_RSTEPS', '150')), key_type=os.environ.get('A0C_RKEY', 'o'), group=int(os.environ.get('A0C_RGROUP', '1'))); sys.exit(0)
        if os.environ.get('A0C_INNOV') == '5':
            self.fit_ro(train_loader); self.retrieval_diag([('train', train_data), ('val', vali_data), ('test', test_data)], bs=int(os.environ.get('A0C_RBS', '48'))); sys.exit(0)
        if os.environ.get('A0C_INNOV') == '4':
            self.fit_ro(train_loader); self.carry_diag(train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_INNOV') == '3':
            self.fit_ro(train_loader); self.final_resid_diag(train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_INNOV') == '2':
            self.innov_state_diag(train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_INNOV'):
            self.innov_diag(train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG7'):
            self.path_lag1_full(os.environ['A0C_DIAG7'], train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG6'):
            self.path_lag1(os.environ['A0C_DIAG6'], train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG5'):
            self.carry_relearn(os.environ['A0C_DIAG5'], train_loader, vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG4'):
            self.carry_decomp(os.environ['A0C_DIAG4'], vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG3'):
            self.diag_carry(os.environ['A0C_DIAG3'], vali_loader, test_loader); sys.exit(0)
        if os.environ.get('A0C_DIAG'):
            self.diag_step(os.environ['A0C_DIAG'], vali_loader, test_loader); sys.exit(0)
        tune = os.environ.get('A0C_TUNE', '')                                             # '' | head: also train the official prediction head (control: A0C_CORR=none)
        assert tune in ('', 'head')
        if m.corr_net is None and not tune:
            return m
        m.use_corr = True
        gamma = float(os.environ.get('A0C_GAMMA', '1e-4')); lam = float(os.environ.get('A0C_STEP', '0'))
        mu = float(os.environ.get('A0C_DECOR', '0'))                                       # weight of the per-channel corr(h-, C)^2 penalty
        lam_m = float(os.environ.get('A0C_STEP_MEM', '0'))                                 # weight of the per-site re-prediction loss from h_mem
        assert m.mem is None or m.mode == 'cross'
        params = list(m.corr_net.parameters()) if m.corr_net is not None else []
        retr_params = list(m.mem_net.parameters()) if m.mem_net is not None else []
        head_params = list(m.base.head.parameters()) if tune == 'head' else []
        for p in head_params:
            p.requires_grad_(True)
        head_factor = float(os.environ.get('A0C_HEAD_LR', '0.1'))
        scratch_factor = float(os.environ.get('A0C_SCRATCH_LR', '1.0'))                # from-scratch only: backbone+ro lr = a.learning_rate * this
        from_scratch_params = (list(m.base.parameters()) + list(m.ro.parameters())) if not m.freeze else []  # scaled down so step loss (shared with corr_net) can't yank backbone as hard as corr_net moves
        print(f'trainable params {sum(p.numel() for p in m.parameters() if p.requires_grad)} (corr_net {sum(p.numel() for p in params)} head {sum(p.numel() for p in head_params)} '
              f'backbone+ro {sum(p.numel() for p in from_scratch_params)}) gamma {gamma} step lambda {lam} tune {tune or "none"} head lr x{head_factor} '
              f'freeze {m.freeze} scratch lr x{scratch_factor} | mem {m.mem} retr {sum(p.numel() for p in retr_params)} step_mem lambda {lam_m}', flush=True)
        path = os.path.join(a.checkpoints, setting); os.makedirs(path, exist_ok=True)
        named_groups = ([('corr', params)] if params else []) + ([('retr', retr_params)] if retr_params else []) + ([('head', head_params)] if head_params else []) + ([('scratch', from_scratch_params)] if from_scratch_params else [])
        groups = [{'params': p} for _, p in named_groups]
        early = EarlyStopping(patience=a.patience, verbose=True); opt = torch.optim.Adam(groups, lr=a.learning_rate)
        idx = {name: i for i, (name, _) in enumerate(named_groups)}                    # adjust_learning_rate resets every group to the same
        def scale_head_lr():                                                           # (decayed) base lr first; re-apply each group's ratio on top of THAT
            if 'head' in idx:
                opt.param_groups[idx['head']]['lr'] *= head_factor
            if 'scratch' in idx:
                opt.param_groups[idx['scratch']]['lr'] *= scratch_factor
        scale_head_lr()
        for epoch in range(a.train_epochs):
            m.train(); t0 = time.time(); tl, tr, tsp, tdc = [], [], [], []
            for i, (bx, by, _, _) in enumerate(train_loader):
                opt.zero_grad()
                out = (m.run_cross if m.mode == 'cross' else m.run)(bx.float().to(self.device))
                mse = crit(out['pred'], by.float().to(self.device)[:, -a.pred_len:]).mean()
                reg = out['reg'] if out['reg'] is not None else mse.new_zeros(()); step = out['step'] if out['step'] is not None else mse.new_zeros(())
                dec = out.get('decor'); dec = dec if dec is not None else mse.new_zeros(())
                smem = out.get('step_mem'); smem = smem if smem is not None else mse.new_zeros(())
                loss = mse + gamma * reg + lam * step + mu * dec + lam_m * smem
                loss.backward(); opt.step()
                tl.append(mse.item()); tr.append(reg.item()); tsp.append(step.item()); tdc.append(dec.item())
                if (i + 1) % 100 == 0: print(f'\titers: {i + 1}, epoch: {epoch + 1} | mse: {mse.item():.7f} reg: {reg.item():.3e} step: {step.item():.4f} decor: {dec.item():.4f} '
                                             f'step_mem: {smem.item():.4f} mem {out.get("mem_stats", {})}', flush=True)
            print(f'Epoch: {epoch + 1} cost time: {time.time() - t0:.0f}')
            vloss = self.vali(vali_data, vali_loader, crit); tloss = self.vali(test_data, test_loader, crit)
            print(f'Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {np.average(tl):.7f} (reg {np.average(tr):.3e} step {np.average(tsp):.4f} decor {np.average(tdc):.4f}) Vali Loss: {vloss:.7f} Test Loss: {tloss:.7f}', flush=True)
            early(vloss, m, path)
            if early.early_stop: print('Early stopping'); break
            adjust_learning_rate(opt, epoch + 1, a); scale_head_lr()
        m.load_state_dict(torch.load(path + '/checkpoint.pth'))
        if m.mem_net is not None:
            self.mem_audit(vali_data, vali_loader, test_data)
        if m.mode == 'cross' and m.corr_net is not None:
            for name, loader in (('val', vali_loader), ('test', test_loader)):
                c1, c0 = self.step_eval(loader)
                print(f'[step {name}] next-patch MSE ro(h): corr {c1:.4f} | A0 {c0:.4f} | rel {c1 / c0 - 1:+.2%}', flush=True)
        return m
