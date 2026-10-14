"""PAT-304 A0-frozen innovation correction: official run.py / data / seed / early stopping / vali / test, model = A0Corr (frozen A0 checkpoint).
Env: A0C_MODE (self = F1/F2 | cross = F3), A0C_STEP (lambda of the per-step innovation loss, default 0), A0C_CKPT (A0 checkpoint.pth, required), A0C_CORR (shared | gated = GatedCorrNet, C(h, e=0) = 0 | none = A0 baseline in the fixed order), A0C_DECOR (weight of the per-channel corr(h-, C)^2 penalty, default 0), A0C_SHUFFLE=1 (residual rolled in batch),
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
        base.load_state_dict(torch.load(os.environ['A0C_CKPT'], map_location='cpu')); base.to(self.device)
        enc = base.encoder.eval(); enc.ids_shuffle = None
        set_seed(a.seed)                                                   # same as run.py before the official test: the first SA solve fixes the order
        with torch.no_grad():
            enc(torch.zeros(1, enc.n_vars * (a.seq_len // a.patch_len), a.d_model, device=self.device))
        order = enc.ids_shuffle[0].tolist()
        corr = os.environ.get('A0C_CORR', 'shared'); corr = None if corr == 'none' else corr
        print(f'A0C: ckpt {os.environ["A0C_CKPT"]} | scan order {order} | corr {corr} shuffle {os.environ.get("A0C_SHUFFLE", "0")}', flush=True)
        return A0Corr(base, order, corr=corr, shuffle=os.environ.get('A0C_SHUFFLE', '0') == '1', mode=os.environ.get('A0C_MODE', 'self')).float()

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
                         tau0=0.1, steps_cap=150):
        """Trainable Q/K retrieval on top of the fixed-cosine headroom test (retrieval_diag): frozen A0, key/query representation = token
        output o, memory = same variable, in-window matured positions + the R preceding non-overlapping windows (the 'own-ext' scope, the
        best fixed-cosine setting on Weather). Only Wq, Wk (linear, no bias), a learnable temperature and an output scale beta are trained,
        regressing the next unobserved error E[k,i] (i >= 1) from a softmax((Wq q)路(Wk k)/tau)-weighted average of matured E[k,j]. Reports R2
        (1 - SSE / sum(E^2), i.e. MSE reduction vs A0) on train/val/test, directly comparable to retrieval_diag's fixed-cosine numbers."""
        m = self.model; m.eval(); c, l, P, Lw = m.n_vars, m.n_patch, m.patch_len, self.args.seq_len; T = l - 1
        dmodel = m.base.value_embedding[1].out_features; dk = dk or dmodel
        Wq = torch.nn.Linear(dmodel, dk, bias=False).to(self.device); Wk = torch.nn.Linear(dmodel, dk, bias=False).to(self.device)
        log_tau = torch.nn.Parameter(torch.tensor(float(np.log(tau0)), device=self.device)); beta = torch.nn.Parameter(torch.tensor(1.0, device=self.device))
        opt = torch.optim.Adam(list(Wq.parameters()) + list(Wk.parameters()) + [log_tau, beta], lr=lr)
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
            b = bsz
            Oq = o[:b, :, :-1].flatten(1, 2).float(); Om = o[:, :, :-1].reshape(R + 1, b, c * T, -1).permute(1, 0, 2, 3).reshape(b, -1, o.shape[-1]).float()
            Eq = E[:b].flatten(1, 2); Em = E.reshape(R + 1, b, c * T, P).permute(1, 0, 2, 3).reshape(b, -1, P)
            return Oq, Om, Eq, Em

        def step(Oq, Om, Eq, Em):
            q = Wq(Oq); k = Wk(Om)
            S = torch.einsum('bqd,bmd->bqm', q, k) * log_tau.exp().reciprocal()
            w = torch.softmax(S.masked_fill(~mask[None], float('-inf')), -1)
            pred = beta * torch.einsum('bqm,bmp->bqp', w, Em)
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
            print(f'[retr-learn] epoch {ep + 1} val R2 {val_r2:+.4f} tau {log_tau.exp().item():.3f} beta {beta.item():.3f}', flush=True)
            if val_r2 > best_val:
                best_val = val_r2; best_state = ({k: v.clone() for k, v in Wq.state_dict().items()}, {k: v.clone() for k, v in Wk.state_dict().items()}, log_tau.item(), beta.item()); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print('early stop', flush=True); break
        Wq.load_state_dict(best_state[0]); Wk.load_state_dict(best_state[1]); log_tau.data.fill_(best_state[2]); beta.data.fill_(best_state[3])
        for name, ds, idx in (('train', train_data, tr_idx), ('val', vali_data, va_idx), ('test', test_data, te_idx)):
            print(f'[retr-learn FINAL] {name} R2 {evaluate(ds, idx, max(1, len(idx) // bs)):+.4f}', flush=True)

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

    def train(self, setting):
        a = self.args
        train_data, train_loader = self._get_data('train'); vali_data, vali_loader = self._get_data('val'); test_data, test_loader = self._get_data('test')
        crit = self._select_criterion(); m = self.model
        assert not a.use_amp and a.features == 'M'
        m.use_corr = False
        t0 = time.time()
        print(f'A0 frozen baseline (fixed order): Vali Loss: {self.vali(vali_data, vali_loader, crit):.7f} Test Loss: {self.vali(test_data, test_loader, crit):.7f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        self.fit_aux(train_loader, vali_loader)
        if m.mode == 'cross':
            self.fit_ro(train_loader)
        if os.environ.get('A0C_INNOV') == '6':
            self.fit_ro(train_loader); self.retrieval_learn(train_data, vali_data, test_data, dk=int(os.environ.get('A0C_RKDIM', '0')) or None, lr=float(os.environ.get('A0C_RLR', '1e-3')), epochs=int(os.environ.get('A0C_REPOCHS', '15')), patience=int(os.environ.get('A0C_RPATIENCE', '3')), bs=int(os.environ.get('A0C_RBS', '8')), trstride=int(os.environ.get('A0C_TRSTRIDE', '12')), evstride=int(os.environ.get('A0C_EVSTRIDE', '2')), tau0=float(os.environ.get('A0C_RTAU0', '0.1')), steps_cap=int(os.environ.get('A0C_RSTEPS', '150'))); sys.exit(0)
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
        params = list(m.corr_net.parameters()) if m.corr_net is not None else []
        head_params = list(m.base.head.parameters()) if tune == 'head' else []
        for p in head_params:
            p.requires_grad_(True)
        head_factor = float(os.environ.get('A0C_HEAD_LR', '0.1'))
        print(f'trainable params {sum(p.numel() for p in m.parameters() if p.requires_grad)} (corr_net {sum(p.numel() for p in params)} head {sum(p.numel() for p in head_params)}) '
              f'gamma {gamma} step lambda {lam} tune {tune or "none"} head lr x{head_factor}', flush=True)
        path = os.path.join(a.checkpoints, setting); os.makedirs(path, exist_ok=True)
        groups = ([{'params': params}] if params else []) + ([{'params': head_params}] if head_params else [])
        early = EarlyStopping(patience=a.patience, verbose=True); opt = torch.optim.Adam(groups, lr=a.learning_rate)
        def scale_head_lr():
            if head_params:
                opt.param_groups[-1]['lr'] *= head_factor
        scale_head_lr()
        for epoch in range(a.train_epochs):
            m.train(); t0 = time.time(); tl, tr, tsp, tdc = [], [], [], []
            for i, (bx, by, _, _) in enumerate(train_loader):
                opt.zero_grad()
                out = (m.run_cross if m.mode == 'cross' else m.run)(bx.float().to(self.device))
                mse = crit(out['pred'], by.float().to(self.device)[:, -a.pred_len:]).mean()
                reg = out['reg'] if out['reg'] is not None else mse.new_zeros(()); step = out['step'] if out['step'] is not None else mse.new_zeros(())
                dec = out.get('decor'); dec = dec if dec is not None else mse.new_zeros(())
                loss = mse + gamma * reg + lam * step + mu * dec
                loss.backward(); opt.step()
                tl.append(mse.item()); tr.append(reg.item()); tsp.append(step.item()); tdc.append(dec.item())
                if (i + 1) % 100 == 0: print(f'\titers: {i + 1}, epoch: {epoch + 1} | mse: {mse.item():.7f} reg: {reg.item():.3e} step: {step.item():.4f} decor: {dec.item():.4f}', flush=True)
            print(f'Epoch: {epoch + 1} cost time: {time.time() - t0:.0f}')
            vloss = self.vali(vali_data, vali_loader, crit); tloss = self.vali(test_data, test_loader, crit)
            print(f'Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {np.average(tl):.7f} (reg {np.average(tr):.3e} step {np.average(tsp):.4f} decor {np.average(tdc):.4f}) Vali Loss: {vloss:.7f} Test Loss: {tloss:.7f}', flush=True)
            early(vloss, m, path)
            if early.early_stop: print('Early stopping'); break
            adjust_learning_rate(opt, epoch + 1, a); scale_head_lr()
        m.load_state_dict(torch.load(path + '/checkpoint.pth'))
        if m.mode == 'cross' and m.corr_net is not None:
            for name, loader in (('val', vali_loader), ('test', test_loader)):
                c1, c0 = self.step_eval(loader)
                print(f'[step {name}] next-patch MSE ro(h): corr {c1:.4f} | A0 {c0:.4f} | rel {c1 / c0 - 1:+.2%}', flush=True)
        return m
