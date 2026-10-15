"""PAT-304 R1 experiment: recurrent MambaTS baseline (patchmem/r1_model.py), random lookback / horizon training, one checkpoint
evaluated on a fixed lookback x horizon grid. Official run.py / data / seed / setting string; run via patchmem/run_r1.py.
Env:
  R1_LMIN / R1_LMAX      training lookback range in time steps, multiples of patch_len (default 384 / 1440)
  R1_HMIN / R1_HMAX      training horizon range in patches (default 2 / 7, i.e. H96..H336 for P=48)
  R1_LAMBDA_HIST         weight of the in-window next-patch auxiliary loss at every history boundary (default 0)
  R1_RO_LN=1             LayerNorm on the readout input only (default off)
  R1_VAL_GRID            early-stopping grid 'L:H,...' (default 384:96,720:96,1440:96,384:336,720:336,1440:336), mean MSE
  R1_VAL_STRIDE          window stride for the per-epoch validation (default 4)
  R1_TEST_L / R1_TEST_H  test grid (default 384,720,1440 / 96,192,336)
  R1_EVAL_BS             evaluation batch size (default 64)
  R1_MAX_ITERS           stop each training epoch after this many iterations (smoke runs; default 0 = full epoch)
The training loss is the forecast MSE over the sampled horizon (+ lambda_hist * L_hist). L_hist is in-window auxiliary supervision:
the lookback's RevIN statistics already include x_(t+1), so it is NOT an online per-origin evaluation."""
import copy, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from patchmem.r1_model import R1
from utils.tools import EarlyStopping, adjust_learning_rate


def _grid(s):
    return [tuple(int(v) for v in item.split(':')) for item in s.split(',')]


class Exp_R1(Exp_Long_Term_Forecast):
    def _build_model(self):
        a = self.args
        m = R1(a, readout_ln=os.environ.get('R1_RO_LN', '0') == '1').float()
        nro = sum(p.numel() for p in m.readout.parameters())
        print(f'R1: d_inner {m.D} d_state {m.N} vars {m.c} patch {m.P} | readout {m.D * m.N} -> {m.c * m.P} ({nro} params) | '
              f'readout LN {m.ro_ln is not None} | VPT {m.vpt} ({a.ATSP_solver}) | trainable {sum(p.numel() for p in m.parameters() if p.requires_grad)}',
              flush=True)
        return m

    def _ds(self, flag, L, H):
        a = copy.copy(self.args)
        a.seq_len, a.pred_len = L, H
        return data_provider(a, flag)[0]

    @staticmethod
    def _batch(ds, ids):
        items = [ds[i] for i in ids]
        return (torch.as_tensor(np.stack([it[0] for it in items]), dtype=torch.float32),
                torch.as_tensor(np.stack([it[1] for it in items]), dtype=torch.float32))

    @torch.no_grad()
    def eval_grid(self, flag, grid, stride=1, naive=False, per_var=None, state_stats=False):
        """Mean MSE/MAE of the forecast per (L, H) on the official `flag` split built with seq_len=L, pred_len=H (every `stride`-th window).
        naive: also last-patch-repeat and per-variable least-squares line over the last patch. per_var: (L, H) for a per-variable report
        in scan position order. state_stats: RMS of the final boundary state per lookback."""
        m = self.model
        m.eval()
        bs = int(os.environ.get('R1_EVAL_BS', '64'))
        P, res = m.P, {}
        for L, H in grid:
            ds = self._ds(flag, L, H)
            ids = list(range(0, len(ds), stride))
            acc = dict(mse=0.0, mae=0.0, last=0.0, lin=0.0, n=0)
            var_se = torch.zeros(m.c)
            rms = []
            for s0 in range(0, len(ids), bs):
                bx, by = self._batch(ds, ids[s0:s0 + bs])
                bx, y = bx.to(self.device), by[:, -H:].to(self.device)
                tr = {} if state_stats else None
                pred = m(bx, H // P, trace=tr)['pred']
                err = pred - y
                acc['mse'] += err.square().sum().item(); acc['mae'] += err.abs().sum().item(); acc['n'] += err.numel()
                var_se += err.square().sum((0, 1)).cpu()
                if naive:
                    last = bx[:, -P:]
                    acc['last'] += (last.repeat(1, H // P, 1) - y).square().sum().item()
                    tt = torch.arange(P, dtype=torch.float32, device=bx.device)
                    tc = tt - tt.mean()
                    slope = (tc[None, :, None] * (last - last.mean(1, keepdim=True))).sum(1, keepdim=True) / tc.square().sum()
                    fut = torch.arange(P, P + H, dtype=torch.float32, device=bx.device) - tt.mean()
                    acc['lin'] += (last.mean(1, keepdim=True) + slope * fut[None, :, None] - y).square().sum().item()
                if state_stats:
                    rms.append(tr['h_last'].flatten(1).square().mean(1).sqrt().cpu())
            n = acc['n']
            res[(L, H)] = dict(mse=acc['mse'] / n, mae=acc['mae'] / n)
            if naive:
                res[(L, H)].update(last=acc['last'] / n, lin=acc['lin'] / n)
            if per_var == (L, H):
                vm = (var_se / (n / m.c)).tolist()
                order = m.ids_shuffle if m.vpt else list(range(m.c))
                print(f'[R1 {flag} per-variable L={L} H={H}] scan order {order}: ' +
                      ' '.join(f'pos{k}:v{v}={vm[v]:.4f}' for k, v in enumerate(order)), flush=True)
            if state_stats:
                r = torch.cat(rms)
                print(f'[R1 {flag} state L={L}] |h_T| RMS mean {r.mean():.4f} q10/50/90 {r.quantile(.1):.4f}/{r.quantile(.5):.4f}/{r.quantile(.9):.4f} '
                      f'max {r.max():.4f}', flush=True)
        return res

    def train(self, setting):
        a, m = self.args, self.model
        P = m.P
        Lmin, Lmax = int(os.environ.get('R1_LMIN', '384')), int(os.environ.get('R1_LMAX', '1440'))
        Hmin, Hmax = int(os.environ.get('R1_HMIN', '2')), int(os.environ.get('R1_HMAX', '7'))
        lam = float(os.environ.get('R1_LAMBDA_HIST', '0'))
        assert Lmin % P == 0 and Lmax % P == 0 and Lmin <= Lmax and 1 <= Hmin <= Hmax
        val_grid = _grid(os.environ.get('R1_VAL_GRID', '384:96,720:96,1440:96,384:336,720:336,1440:336'))
        vstride = int(os.environ.get('R1_VAL_STRIDE', '4'))
        max_it = int(os.environ.get('R1_MAX_ITERS', '0'))
        tr_ds = self._ds('train', Lmax, Hmax * P)
        loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.num_workers)
        print(f'R1 train: {len(tr_ds)} windows of {Lmax}+{Hmax * P} | lookback {Lmin}..{Lmax} step {P} | horizon {Hmin}..{Hmax} patches | '
              f'lambda_hist {lam} | val grid {val_grid} stride {vstride}', flush=True)
        rng = np.random.default_rng(a.seed)
        path = os.path.join(a.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        opt = torch.optim.Adam(m.parameters(), lr=a.learning_rate)
        early = EarlyStopping(patience=a.patience, verbose=True)
        for epoch in range(a.train_epochs):
            m.train()
            t0 = time.time()
            lf, lh = [], []
            for i, (bx, by, _, _) in enumerate(loader):
                L = int(rng.choice(np.arange(Lmin, Lmax + 1, P)))
                nH = int(rng.integers(Hmin, Hmax + 1))
                x = bx[:, -L:].float().to(self.device)
                y = by[:, -Hmax * P:][:, :nH * P].float().to(self.device)
                opt.zero_grad()
                out = m(x, nH)
                el = (out['pred'] - y).square()                                        # (b, H, C)
                fut = el.mean()
                hist = (out['hist'] - x[:, P:].reshape(x.shape[0], -1, P, m.c)).square().mean()
                loss = fut + lam * hist
                loss.backward()
                opt.step()
                m.update_vpt_cost(el.detach())
                lf.append(fut.item()); lh.append(hist.item())
                if not np.isfinite(lf[-1]):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch + 1} iter {i + 1} (L={L}, H={nH * P})')
                if (i + 1) % 100 == 0:
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | L {L} H {nH * P} | future mse {fut.item():.5f} hist mse {hist.item():.5f}', flush=True)
                if max_it and i + 1 >= max_it:
                    break
            res = self.eval_grid('val', val_grid, stride=vstride)
            vloss = float(np.mean([r['mse'] for r in res.values()]))
            print(f'Epoch: {epoch + 1} cost time: {time.time() - t0:.0f} | train future {np.mean(lf):.5f} hist {np.mean(lh):.5f} | val mean mse {vloss:.5f} | ' +
                  ' '.join(f'{L}/{H}:{r["mse"]:.4f}' for (L, H), r in res.items()), flush=True)
            early(vloss, m, path)
            if early.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(opt, epoch + 1, a)
        m.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return m

    def test(self, setting, test=0):
        m = self.model
        if test:
            m.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        m.reset_ids_shuffle()
        Ls = [int(v) for v in os.environ.get('R1_TEST_L', '384,720,1440').split(',')]
        Hs = [int(v) for v in os.environ.get('R1_TEST_H', '96,192,336').split(',')]
        for flag in ('val', 'test'):
            res = self.eval_grid(flag, [(L, H) for L in Ls for H in Hs], naive=True, per_var=(720, 96) if 720 in Ls and 96 in Hs else None)
            for (L, H), r in res.items():
                print(f'[R1 {flag}] L={L:4d} H={H:3d} | mse {r["mse"]:.6f} mae {r["mae"]:.6f} | naive last-patch mse {r["last"]:.6f} '
                      f'linear mse {r["lin"]:.6f}', flush=True)
        self.eval_grid('val', [(L, Hs[0]) for L in Ls], stride=4, state_stats=True)
        if m.vpt:
            print(f'[R1 VPT] adjacency (mean transition cost):\n{m.enc.get_adjacency_matrix().cpu().numpy().round(3)}', flush=True)
