"""PAT-304 R1 experiment: recurrent MambaTS baseline (patchmem/r1_model.py), random lookback / horizon training, one checkpoint
evaluated on a fixed lookback x horizon grid. Official run.py / data / seed / setting string; run via patchmem/run_r1.py.
Env:
  R1_LMIN / R1_LMAX      training lookback range in time steps, multiples of patch_len (default 384 / 1440)
  R1_HMIN / R1_HMAX      training horizon range in patches (default 2 / 7, i.e. H96..H336 for P=48)
  R1_MIX=1               per-sample random lookback / horizon inside every batch (default 0: one (L, H) per batch). Windows are
                         right-aligned and padded to the batch maximum (exact, see r1_model.py); the future loss is each sample's mean
                         over its own horizon, averaged over the batch (each sample weighs the same, as with one H per batch)
  R1_LAMBDA_HIST         weight of the in-window next-patch auxiliary loss at every history boundary (default 0)
  R1_RO_LN=1             LayerNorm on the readout input only (default off)
  R1_VAL_GRID            early-stopping grid 'L:H,...' (default 384:96,720:96,1440:96,384:336,720:336,1440:336), mean MSE
  R1_VAL_STRIDE          window stride for the per-epoch validation (default 4)
  R1_TEST_L / R1_TEST_H  test grid (default 384,720,1440 / 96,192,336)
  R1_EVAL_BS             evaluation batch size (default 64)
  R1_EVAL_PERM           fixed evaluation scan order 'v0,v1,...' (overrides the ATSP order; matched-pi comparisons across models)
  R1_ATSP_SEED           numpy seed of the evaluation ATSP solve (default 0): the eval order is a deterministic function of the adjacency
  R1_MAX_ITERS           stop each training epoch after this many iterations (smoke runs; default 0 = full epoch)
  R1_MEM=1               R1 + state-level retrieval memory (patchmem/r1_mem.py, R1Mem); default 0 = plain R1 baseline
  R1_CORR                CorrNet variant for the memory values: gated (default, GatedCorrNet: C(h, e=0) = 0) | shared | shared_id | channel
  R1_MEM_GROUP           StateMem channels per softmax head (default 4)
  R1_CORR_SHUFFLE=1      control: residual rolled across the batch before CorrNet
  R1_LAMBDA_CORR         weight of the writer loss ||R_sg(sg h+ + C) - x_(t+1)||^2 (memory runs only, default 1; trains CorrNet only)
  R1_MEM_VALUE_DETACH=1  Pure-B (B1): memory values detached when written (default 0 = Hybrid B2)
  R1_MEM_RETRIEVE=0      B3: writer trained, retrieval injection off (default 1)
  R1_MEM_ROLL_RETRIEVE=0 B2-H: retrieval at the history boundaries only, none in the rollout (default 1)
  R1_MEM_QK_DIM          per-head q/k dimension of the retrieval score (default group*N = full-state W_Q, W_K: D*N -> D*N)
  R1_CORR_ZDIM           GatedCorrNet code size z_dim (default 32)
  R1_MEM_UNIFORM=1       selection-free retrieval trained from scratch: no W_Q / W_K / tau, a = 1/(available slots) (default 0)
  R1_MEM_NULL=1          learnable null slot (value 0) in the retrieval softmax: abstain when no slot matches (default 0)
  R1_MEM_SLOT_KEEP       B2-drop: training-time per-sample probability that a history slot is written (default 1 = all; eval: all)
  R1_OBS=1               R1 + observation-space error memory (patchmem/r1_obsmem.py, R1Obs): values = the backbone's lookback errors,
                         the retrieved correction is added to the forecast. Uses R1_MEM_GROUP, R1_MEM_QK_DIM (default 8),
                         R1_MEM_SLOT_KEEP, R1_MEM_UNIFORM; R1_OBS_FEED=0 feeds the uncorrected backbone patch back (default 1: the forecast);
                         R1_OBS_VPATH=1: values (errors) transported by a state-driven Householder path, logits as O;
                         R1_OBS_PATH=1: PaTH-style logits q^T H_t..H_{j+1} k (per-head state-driven Householder, values untouched);
                         R1_OBS_PLAIN=1: one head, correction = gamma * retrieved errors (scalar gamma, no W / gate; RATL-style)
  R1_PATHRES=1           R1 + path-transported residual correction (patchmem/r1_pathres.py, R1PathRes, PAT-307): no attention, the
                         lookback errors are carried by a state-driven Householder path with a forget gate. R1_PATHRES_RANK (default 32),
                         R1_PATHRES_NOH=1 (H = I ablation), R1_MEM_SLOT_KEEP (default 1)
  R1_TEFL=native|faithful  R1 + TEFL-style error feedback (patchmem/r1_tefl.py, R1Tefl, PAT-307): T0 = last Hc one-step lookback
                         errors, T1 = multi-step errors of R1's own rollout from t-Hc; y += ReLU(eps W1) W2 (W2 = 0 at init).
                         R1_TEFL_HC (Hc in patches, default 2; 0 = all lookback one-step errors, native only), R1_TEFL_RANK (default 64),
                         R1_TEFL_FEED=0 feeds the backbone patch back, R1_TEFL_EXTRAP last|zero beyond Hmax patches (default last)
  R1_DIAG_EVERY          training diagnostics (JSONL) every N steps (default 100; 0 = off)
  R1_DIAG_VAL_BATCHES    eval batches per (L, H) with memory diagnostics at every validation (default 2)
  R1_AUDIT=0             skip the same-checkpoint inference audits after test (memory runs; default 1)
  R1_AUDIT_GRID          audit grid 'L:H,...' (default: the test grid); R1_AUDIT_PERM_REPS random key-value pairings (default 3)
  R1_LOG_DIR             root for param snapshots / diagnostics (default patchmem/logs)
The training loss is the forecast MSE over the sampled horizon (+ lambda_hist * L_hist). L_hist is in-window auxiliary supervision:
the lookback's RevIN statistics already include x_(t+1), so it is NOT an online per-origin evaluation. L_corr (writer loss) is in the
RevIN-normalised space, the forecast losses in the dataset scale: absolute values are not comparable."""
import copy, json, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from patchmem.r1_model import R1, atsp_order
from patchmem.r1_mem import R1Mem
from patchmem.r1_obsmem import R1Obs
from patchmem.r1_tefl import R1Tefl
from patchmem.r1_pathres import R1PathRes
from patchmem.param_snapshot import log_model_parameter_stats
from utils.tools import EarlyStopping, adjust_learning_rate


def _grid(s):
    return [tuple(int(v) for v in item.split(':')) for item in s.split(',')]


def solve_eval_order(m, seed=0):
    """Cache the evaluation scan order of model m (R1 / R1Mem with VPT) as a deterministic function of its VPT adjacency: the python_tsp
    SA / LS solvers draw from numpy's global RNG (acceptance) and Python's `random` (perturbation order), so the same checkpoint could
    otherwise get a different pi in another process (or after other draws). Both global states are restored afterwards, so the
    training random stream is unchanged."""
    if not m.vpt or m.ids_shuffle is not None:
        return
    st_np, st_py = np.random.get_state(), random.getstate()
    np.random.seed(seed); random.seed(seed)
    try:
        m.ids_shuffle = atsp_order(m.vpt_adjacency(), m.enc.ATSP_solver, m.c)
    finally:
        np.random.set_state(st_np); random.setstate(st_py)


def _grad_group(name):
    """Gradient-statistics group: encoder layer, CorrNet / StateMem sub-module (E, Wc, G1, G2, Wq, Wk, log_tau, Wg), else 2 components."""
    parts = name.split('.')
    if parts[0] in ('corr_net', 'mem_net'):
        return '.'.join(parts[:2])
    if 'layers' in parts:
        i = parts.index('layers')
        return '.'.join(parts[:i + 2])
    return '.'.join(parts[:2]) if len(parts) > 2 else parts[0]


@torch.no_grad()
def grad_param_stats(model):
    """Per group: grad L2, param L2, grad/param, None / zero / non-finite grad tensor counts (from the grads currently stored)."""
    out = {}
    for n, p in model.named_parameters():
        g = out.setdefault(_grad_group(n), dict(grad_sq=0.0, param_sq=0.0, n=0, none=0, zero=0, nonfinite=0))
        g['n'] += 1
        g['param_sq'] += p.detach().double().square().sum().item()
        if p.grad is None:
            g['none'] += 1
            continue
        gd = p.grad.detach().double()
        if not torch.isfinite(gd).all():
            g['nonfinite'] += 1
        s = gd.square().sum().item()
        g['grad_sq'] += s
        g['zero'] += s == 0
    return {k: dict(grad_norm=v['grad_sq'] ** 0.5, param_norm=v['param_sq'] ** 0.5,
                    grad_over_param=v['grad_sq'] ** 0.5 / max(v['param_sq'] ** 0.5, 1e-12),
                    tensors=v['n'], none=v['none'], zero=int(v['zero']), nonfinite=v['nonfinite']) for k, v in out.items()}


class Exp_R1(Exp_Long_Term_Forecast):
    def _run_tag(self):
        a = self.args
        return f'{a.model_id}_{a.des}_s{a.seed}'

    def _log_root(self):
        return os.environ.get('R1_LOG_DIR', os.path.join('patchmem', 'logs'))

    def jlog(self, record):
        """Append one structured record to <log root>/diag/<run tag>.jsonl."""
        path = os.path.join(self._log_root(), 'diag', self._run_tag() + '.jsonl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(dict(time=time.time(), **record), default=float) + '\n')

    def snapshot(self, event, **meta):
        a = self.args
        log_model_parameter_stats(self.model, event, os.path.join(self._log_root(), 'param_snapshots', self._run_tag()),
                                  meta=dict(seed=a.seed, model_id=a.model_id, des=a.des, patch_len=a.patch_len, VPT=a.VPT_mode, **meta))

    def _build_model(self):
        a = self.args
        ro_ln = os.environ.get('R1_RO_LN', '0') == '1'
        if os.environ.get('R1_MEM', '0') == '1':
            m = R1Mem(a, readout_ln=ro_ln, corr=os.environ.get('R1_CORR', 'gated'), mem_group=int(os.environ.get('R1_MEM_GROUP', '4')),
                      corr_shuffle=os.environ.get('R1_CORR_SHUFFLE', '0') == '1',
                      value_detach=os.environ.get('R1_MEM_VALUE_DETACH', '0') == '1',
                      retrieve=os.environ.get('R1_MEM_RETRIEVE', '1') == '1',
                      roll_retrieve=os.environ.get('R1_MEM_ROLL_RETRIEVE', '1') == '1',
                      slot_keep=float(os.environ.get('R1_MEM_SLOT_KEEP', '1')),
                      qk_dim=int(os.environ['R1_MEM_QK_DIM']) if os.environ.get('R1_MEM_QK_DIM') else None,
                      z_dim=int(os.environ.get('R1_CORR_ZDIM', '32')),
                      uniform=os.environ.get('R1_MEM_UNIFORM', '0') == '1',
                      null=os.environ.get('R1_MEM_NULL', '0') == '1').float()
            print(f'R1Mem: corr {os.environ.get("R1_CORR", "gated")} ({sum(p.numel() for p in m.corr_net.parameters())} params) | '
                  f'StateMem group {m.mem_net.group} heads {m.mem_net.nh} qk_dim {m.mem_net.dh} ({sum(p.numel() for p in m.mem_net.parameters())} params) | '
                  f'corr_shuffle {m.corr_shuffle} | value_detach {m.value_detach} (Pure-B) | retrieve {m.retrieve} | '
                  f'rollout retrieve {m.roll_retrieve} | train slot keep {m.slot_keep} | null slot {m.mem_net.null}', flush=True)
        elif os.environ.get('R1_PATHRES', '0') == '1':
            m = R1PathRes(a, readout_ln=ro_ln, rank=int(os.environ.get('R1_PATHRES_RANK', '32')),
                          noH=os.environ.get('R1_PATHRES_NOH', '0') == '1', slot_keep=float(os.environ.get('R1_MEM_SLOT_KEEP', '1')),
                          feed=os.environ.get('R1_OBS_FEED', '1') == '1').float()
            print(f'R1PathRes: rank {m.obs_net.r} noH {m.obs_net.noH} ({sum(p.numel() for p in m.obs_net.parameters())} params) | '
                  f'train slot keep {m.slot_keep} | feed forecast {m.feed}', flush=True)
        elif os.environ.get('R1_OBS', '0') == '1':
            m = R1Obs(a, readout_ln=ro_ln, mem_group=int(os.environ.get('R1_MEM_GROUP', '4')),
                      qk_dim=int(os.environ.get('R1_MEM_QK_DIM') or 8), slot_keep=float(os.environ.get('R1_MEM_SLOT_KEEP', '1')),
                      uniform=os.environ.get('R1_MEM_UNIFORM', '0') == '1', feed=os.environ.get('R1_OBS_FEED', '1') == '1',
                      plain=os.environ.get('R1_OBS_PLAIN', '0') == '1', path=os.environ.get('R1_OBS_PATH', '0') == '1',
                      vpath=os.environ.get('R1_OBS_VPATH', '0') == '1').float()
            print(f'R1Obs: ObsMem heads {m.obs_net.nh} qk_dim {m.obs_net.dh} uniform {m.obs_net.uniform} plain {m.obs_net.plain} path {m.obs_net.path} vpath {m.obs_net.vpath} '
                  f'({sum(p.numel() for p in m.obs_net.parameters())} params) | train slot keep {m.slot_keep} | feed forecast {m.feed}', flush=True)
        elif os.environ.get('R1_TEFL'):
            m = R1Tefl(a, readout_ln=ro_ln, mode=os.environ['R1_TEFL'], hc=int(os.environ.get('R1_TEFL_HC', '2')),
                       rank=int(os.environ.get('R1_TEFL_RANK', '64')), hmax=int(os.environ.get('R1_HMAX', '7')),
                       feed=os.environ.get('R1_TEFL_FEED', '1') == '1', tmax=int(os.environ.get('R1_LMAX', '1440')) // a.patch_len,
                       extrap=os.environ.get('R1_TEFL_EXTRAP', 'last')).float()
            print(f'R1Tefl: mode {m.mode} Hc {m.hc} patches (0 = all lookback, input {m.din}) rank {m.W1.out_features} Hmax {m.hmax} '
                  f'extrap {m.extrap} | '
                  f'g params {m.W1.weight.numel() + m.W2.weight.numel()} | feed forecast {m.feed}', flush=True)
        else:
            m = R1(a, readout_ln=ro_ln).float()
        nro = sum(p.numel() for p in m.readout.parameters())
        print(f'R1: d_inner {m.D} d_state {m.N} vars {m.c} patch {m.P} | readout {m.D * m.N} -> {m.c * m.P} ({nro} params) | '
              f'readout LN {m.ro_ln is not None} | VPT {m.vpt} ({a.ATSP_solver}) | trainable {sum(p.numel() for p in m.parameters() if p.requires_grad)}',
              flush=True)
        self.model = m                                                                  # snapshot before Exp_Basic moves it to the device
        self.snapshot('init')
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

    def _fix_eval_order(self):
        """Evaluation scan order: R1_EVAL_PERM if set, else the ATSP order solved deterministically (R1_ATSP_SEED, default 0)."""
        s = os.environ.get('R1_EVAL_PERM')
        m = self.model
        if s and m.vpt:
            order = [int(v) for v in s.split(',')]
            assert sorted(order) == list(range(m.c)), order
            m.ids_shuffle = order
        solve_eval_order(m, int(os.environ.get('R1_ATSP_SEED', '0')))

    def eval_order(self):
        m = self.model
        return list(m.ids_shuffle) if (m.vpt and m.ids_shuffle is not None) else (None if m.vpt else list(range(m.c)))

    @torch.no_grad()
    def eval_grid(self, flag, grid, stride=1, naive=False, per_var=None, state_stats=False, diag_batches=0, timing=False):
        """Mean MSE/MAE of the forecast per (L, H) on the official `flag` split built with seq_len=L, pred_len=H (every `stride`-th window).
        naive: also last-patch-repeat and per-variable least-squares line over the last patch. per_var: (L, H) for a per-variable report
        in scan position order. state_stats: RMS of the final boundary state per lookback. diag_batches: memory diagnostics on the first
        n batches of every (L, H) (R1Mem). timing: wall time and peak GPU memory per (L, H)."""
        m = self.model
        m.eval()
        self._fix_eval_order()
        bs = int(os.environ.get('R1_EVAL_BS', '64'))
        P, res = m.P, {}
        for L, H in grid:
            ds = self._ds(flag, L, H)
            ids = list(range(0, len(ds), stride))
            acc = dict(mse=0.0, mae=0.0, last=0.0, lin=0.0, n=0)
            var_se = torch.zeros(m.c)
            k_se, k_ae = torch.zeros(H // P, dtype=torch.float64), torch.zeros(H // P, dtype=torch.float64)   # per rollout patch k
            rms = []
            if timing and torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for bi, s0 in enumerate(range(0, len(ids), bs)):
                bx, by = self._batch(ds, ids[s0:s0 + bs])
                bx, y = bx.to(self.device), by[:, -H:].to(self.device)
                tr = {} if state_stats else None
                use_diag = isinstance(m, R1Mem) and bi < diag_batches
                if use_diag:
                    m.diag = True
                if isinstance(m, R1Tefl) and m.audit == 'perm':                       # errors of windows half the split away
                    bd, _ = self._batch(ds, [(j + len(ds) // 2) % len(ds) for j in ids[s0:s0 + bs]])
                    m.audit = None
                    eps_d = m(bd.to(self.device), H // P, ret_eps=True)['eps']
                    m.audit = 'perm'
                    pred = m(bx, H // P, trace=tr, eps_override=eps_d)['pred']
                else:
                    pred = m(bx, H // P, trace=tr)['pred']
                m.diag = False
                err = pred - y
                acc['mse'] += err.square().sum().item(); acc['mae'] += err.abs().sum().item(); acc['n'] += err.numel()
                var_se += err.square().sum((0, 1)).cpu()
                ek = err.reshape(err.shape[0], H // P, P, -1)
                k_se += ek.square().sum((0, 2, 3)).double().cpu(); k_ae += ek.abs().sum((0, 2, 3)).double().cpu()
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
            nk = n / (H // P)
            res[(L, H)] = dict(mse=acc['mse'] / n, mae=acc['mae'] / n, windows=len(ids),
                               patch_mse=(k_se / nk).tolist(), patch_mae=(k_ae / nk).tolist())
            if timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    res[(L, H)]['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
                res[(L, H)]['seconds'] = time.time() - t0
            if isinstance(m, R1Mem) and diag_batches:
                res[(L, H)]['diag'] = m.pop_diag()
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

    def check_routing(self, x, nH, lam_c, lens=None, hvalid=None):
        """Gradient routing on one batch, each loss backpropagated ALONE (L_future; L_corr when lambda_corr > 0). Run under fork_rng so
        the training random stream (dropout, VPT pi) is untouched; grads are cleared afterwards, no optimizer step."""
        m, P = self.model, self.model.P
        out = {}
        devs = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devs):
            for name in ('future', 'corr'):
                if name == 'corr' and not lam_c > 0:
                    continue
                m.zero_grad(set_to_none=True)
                o = m(x[0], nH, lens=lens)
                loss = self._future_loss((o['pred'] - x[1]).square(), hvalid) if name == 'future' else m.corr_loss
                if loss is None:
                    continue
                loss.backward()
                out[name] = grad_param_stats(m)
        m.zero_grad(set_to_none=True)
        return out

    @staticmethod
    def _future_loss(el, hvalid=None):
        """el (b, H, C) squared errors; hvalid (b, H) bool or None. Mean over (H, C) per sample over its own horizon, then over samples."""
        if hvalid is None:
            return el.mean()
        w = hvalid.to(el.dtype).unsqueeze(-1)
        return ((el * w).sum((1, 2)) / (w.sum((1, 2)) * el.shape[-1])).mean()

    def train(self, setting):
        a, m = self.args, self.model
        P = m.P
        Lmin, Lmax = int(os.environ.get('R1_LMIN', '384')), int(os.environ.get('R1_LMAX', '1440'))
        Hmin, Hmax = int(os.environ.get('R1_HMIN', '2')), int(os.environ.get('R1_HMAX', '7'))
        lam = float(os.environ.get('R1_LAMBDA_HIST', '0'))
        is_mem = isinstance(m, R1Mem)
        lam_c = float(os.environ.get('R1_LAMBDA_CORR', '1')) if is_mem else 0.0
        if is_mem:
            m.corr_aux = lam_c > 0
        diag_every = int(os.environ.get('R1_DIAG_EVERY', '100'))
        diag_val = int(os.environ.get('R1_DIAG_VAL_BATCHES', '2'))
        assert Lmin % P == 0 and Lmax % P == 0 and Lmin <= Lmax and 1 <= Hmin <= Hmax
        mix = os.environ.get('R1_MIX', '0') == '1'
        val_grid = _grid(os.environ.get('R1_VAL_GRID', '384:96,720:96,1440:96,384:336,720:336,1440:336'))
        vstride = int(os.environ.get('R1_VAL_STRIDE', '4'))
        max_it = int(os.environ.get('R1_MAX_ITERS', '0'))
        tr_ds = self._ds('train', Lmax, Hmax * P)
        loader = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.num_workers)
        print(f'R1 train: {len(tr_ds)} windows of {Lmax}+{Hmax * P} | lookback {Lmin}..{Lmax} step {P} | horizon {Hmin}..{Hmax} patches | '
              f'per-sample L/H {mix} | lambda_hist {lam} | lambda_corr {lam_c} | val grid {val_grid} stride {vstride}', flush=True)
        self.jlog(dict(record='config', setting=setting, model=type(m).__name__, lambda_hist=lam, lambda_corr=lam_c,
                       value_detach=getattr(m, 'value_detach', None), retrieve=getattr(m, 'retrieve', None),
                       corr_shuffle=getattr(m, 'corr_shuffle', None), mix=mix, Lmin=Lmin, Lmax=Lmax, Hmin=Hmin, Hmax=Hmax, val_grid=val_grid,
                       seed=a.seed, batch_size=a.batch_size, lr=a.learning_rate, epochs=a.train_epochs))
        rng = np.random.default_rng(a.seed)
        path = os.path.join(a.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        opt = torch.optim.Adam(m.parameters(), lr=a.learning_rate)
        early = EarlyStopping(patience=a.patience, verbose=True)
        step, best_epoch, best_step = 0, None, None
        for epoch in range(a.train_epochs):
            m.train()
            t0 = time.time()
            lf, lh, lc = [], [], []
            t_int, n_int = time.time(), 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for i, (bx, by, _, _) in enumerate(loader):
                if mix:                                                                 # per-sample (L, H), padded to the batch max
                    bsz = bx.shape[0]
                    Ls = rng.choice(np.arange(Lmin, Lmax + 1, P), size=bsz)
                    nHs = rng.integers(Hmin, Hmax + 1, size=bsz)
                    L, nH = int(Ls.max()), int(nHs.max())
                    lens = torch.as_tensor(Ls, device=self.device)
                    hvalid = torch.arange(nH * P, device=self.device)[None, :] < torch.as_tensor(nHs * P, device=self.device)[:, None]
                else:
                    L = int(rng.choice(np.arange(Lmin, Lmax + 1, P)))
                    nH = int(rng.integers(Hmin, Hmax + 1))
                    lens = hvalid = None
                x = bx[:, -L:].float().to(self.device)
                y = by[:, -Hmax * P:][:, :nH * P].float().to(self.device)
                if is_mem and i == 0:                                                   # per-epoch loss-by-loss gradient routing check
                    self.jlog(dict(record='routing', epoch=epoch + 1, step=step, L=L, H=nH * P,
                                   value_detach=m.value_detach, retrieve=m.retrieve, groups=self.check_routing((x, y), nH, lam_c, lens, hvalid)))
                do_diag = diag_every and (step + 1) % diag_every == 0
                if is_mem and do_diag:
                    m.diag = True
                opt.zero_grad()
                out = m(x, nH) if lens is None else m(x, nH, lens=lens)
                if is_mem:
                    m.diag = False
                el = (out['pred'] - y).square()                                        # (b, H, C)
                fut = self._future_loss(el, hvalid)
                eh = (out['hist'] - x[:, P:].reshape(x.shape[0], -1, P, m.c)).square()   # (b, T-1, P, C)
                if lens is None:
                    hist = eh.mean()
                else:                                                                   # history prediction j (of patch j+1) valid iff j >= t0
                    hv = (torch.arange(L // P - 1, device=self.device)[None, :] >= (L - lens)[:, None] // P).to(eh.dtype)
                    hist = ((eh.mean((2, 3)) * hv).sum(1) / hv.sum(1).clamp_min(1.0)).mean()
                loss = fut + lam * hist
                corr = m.corr_loss if lam_c > 0 else None                              # writer loss, trains CorrNet only
                if corr is not None:
                    loss = loss + lam_c * corr
                loss.backward()
                if do_diag:
                    rec = dict(record='train', epoch=epoch + 1, iter=i + 1, step=step + 1, L=L, H=nH * P,
                               L_mean=float(lens.float().mean()) if lens is not None else L,
                               L_future=fut.item(), L_hist=hist.item(), L_corr=None if corr is None else corr.item(),
                               weighted=dict(future=fut.item(), hist=lam * hist.item(), corr=None if corr is None else lam_c * corr.item()),
                               steps_per_sec=(n_int + 1) / max(time.time() - t_int, 1e-9), grads=grad_param_stats(m))
                    if torch.cuda.is_available():
                        rec['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
                    if is_mem:
                        rec['mem'] = m.pop_diag()
                    self.jlog(rec)
                    t_int, n_int = time.time(), -1
                opt.step()
                m.update_vpt_cost(el.detach(), hvalid)
                step += 1; n_int += 1
                lf.append(fut.item()); lh.append(hist.item())
                if corr is not None:
                    lc.append(corr.item())
                if not np.isfinite(lf[-1]):
                    raise FloatingPointError(f'non-finite loss at epoch {epoch + 1} iter {i + 1} (L={L}, H={nH * P})')
                if (i + 1) % 100 == 0:
                    ms = ' | mem ' + ' '.join(f'{k} {v:.3g}' for k, v in m.stats().items()) if is_mem or isinstance(m, (R1Obs, R1Tefl)) else ''
                    ms += f' corr_loss {corr.item():.5f}' if corr is not None else ''
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | L {L} H {nH * P} | future mse {fut.item():.5f} hist mse {hist.item():.5f}{ms}', flush=True)
                if max_it and i + 1 >= max_it:
                    break
            res = self.eval_grid('val', val_grid, stride=vstride, diag_batches=diag_val)
            vloss = float(np.mean([r['mse'] for r in res.values()]))
            print(f'Epoch: {epoch + 1} cost time: {time.time() - t0:.0f} | train future {np.mean(lf):.5f} hist {np.mean(lh):.5f}'
                  f'{f" corr {np.mean(lc):.5f}" if lc else ""} | val mean mse {vloss:.5f} | ' +
                  ' '.join(f'{L}/{H}:{r["mse"]:.4f}' for (L, H), r in res.items()), flush=True)
            self.jlog(dict(record='val', epoch=epoch + 1, step=step, seconds=time.time() - t0, val_mean_mse=vloss, order=self.eval_order(),
                           train_future=float(np.mean(lf)), train_hist=float(np.mean(lh)), train_corr=float(np.mean(lc)) if lc else None,
                           grid={f'{L}/{H}': r for (L, H), r in res.items()}))
            early(vloss, m, path)
            if early.counter == 0:
                best_epoch, best_step = epoch + 1, step
            if early.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(opt, epoch + 1, a)
        ck = os.path.join(path, 'checkpoint.pth')
        m.load_state_dict(torch.load(ck))
        print(f'[R1 best] epoch {best_epoch} step {best_step} of {step} steps', flush=True)
        self.jlog(dict(record='best', best_epoch=best_epoch, best_step=best_step, total_steps=step))
        self.snapshot('checkpoint_loaded', checkpoint=ck, source='train: early-stopping best', best_epoch=best_epoch, best_step=best_step,
                      total_steps=step, setting=setting)
        return m

    def test(self, setting, test=0):
        m = self.model
        if test:
            ck = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            m.load_state_dict(torch.load(ck))
            self.snapshot('checkpoint_loaded', checkpoint=ck, source='test: reload', setting=setting)
        m.reset_ids_shuffle()
        Ls = [int(v) for v in os.environ.get('R1_TEST_L', '384,720,1440').split(',')]
        Hs = [int(v) for v in os.environ.get('R1_TEST_H', '96,192,336').split(',')]
        grid = [(L, H) for L in Ls for H in Hs]
        for flag in ('val', 'test'):
            res = self.eval_grid(flag, grid, naive=True, per_var=(720, 96) if 720 in Ls and 96 in Hs else None, timing=True,
                                 diag_batches=int(os.environ.get('R1_DIAG_VAL_BATCHES', '2')))
            for (L, H), r in res.items():
                peak = f' peak {r["peak_mem_mb"]:.0f}MB' if 'peak_mem_mb' in r else ''
                print(f'[R1 {flag}] L={L:4d} H={H:3d} | mse {r["mse"]:.6f} mae {r["mae"]:.6f} | naive last-patch mse {r["last"]:.6f} '
                      f'linear mse {r["lin"]:.6f} | {r["seconds"]:.1f}s{peak} | per-patch mse ' + ' '.join(f'{v:.4f}' for v in r['patch_mse']),
                      flush=True)
            self.jlog(dict(record='test', flag=flag, setting=setting, order=self.eval_order(), grid={f'{L}/{H}': r for (L, H), r in res.items()}))
        self.eval_grid('val', [(L, Hs[0]) for L in Ls], stride=4, state_stats=True)
        if m.vpt:
            adj = m.vpt_adjacency()
            print(f'[R1 VPT] skipped cost updates {m.vpt_skipped} | edge counts min {m.enc.count_matrix.min().item():.0f} | adjacency (mean transition cost):\n'
                  f'{None if adj is None else adj.cpu().numpy().round(3)}', flush=True)
            self.jlog(dict(record='vpt', order=self.eval_order(), skipped=m.vpt_skipped, count_min=m.enc.count_matrix.min().item(),
                           adjacency=None if adj is None else adj.cpu().tolist()))
        if isinstance(m, (R1Mem, R1Obs, R1Tefl)) and os.environ.get('R1_AUDIT', '1') == '1':
            self.audit(setting, grid)

    def audit(self, setting, grid):
        """Same-checkpoint inference audits (eval only, same fixed scan order, same windows): normal | zero (gate 0, memory off) |
        uniform attention | perm (key-value time pairing randomly broken, R1_AUDIT_PERM_REPS seeds). Delta = MSE(mode) - MSE(normal)."""
        m = self.model
        g = _grid(os.environ['R1_AUDIT_GRID']) if os.environ.get('R1_AUDIT_GRID') else grid
        reps = int(os.environ.get('R1_AUDIT_PERM_REPS', '3'))
        if isinstance(m, R1Tefl):                                                       # no attention; perm = errors of distant windows
            modes = [('normal', None, 0), ('zero', 'zero', 0), ('perm', 'perm', 0)] + ([('recent2', 'recent', 0)] if m.hc == 0 else [])
        else:
            modes = [('normal', None, 0), ('zero', 'zero', 0), ('uniform', 'uniform', 0)] + [(f'perm{s}', 'perm', s) for s in range(reps)]
            if getattr(m.obs_net, 'path', False) or getattr(m.obs_net, 'vpath', False):                                       # O-PaTH: path removed at inference
                modes.append(('noH', 'noH', 0))
        order = self.eval_order()
        for flag in ('val', 'test'):
            out = {}
            for name, mode, seed in modes:
                m.set_audit(mode, seed)
                out[name] = self.eval_grid(flag, g)
            m.set_audit(None)
            assert self.eval_order() == order, 'audit changed the scan order'
            for (L, H) in g:
                base = out['normal'][(L, H)]
                print(f'[R1Mem audit {flag}] L={L:4d} H={H:3d} | normal mse {base["mse"]:.6f} mae {base["mae"]:.6f} | ' +
                      ' | '.join(f'{k} dMSE {r[(L, H)]["mse"] - base["mse"]:+.6f} dMAE {r[(L, H)]["mae"] - base["mae"]:+.6f}'
                                 for k, r in out.items() if k != 'normal'), flush=True)
                for k, r in out.items():                                                # per rollout patch: MSE(mode)_k - MSE(normal)_k
                    if k != 'normal':
                        print(f'[R1Mem audit {flag} per-patch] L={L:4d} H={H:3d} {k:8s} dMSE_k ' +
                              ' '.join(f'{a - b:+.5f}' for a, b in zip(r[(L, H)]['patch_mse'], base['patch_mse'])), flush=True)
            self.jlog(dict(record='audit', flag=flag, setting=setting, order=order,
                           modes={k: {f'{L}/{H}': r for (L, H), r in v.items()} for k, v in out.items()}))
