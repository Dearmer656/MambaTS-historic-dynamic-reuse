"""PAT-307 joint training of TEFL + self-attention over the model's own past forecast errors (R1TeflAtt, patchmem/r1_tefl_att.py).
Everything is trained (backbone, TEFL W1 / W2, attention), with the T0r (ETTh2: Hc = 2) / TF0 (Weather: Hc = 0) recipe of
patchmem/exp_r1.py: random lookback 384..1440 step 48 and horizon 2..7 patches per batch, batch 8, Adam, lr type1, 10 epochs,
patience 3, early stopping on the mean val MSE of the R1 val grid (stride 4). Same seeds as T0r / TF0.
Memory bank: for every origin s of the split (lookback LB = 1440 available), the model's forecast with the attention term OFF (= its
TEFL forecast) at L = 1440, H = 336: raw error v_s (data units) and the TEFL input eps_s. Refreshed with the current weights at the
start of every epoch (train split; stale and gradient-free inside the epoch) and before every evaluation (val / test split, read with
the lookback extended by LB + 8H so the memory reaches before the split start: an online setting that logs its own past forecasts;
disclosed, no leakage: a demo s = t - d has d >= H, so its stored future ends by t).
Arms: 'pos' (learned window over the look-back distance), 'qk' (window + low-rank Wq / Wk content selection). Baseline: the existing
T0r / TF0 checkpoint of the same seed, evaluated by the same code. Test grid L 384/720/1440 x H 96/192/336 (official origins);
audits on the trained model: 'off' (attention term removed), 'perm' (values permuted against the demos).
Pre-registered (ETTh2, test L = 1440 / H = 336, ALL three seeds): qk < T0r and qk < pos -> content selection helps; pos < T0r ->
a learned window over own past errors helps. Weather and the other cells are secondary.
usage: python patchmem/tefl_att.py --data E|W [--seeds 3047,3048,3049] [--variants pos,qk] [--out DIR] [--smoke]"""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, H, NP = 48, 336, 7
LB, DMAX = 1440, 8 * H
DS = np.arange(H, DMAX + 1, P)                                                          # 50 look-back distances
EXT = LB + DMAX                                                                         # val / test lookback extension
VAL_GRID = [(384, 96), (720, 96), (1440, 96), (384, 336), (720, 336), (1440, 336)]
TEST_GRID = [(L, h) for L in (384, 720, 1440) for h in (96, 192, 336)]
ARM = {'E': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64'), des='R1TfnH2R64h0s{s}', tag='R1TfnH2R64'),
       'W': dict(env=dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64'), des='R1WTfnH0R64h0s{s}', tag='R1WTfnH0R64')}


class Bank:
    """Own-forecast memory of one series X (T, C): origins s = LB .. T - H."""
    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.n = len(X) - H + 1 - LB

    @torch.no_grad()
    def refresh(self, ex, bs=128):
        m = ex.model
        was = m.training
        m.eval(); m.mem = None
        m.reset_ids_shuffle(); ex._fix_eval_order()                                     # re-solve the deterministic eval order for these weights
        vs, es = [], []
        for i0 in range(LB, LB + self.n, bs):
            ts = range(i0, min(i0 + bs, LB + self.n))
            bx = torch.stack([self.X[t - LB:t] for t in ts]).to(dev)
            y = torch.stack([self.X[t:t + H] for t in ts]).to(dev)
            out = m(bx, NP, ret_eps=True)
            vs.append((y - out['pred']).cpu()); es.append(out['eps'].half().cpu())
        self.v, self.eps = torch.cat(vs), torch.cat(es)                                 # (n, H, C), (n, C, Din)
        m.train(was)

    def mem(self, ts):
        """ts origins (X coordinates) -> dict(v (b, n, C, H), eps (b, n, C, Din), mask (b, n)) on the device."""
        s = np.asarray(ts)[:, None] - DS[None, :]
        assert (s + H <= np.asarray(ts)[:, None]).all()                                 # every stored future ends by t
        valid = torch.as_tensor(s >= LB)
        idx = torch.as_tensor(np.clip(s - LB, 0, self.n - 1))
        return dict(v=self.v[idx].permute(0, 1, 3, 2).to(dev), eps=self.eps[idx].float().to(dev), mask=valid.to(dev))


@torch.no_grad()
def eval_cells(ex, X, bank, grid, stride=1, audit=None, use_mem=True, bs=64, seed=0):
    """Official-origin MSE per (L, H) on an extended split X (official origins start at index EXT). bank refreshed by the caller."""
    m = ex.model; m.eval(); ex._fix_eval_order()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    if hasattr(m, 'mem_audit'):
        m.mem_audit = audit
    torch.manual_seed(seed)
    res = {}
    for L, Hc in grid:
        ts_all = list(range(EXT, len(X) - Hc + 1, stride))
        se, n = 0.0, 0
        for i0 in range(0, len(ts_all), bs):
            ts = ts_all[i0:i0 + bs]
            bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
            y = torch.stack([Xt[t:t + Hc] for t in ts]).to(dev)
            if hasattr(m, 'mem'):
                m.mem = bank.mem(ts) if use_mem else None
            err = m(bx, Hc // P)['pred'] - y
            se += err.square().sum().item(); n += err.numel()
        res[f'{L}/{Hc}'] = se / n
    if hasattr(m, 'mem'):
        m.mem, m.mem_audit = None, None
    return res


def build(data, seed, variant):
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG
    cf, arm = CFG[data], ARM[data]
    os.environ.pop('R1_OBS', None); os.environ.update(arm['env'])
    if variant:
        os.environ['R1_TEFLATT'] = variant
    else:
        os.environ.pop('R1_TEFLATT', None)
    des = arm['des'].format(s=seed)
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
    from patchmem.exp_r1 import Exp_R1
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return Exp_R1(ar), ar, cf['r1ck'].replace(cf['r1des'], des)


def train_one(data, seed, variant, out, smoke):
    from patchmem.resid_cov import split_x
    from utils.tools import EarlyStopping, adjust_learning_rate
    ex, a, _ = build(data, seed, variant)
    m = ex.model
    path = f'checkpoints/tefl_att/{data}_{variant}_s{seed}'
    os.makedirs(path, exist_ok=True)
    Xtr = split_x(a, LB, 'train')                                                       # train data_x starts at 0
    Xva = split_x(a, EXT, 'val')
    btr, bva = Bank(Xtr), Bank(Xva)
    origins = np.arange(LB, len(Xtr) - H + 1)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), lr=a.learning_rate)
    early = EarlyStopping(patience=a.patience, verbose=True)
    epochs, max_it, vstride = (1, 30, 64) if smoke else (a.train_epochs, 0, 4)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    print(f'[{data} {variant} s{seed}] train origins {len(origins)} | demos {len(DS)} (d = {DS[0]}..{DS[-1]}) | epochs {epochs} '
          f'| trainable {sum(p.numel() for p in m.parameters())}', flush=True)
    for ep in range(epochs):
        t0 = time.time()
        btr.refresh(ex)
        m.train()
        order = rng.permutation(origins)
        lf = []
        for i in range(len(order) // a.batch_size):
            ts = order[i * a.batch_size:(i + 1) * a.batch_size]
            L = int(rng.choice(np.arange(384, LB + 1, P))); nH = int(rng.integers(2, NP + 1))
            x = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
            y = torch.stack([Xt[t:t + nH * P] for t in ts]).to(dev)
            m.mem = btr.mem(ts)
            opt.zero_grad()
            el = (m(x, nH)['pred'] - y).square()
            loss = el.mean()
            loss.backward(); opt.step()
            m.update_vpt_cost(el.detach())
            lf.append(loss.item())
            if not np.isfinite(lf[-1]):
                raise FloatingPointError(f'non-finite loss epoch {ep + 1} iter {i + 1}')
            if (i + 1) % 200 == 0:
                print(f'\titers {i + 1} epoch {ep + 1} | L {L} H {nH * P} | loss {np.mean(lf[-200:]):.5f} | eta {m.eta.detach().abs().mean().item():.4f} '
                      f'| pos range {(m.pos.max() - m.pos.min()).item():.3f}', flush=True)
            if max_it and i + 1 >= max_it:
                break
        m.mem = None
        bva.refresh(ex)
        res = eval_cells(ex, Xva, bva, VAL_GRID, stride=vstride)
        vloss = float(np.mean(list(res.values())))
        print(f'Epoch {ep + 1} {time.time() - t0:.0f}s | train {np.mean(lf):.5f} | val mean {vloss:.5f} | ' + ' '.join(f'{k}:{v:.4f}' for k, v in res.items())
              + f' | eta {[round(v, 3) for v in m.eta.detach().cpu().tolist()]}', flush=True)
        early(vloss, m, path)
        if early.early_stop:
            print('Early stopping', flush=True)
            break
        adjust_learning_rate(opt, ep + 1, a)
    m.load_state_dict(torch.load(f'{path}/checkpoint.pth', map_location=dev))
    m.reset_ids_shuffle()                                                               # the cached order belonged to the last epoch
    Xte = split_x(a, EXT, 'test')
    bte = Bank(Xte); bte.refresh(ex)
    ts_ = 64 if smoke else 1
    r = {'normal': eval_cells(ex, Xte, bte, TEST_GRID, ts_), 'off': eval_cells(ex, Xte, bte, TEST_GRID, ts_, audit='off'),
         'perm': eval_cells(ex, Xte, bte, TEST_GRID, ts_, audit='perm', seed=1)}
    r['eta'] = m.eta.detach().cpu().tolist(); r['pos'] = m.pos.detach().cpu().tolist(); r['T'] = float(torch.exp(m.logt))
    for k in ('normal', 'off', 'perm'):
        print(f'  [test {k}] ' + ' '.join(f'{c}:{v:.4f}' for c, v in r[k].items()) + f' | mean {np.mean(list(r[k].values())):.4f}', flush=True)
    del ex, m; torch.cuda.empty_cache()
    return r


def baseline(data, seed, smoke):
    """The existing T0r / TF0 checkpoint of this seed, evaluated by the same code (no memory)."""
    from patchmem.resid_cov import split_x
    ex, a, ck = build(data, seed, None)
    ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
    r = eval_cells(ex, split_x(a, EXT, 'test'), None, TEST_GRID, 64 if smoke else 1, use_mem=False)
    print(f'  [test T0 baseline] ' + ' '.join(f'{c}:{v:.4f}' for c, v in r.items()) + f' | mean {np.mean(list(r.values())):.4f}', flush=True)
    del ex; torch.cuda.empty_cache()
    return r


def dump(res, fn):
    tmp = fn + '.tmp'
    json.dump(res, open(tmp, 'w')); os.replace(tmp, fn)                                 # atomic: an interruption never corrupts fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--variants', default='pos,qk')
    ap.add_argument('--out', default='patchmem/logs/server'); ap.add_argument('--smoke', action='store_true')
    a_ = ap.parse_args()
    os.makedirs(a_.out, exist_ok=True)
    seeds = [int(s) for s in a_.seeds.split(',')][:1 if a_.smoke else None]
    for seed in seeds:
        fn = f'{a_.out}/tefl_att_{a_.data}_s{seed}{"_smoke" if a_.smoke else ""}.json'
        res = json.load(open(fn)) if os.path.exists(fn) else {}                        # resume: finished runs are skipped, an interrupted one restarts
        if 'T0' not in res:
            res['T0'] = baseline(a_.data, seed, a_.smoke); dump(res, fn)
        for v in a_.variants.split(','):
            if v in res:
                print(f'[skip] {a_.data} {v} s{seed} already done', flush=True); continue
            res[v] = train_one(a_.data, seed, v, a_.out, a_.smoke); dump(res, fn)
        c = '1440/336'
        print(f'===== {a_.data} s{seed} test {c}: T0 {res["T0"][c]:.4f} | ' + ' | '.join(
            f'{v} {res[v]["normal"][c]:.4f} (off {res[v]["off"][c]:.4f} perm {res[v]["perm"][c]:.4f})' for v in a_.variants.split(',') if v in res), flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
