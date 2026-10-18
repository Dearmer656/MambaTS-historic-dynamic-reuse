"""PAT-306 baseline sanity audit, eval-only part (no training). Checks on the trained ETTh2 checkpoints that the memory gain over A1 is not
an artefact of a bug in the evaluation or of the baseline:
  E1  leakage: the forecast of a window depends only on its own lookback -- identical in a batch of 64 and alone, identical when the other
      samples of the batch are replaced by noise, identical before / after other batches (no state carried across forwards); the dataset
      layout puts the lookback strictly before the target (seq_x = data[s:s+L], target = data[s+L:s+L+H])
  E2  metric: A1's eval_grid MSE / MAE at (720, 96) == the official test loader (batch 1, drop_last) + utils.metrics.metric on the same
      model, same window count
  E3  pipeline: the original MambaTS (A0, H=96 checkpoint) run through R1's eval_grid reproduces its official test MSE (0.2813)
  E4  scan order: A1 and Q8+drop50 of each seed evaluated on the test grid under their own pi (reproduces the stored numbers) and under
      the other model's pi (shared-pi paired gain); A1 of one seed under random pi (order sensitivity)
  E5  A1 training curves (from the logs): best epoch, val at stop
usage (repo root): python patchmem/sanity_eval.py [--parts E1,E2,E3,E4,E5] [--seeds 3047,3048,3049] [--out f.json]"""
import argparse, copy, json, os, re, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import set_seed

CK = 'long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_{}s{}_0'
ARMS = {'A1': ('R1h0', {}), 'Q8': ('R1MK0.5Q8c1h0', {'R1_MEM': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'})}
A0_SETTING = 'long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0_0'
A0_ARGS = ('--task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 '
           '--c_out 7 --model_id ETTh2_720_96 --model MambaTS --data ETTh2 --features M --seq_len 720 --label_len 48 --pred_len 96 '
           '--e_layers 5 --d_layers 2 --factor 1 --des ExpA0 --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 '
           '--VPT_mode 1 --ATSP_solver SA --batch_size 8 --learning_rate 0.001').split()
ENV_KEYS = ('R1_MEM', 'R1_MEM_QK_DIM', 'R1_MEM_SLOT_KEEP', 'R1_EVAL_PERM')
GRID = [(L, H) for L in (384, 720, 1440) for H in (96, 192, 336)]


def load(arm, seed):
    """Exp_R1 with the arm's trained checkpoint, eval mode, its own deterministic eval order."""
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    des, env = ARMS[arm]
    os.environ.update(env)
    from patchmem.exp_r1 import Exp_R1
    args = official_args(ETTH2 + ['--seed', str(seed), '--des', f'{des}s{seed}'])
    exp = Exp_R1(args)
    m = exp.model
    m.load_state_dict(torch.load(os.path.join(args.checkpoints, CK.format(des, seed), 'checkpoint.pth'), map_location=exp.device))
    m.eval(); m.reset_ids_shuffle(); exp._fix_eval_order()
    return exp


def stored(arm, seed):
    rec = None
    for l in open(f'patchmem/logs/diag/ETTh2_r1_{ARMS[arm][0]}s{seed}_s{seed}.jsonl'):
        r = json.loads(l)
        if r.get('record') == 'test' and r.get('flag') == 'test':
            rec = r
    return rec


@torch.no_grad()
def e1(seed, out):
    for arm in ARMS:
        exp = load(arm, seed)
        m, dev = exp.model, exp.device
        for L, H in ((720, 96), (1440, 336)):
            ds = exp._ds('test', L, H)
            s = 100
            sx, sy = ds[s][0], ds[s][1]
            lay = (np.array_equal(sx, ds.data_x[s:s + L]) and np.array_equal(sy[-H:], ds.data_y[s + L:s + L + H]))
            bx, _ = exp._batch(ds, list(range(64)))
            bx = bx.to(dev)
            p64 = m(bx, H // m.P)['pred']
            p1 = torch.cat([m(bx[i:i + 1], H // m.P)['pred'] for i in range(64)])
            noise = bx.clone(); noise[1:] = torch.randn_like(noise[1:]) * 5
            pn = m(noise, H // m.P)['pred'][0]
            bx2, _ = exp._batch(ds, list(range(64, 128)))
            m(bx2.to(dev), H // m.P)
            pa = m(bx, H // m.P)['pred']
            r = dict(layout_ok=bool(lay), batch64_vs_alone=float((p64 - p1).abs().max()),
                     others_noise=float((pn - p64[0]).abs().max()), after_other_batch=float((pa - p64).abs().max()),
                     pred_scale=float(p64.abs().mean()))
            out[f'E1/{arm}/s{seed}/{L}/{H}'] = r
            print(f'[E1] {arm} s{seed} L={L} H={H} | layout ok {lay} | max|diff| batch64 vs alone {r["batch64_vs_alone"]:.2e} | '
                  f'others->noise {r["others_noise"]:.2e} | after another batch {r["after_other_batch"]:.2e} | mean|pred| {r["pred_scale"]:.3f}',
                  flush=True)


@torch.no_grad()
def e2(seed, out):
    exp = load('A1', seed)
    m, dev = exp.model, exp.device
    g = exp.eval_grid('test', [(720, 96)])[(720, 96)]
    a = copy.copy(exp.args); a.seq_len, a.pred_len = 720, 96
    _, loader = data_provider(a, 'test')
    preds, trues = [], []
    for bx, by, _, _ in loader:
        preds.append(m(bx.float().to(dev), 2)['pred'].cpu().numpy()); trues.append(by[:, -96:].numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    mae, mse = metric(preds, trues)[:2]
    st = stored('A1', seed)['grid']['720/96']
    out[f'E2/A1/s{seed}'] = dict(grid_mse=g['mse'], grid_mae=g['mae'], grid_windows=g['windows'], official_mse=float(mse),
                                 official_mae=float(mae), official_windows=len(preds), stored_mse=st['mse'])
    print(f'[E2] A1 s{seed} L=720 H=96 | eval_grid mse {g["mse"]:.6f} mae {g["mae"]:.6f} ({g["windows"]} windows) | official loader + '
          f'metric mse {mse:.6f} mae {mae:.6f} ({len(preds)} windows) | stored at training {st["mse"]:.6f}', flush=True)


class A0Adapter(torch.nn.Module):
    """The original MambaTS (direct head) behind R1's call signature m(x, n_patches) -> {'pred'}; its ATSP eval order is solved once, as in
    exp.test (set_seed(seed), first batch of size 1), then reused for every batch size."""
    def __init__(self, model, P=48, c=7):
        super().__init__()
        self.model, self.P, self.c, self.vpt, self.ids_shuffle, self.order = model, P, c, False, None, None

    def forward(self, x, n, trace=None):
        enc = self.model.encoder
        if self.order is None:
            set_seed(3047)
            self.model.reset_ids_shuffle()
            self.model(x[:1], None, None, None)
            self.order = enc.ids_shuffle[0].clone()
        enc.ids_shuffle = self.order.repeat(x.shape[0], 1)
        return dict(pred=self.model(x, None, None, None)[:, :n * self.P])


@torch.no_grad()
def e3(out):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    from patchmem.exp_r1 import Exp_R1
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    a0 = Exp_Long_Term_Forecast(official_args(A0_ARGS))
    a0.model.load_state_dict(torch.load(os.path.join(a0.args.checkpoints, A0_SETTING, 'checkpoint.pth'), map_location=a0.device))
    a0.model.eval()
    exp = Exp_R1(official_args(ETTH2 + ['--des', 'sanityA0']))
    exp.model = A0Adapter(a0.model).to(exp.device)
    g = exp.eval_grid('test', [(720, 96)])[(720, 96)]
    out['E3/A0'] = dict(mse=g['mse'], mae=g['mae'], windows=g['windows'], order=exp.model.order.tolist(), official_mse=0.28125518560409546)
    print(f'[E3] A0 (H=96 ckpt) through R1 eval_grid L=720 H=96 | mse {g["mse"]:.6f} mae {g["mae"]:.6f} ({g["windows"]} windows) | '
          f'order {exp.model.order.tolist()} | official test 0.281255 / 0.346981', flush=True)


def _grid(exp, order):
    exp.model.ids_shuffle = list(order)
    return {f'{L}/{H}': v for (L, H), v in exp.eval_grid('test', GRID).items()}


def e4(seeds, out, n_rand=3):
    for s in seeds:
        own = {arm: stored(arm, s)['order'] for arm in ARMS}
        res = {}
        for arm in ARMS:
            exp = load(arm, s)
            assert exp.eval_order() == own[arm], (exp.eval_order(), own[arm])
            for pi_arm in ARMS:
                res[(arm, pi_arm)] = _grid(exp, own[pi_arm])
            if arm == 'A1' and s == seeds[0]:
                rng = np.random.default_rng(0)
                for r in range(n_rand):
                    o = rng.permutation(7).tolist()
                    g = _grid(exp, o)
                    out[f'E4/A1rand/s{s}/{o}'] = {k: v['mse'] for k, v in g.items()}
                    print(f'[E4] A1 s{s} random pi {o} | test mse ' + ' '.join(f'{k}:{v["mse"]:.4f}' for k, v in g.items()), flush=True)
        st = {arm: stored(arm, s)['grid'] for arm in ARMS}
        rep = max(abs(res[(arm, arm)][k]['mse'] - st[arm][k]['mse']) for arm in ARMS for k in st[arm])
        print(f'[E4] s{s} own-pi re-evaluation vs stored test numbers: max |diff| {rep:.2e} | pi A1 {own["A1"]} Q8 {own["Q8"]}', flush=True)
        for pi_arm in ARMS:
            gains = {k: 100 * (res[('A1', pi_arm)][k]['mse'] - res[('Q8', pi_arm)][k]['mse']) / res[('A1', pi_arm)][k]['mse']
                     for k in res[('A1', pi_arm)]}
            print(f'[E4] s{s} shared pi = {pi_arm}\'s {own[pi_arm]} | Q8 gain over A1 % ' + ' '.join(f'{k}:{v:+.2f}' for k, v in gains.items()),
                  flush=True)
            out[f'E4/s{s}/pi_{pi_arm}'] = dict(order=own[pi_arm], gain=gains,
                                                 mse={arm: {k: v['mse'] for k, v in res[(arm, pi_arm)].items()} for arm in ARMS})
        out[f'E4/s{s}/own_vs_stored_maxdiff'] = rep


def e5(seeds, out):
    for s in seeds:
        ep = [(int(m.group(1)), float(m.group(2))) for m in
              re.finditer(r'Epoch: (\d+) cost time.*?val mean mse ([\d.]+)', open(f'patchmem/logs/stage3/A1_s{s}.txt').read())]
        best = min(ep, key=lambda t: t[1])
        out[f'E5/A1/s{s}'] = dict(epochs=ep, best_epoch=best[0])
        print(f'[E5] A1 s{s} | val mean mse per epoch ' + ' '.join(f'{e}:{v:.4f}' for e, v in ep) + f' | best epoch {best[0]} '
              f'({best[1]:.4f}) | last {ep[-1][0]} ({ep[-1][1]:.4f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='E5,E1,E2,E3,E4')
    ap.add_argument('--seeds', default='3047,3048,3049')
    ap.add_argument('--out', default='patchmem/logs/sanity/sanity_eval.json')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    parts = a.parts.split(',')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = {}
    for p in parts:
        if p == 'E1':
            e1(seeds[0], out)
        elif p == 'E2':
            e2(seeds[0], out)
        elif p == 'E3':
            e3(out)
        elif p == 'E4':
            e4(seeds, out)
        elif p == 'E5':
            e5(seeds, out)
        json.dump(out, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
