"""PAT-315 B key-window sweep: the fixed same-hour-of-week B retrieval of eval_freshcast (config from the obsicl2 json:
P, readout, beta, a, c) with its query / key window R in {24, 48, 96, 168, 336, 720} (default 96).  Saves the raw B
correction y_B - f for every val + test origin as ext npz (keys B24, B48, ...) for eval_sr --ext; R = 96 is checked
against the eval_freshcast ext B.
Usage: ITR_L=720 ITR_H=192 python -m patchmem.eval_bwin --seed 3047
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, L
import patchmem.eval_obs_icl as oi
import patchmem.eval_freshcast as efc

WINS = (24, 48, 96, 168, 336, 720)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--fc_dir', default='patchmem/logs/server/error_path')
    p.add_argument('--ref_ext', default='patchmem/logs/server/err_stack/fc')
    p.add_argument('--out_dir', default='patchmem/logs/server/err_bwin')
    a = p.parse_args()
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base('T', a.seed)
    dev = exp.device
    X, Mk, n_tr, n_va = ev.load_series(exp, 'T')
    C = X.shape[1]
    T0 = L + 1
    E = ev.stream_errors(exp, X, Mk, T0, a.bs)                                     # n H C, origins T0 ..
    kh = torch.arange(H)
    ext_o = torch.cat([torch.arange(n_tr, n_tr + n_va - H + 1), torch.arange(n_tr + n_va, len(X) - H + 1)])
    Fd = X[ext_o[:, None] + kh] - E[ext_o - T0]                                    # backbone forecasts (rows H C)
    del E
    g = json.loads((Path(a.fc_dir) / f'T_H{H}_L{L}_obsicl2_s{a.seed}.json').read_text())['families']['phase:B']['global']
    Xd = X.to(dev)
    ost = oi.rolling(Xd, g['P'])
    ages = torch.as_tensor(oi.pool_cands('phase')[0], device=dev)
    print(f'[B] P {g["P"]} readout {g["readout"]} beta {g["hyper"]} a {g["a"]:.4f} c {g["c"]:.4f} ages {len(ages)} '
          f'{int(ages.min())}..{int(ages.max())}', flush=True)
    ext = {f'B{w}': np.zeros((len(ext_o), C, H), dtype=np.float16) for w in WINS}
    call = torch.arange(C, device=dev)
    with torch.no_grad():
        for w in WINS:
            oi.R = w                                                               # b_arm reads oi.R at call time
            for i, t0 in enumerate(ext_o.tolist()):
                t = torch.full_like(call, t0)
                f = Fd[i].to(dev).T                                                # C H
                mt, st, vh = efc.b_arm(Xd, ost, t, call, ages, g['readout'], g['hyper'])
                yb = f + g['a'] * (mt - f) + g['c'] * st * vh
                ext[f'B{w}'][i] = (yb - f).half().cpu().numpy()
            print(f'[B{w}] done', flush=True)
    oi.R = 96
    ref = np.load(Path(a.ref_ext) / f'T_H{H}_L{L}_s{a.seed}_ext.npz')
    assert (ref['origins'] == ext_o.numpy()).all()
    d = float(np.abs(ref['B'].astype(np.float32) - ext['B96'].astype(np.float32)).max())
    print(f'[check] B96 vs eval_freshcast B: max |diff| {d:.2e}', flush=True)
    assert d < 1e-2, d
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f'T_H{H}_L{L}_s{a.seed}_ext.npz', origins=ext_o.numpy(), **ext)
    print(f'[saved] {od}', flush=True)


if __name__ == '__main__':
    main()
