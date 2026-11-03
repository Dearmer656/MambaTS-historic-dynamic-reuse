"""Zero-carry reference for eval_sr train_loss: mean E^2 over the carry training targets (k in [burn, k_tr_end))."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
import patchmem.eval_freshcast as fc
from patchmem.eval_err_value import H, L
from patchmem.eval_itransformer_pure_icl import load_base

for seed in map(int, sys.argv[1].split(',')):
    exp, _, _ = load_base('T', seed)
    X, M, n_tr, n_va = ev.load_series(exp, 'T')
    T0 = max(L, fc.LQ) + 1
    E = ev.stream_errors(exp, X, M, T0, 64)                                       # n H C
    times = torch.arange(T0, T0 + E.shape[0])
    k_tr_end = int(torch.nonzero(times <= n_tr - H).max()) + 1
    burn = 2 * 168 + H
    e = E[burn:k_tr_end].double()
    diag0 = e[:, 0].square().mean()
    print(f'[zero] s{seed} H{H} L{L} train E^2 {float(e.square().mean()):.5f} lead0 {float(diag0):.5f} n_k {k_tr_end - burn}', flush=True)
