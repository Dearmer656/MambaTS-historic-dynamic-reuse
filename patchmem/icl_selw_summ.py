"""PAT-313 plan 1 in-model check: real test errors of iclselw_hedge / iclselw_hx (job icl_selw) vs iclselavg1 / iclsel1 / hyb
(job icl_sel2 dumps), MSE on the second half of the test windows (Hedge's beta, gamma were chosen on the first half) and on all.
usage: python patchmem/icl_selw_summ.py"""
import os
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for D in ('W', 'T'):
    for s in (3047, 3048, 3049):
        ld = lambda d, m: torch.load(f'patchmem/logs/server/{d}/{D}_s{s}_test_L1440H96_{m}.pt')['err'].double()
        e = dict(hyb=ld('icl_sel2', 'normal'), avg1=ld('icl_sel2', 'iclselavg1'), sel1=ld('icl_sel2', 'iclsel1'),
                 hedge=ld('icl_selw', 'iclselw_hedge'), hx=ld('icl_selw', 'iclselw_hx'))
        N = e['hyb'].shape[0]
        assert all(v.shape[0] == N for v in e.values())
        h2 = slice(N // 2, N)
        m2 = {k: float(v[h2].square().mean()) for k, v in e.items()}
        ma = {k: float(v.square().mean()) for k, v in e.items()}
        r = lambda d, k: f'{d[k]:.4f} ({100 * (d[k] / d["avg1"] - 1):+.2f}%)'
        print(f'{D} s{s} | half 2: hyb {m2["hyb"]:.4f} avg1 {m2["avg1"]:.4f} sel1 {r(m2, "sel1")} hedge {r(m2, "hedge")} hedge x valid {r(m2, "hx")} | '
              f'all windows: hyb {ma["hyb"]:.4f} avg1 {ma["avg1"]:.4f} sel1 {ma["sel1"]:.4f} hedge {ma["hedge"]:.4f} hx {ma["hx"]:.4f}', flush=True)
