"""PAT-313 point 6 follow-up: is the select-then-adapt gain per-case selection or an ensemble? (fixed before results, see the issue)
Inputs (patchmem/slurm/icl_sel2.sbatch): {D}_s{seed}_test_L1440H96_{mode}.pt with err = pred - y (N, 96, C) for every iclsel* mode
and w (S, N, C) for iclsel / iclsel1.
  1 MSE of iclsel1 vs iclselavg1 and iclsel vs iclselavg (validated vs uniform mix)
  2 spread of w across windows: mean max_S w, mean KL(w || uniform), std of each w_S; gain over the uniform mix per (window, variable)
    (MSE_avg1 - MSE_sel1 over the 96 leads) by KL quartile
  3 argmax_S w vs argmin_S of the per-(window, variable) MSE of the one-step fixed sets (chance 25%); the oracle per-case pick's MSE
usage: python patchmem/icl_sel_summ.py"""
import os
import torch

O = 'patchmem/logs/server/icl_sel2'
SETS = ('all', 'recent4', 'phase', 'topk8')


def load(D, s, mode):
    return torch.load(f'{O}/{D}_s{s}_test_L1440H96_{mode}.pt')


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for D in ('W', 'T'):
        for s in (3047, 3048, 3049):
            e = {m: load(D, s, m)['err'].double() for m in ('iclsel1', 'iclselavg1', 'iclsel', 'iclselavg') + tuple(f'iclselfix1_{k}' for k in SETS)}
            mse = {m: float(v.square().mean()) for m, v in e.items()}
            w = load(D, s, 'iclsel1')['w'].double()                                     # (S, N, C)
            maxw = float(w.max(0).values.mean())
            kl = (w * (w.clamp_min(1e-12) * len(SETS)).log()).sum(0)                    # (N, C)
            sd = w.std((1, 2)).tolist()
            pc = lambda m: e[m].square().mean(1)                                         # (N, C) per window and variable
            g = pc('iclselavg1') - pc('iclsel1')
            qs = torch.quantile(kl.flatten(), torch.tensor([.25, .5, .75], dtype=torch.float64))
            bins = torch.bucketize(kl, qs)
            gq = [float(g[bins == i].mean()) for i in range(4)]
            fixed = torch.stack([pc(f'iclselfix1_{k}') for k in SETS])                   # (S, N, C)
            agree = float((w.argmax(0) == fixed.argmin(0)).double().mean())
            oracle = float(fixed.min(0).values.mean())
            print(f'{D} s{s} | MSE sel1 {mse["iclsel1"]:.4f} vs avg1 {mse["iclselavg1"]:.4f} ({100 * (mse["iclsel1"] / mse["iclselavg1"] - 1):+.2f}%) | '
                  f'sel {mse["iclsel"]:.4f} vs avg {mse["iclselavg"]:.4f} ({100 * (mse["iclsel"] / mse["iclselavg"] - 1):+.2f}%) | fixed1 ' +
                  ' '.join(f'{k} {mse[f"iclselfix1_{k}"]:.4f}' for k in SETS), flush=True)
            print(f'   weights: mean max {maxw:.3f} (uniform .250), mean KL {float(kl.mean()):.4f}, std ' + ' '.join(f'{k} {v:.3f}' for k, v in zip(SETS, sd)) +
                  ' | gain over uniform by KL quartile (low -> high) ' + ' '.join(f'{v:+.5f}' for v in gq) +
                  f' | argmax w = best fixed set {agree:.3f} (chance .250) | oracle per-case pick MSE {oracle:.4f}', flush=True)


if __name__ == '__main__':
    main()
