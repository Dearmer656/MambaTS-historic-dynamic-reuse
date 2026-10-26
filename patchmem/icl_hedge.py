"""PAT-313 plan 1: online expert weighting across windows (Hedge) over the 4 one-step fixed-set fits, offline on the dumps of
patchmem/slurm/icl_sel2.sbatch (fixed before results, see the issue). Mixture error ~ sum_S w_S err_S (checked in step 0).
Causal: at window t only windows u <= t - H (H = 96 steps; stride 1 step) have observed targets.
  loss_{S,u,c} = mean over leads of err^2;  L_S(t) = sum_u gamma^(t-H-u) loss_{S,u};  w = softmax(-beta * L_S / mean_S L_S)
(beta, gamma) chosen on the first half of the test windows, reported on the second half, with iclselavg1, iclsel1, the best single
expert and the per-variable best expert in hindsight (cheating) on the same half. Variant hedge x valid: w ~ w_hedge * w_iclsel1.
usage: python patchmem/icl_hedge.py"""
import itertools, json, os
import torch

O = 'patchmem/logs/server/icl_sel2'
SETS = ('all', 'recent4', 'phase', 'topk8')
H = 96
BETAS = (1, 3, 10, 30, 100)
GAMMAS = (0.9, 0.97, 0.99, 0.997, 1.0)


def load(D, s, mode):
    return torch.load(f'{O}/{D}_s{s}_test_L1440H96_{mode}.pt')


def disc_loss(loss, gamma):
    """loss (S, N, C) -> L (S, N, C): L[:, t] = sum_{u <= t-H} gamma^(t-H-u) loss[:, u]; 0 for t < H."""
    S, N, C = loss.shape
    L = torch.zeros_like(loss)
    acc = torch.zeros(S, C, dtype=loss.dtype)
    for t in range(H, N):
        acc = gamma * acc + loss[:, t - H]
        L[:, t] = acc
    return L


def hedge_w(L, beta):
    m = L.mean(0, keepdim=True)
    w = torch.softmax(-beta * L / m.clamp_min(1e-12), dim=0)
    w[:, :H] = 1.0 / L.shape[0]
    return w


def mix_mse(err, w, sl):
    """err (S, N, 96, C), w (S, N, C) -> MSE of sum_S w err over windows sl."""
    return float((w[:, sl, None, :] * err[:, sl]).sum(0).square().mean())


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    res = {}
    for D in ('W', 'T'):
        for s in (3047, 3048, 3049):
            err = torch.stack([load(D, s, f'iclselfix1_{k}')['err'].double() for k in SETS])   # (S, N, 96, C)
            S, N = err.shape[:2]
            sel1 = load(D, s, 'iclsel1'); avg1 = load(D, s, 'iclselavg1')
            wv = sel1['w'].double()                                                       # (S, N, C)
            e_sel, e_avg = sel1['err'].double(), avg1['err'].double()
            lin_sel = float(((wv[:, :, None, :] * err).sum(0) - e_sel).norm() / e_sel.norm())
            lin_avg = float((err.mean(0) - e_avg).norm() / e_avg.norm())
            loss = err.square().mean(2)                                                   # (S, N, C)
            h1, h2 = slice(0, N // 2), slice(N // 2, N)
            grid = {}
            for g in GAMMAS:
                L = disc_loss(loss, g)
                for b in BETAS:
                    w = hedge_w(L, b)
                    grid[(b, g)] = (w, mix_mse(err, w, h1), mix_mse(err, w * wv / (w * wv).sum(0, keepdim=True), h1))
            bh = min(grid, key=lambda k: grid[k][1]); bx = min(grid, key=lambda k: grid[k][2])
            w_h = grid[bh][0]; w_x = grid[bx][0] * wv; w_x = w_x / w_x.sum(0, keepdim=True)
            mse2 = lambda e: float(e[h2].square().mean())
            single = {k: float(err[i, h2].square().mean()) for i, k in enumerate(SETS)}
            pv = err[:, h2].square().mean((1, 2))                                         # (S, C): per-variable expert MSE on half 2
            hind = float(torch.stack([err[pv[:, c].argmin(), h2, :, c] for c in range(err.shape[-1])], -1).square().mean())
            uni = torch.full_like(wv, 1.0 / S)
            ap = dict(avg1=mix_mse(err, uni, h2), sel1=mix_mse(err, wv, h2))                 # same linear approximation as hedge
            r = dict(approx_half2=ap, lin_gap_sel1=lin_sel, lin_gap_avg1=lin_avg, beta_gamma_hedge=bh, beta_gamma_x=bx,
                     half2=dict(avg1=mse2(e_avg), sel1=mse2(e_sel), hedge=mix_mse(err, w_h, h2), hedge_x_valid=mix_mse(err, w_x, h2),
                                best_single=min(single.values()), best_single_set=min(single, key=single.get), hindsight_per_var=hind),
                     hedge_mean_w={k: float(w_h[i, h2].mean()) for i, k in enumerate(SETS)})
            res[f'{D}_s{s}'] = r
            q = r['half2']
            rel = lambda v: f'{v:.4f} ({100 * (v / q["avg1"] - 1):+.2f}%)'
            print(f'{D} s{s} | linearity gap sel1 {lin_sel:.4f} avg1 {lin_avg:.4f} | chosen (beta, gamma) hedge {bh} x {bx} | 2nd half MSE: '
                  f'avg1 {q["avg1"]:.4f} | sel1 {rel(q["sel1"])} | hedge {rel(q["hedge"])} | hedge x valid {rel(q["hedge_x_valid"])} | '
                  f'best single ({q["best_single_set"]}) {rel(q["best_single"])} | hindsight per var {rel(q["hindsight_per_var"])}', flush=True)
            ra = lambda v: f'{v:.4f} ({100 * (v / ap["avg1"] - 1):+.2f}%)'
            print(f'   SAME APPROXIMATION (sum_S w err_S) 2nd half: avg1 {ap["avg1"]:.4f} | sel1 {ra(ap["sel1"])} | hedge {ra(q["hedge"])} | '
                  f'hedge x valid {ra(q["hedge_x_valid"])} | best single {ra(q["best_single"])} | hindsight per var {ra(q["hindsight_per_var"])}', flush=True)
            print('   hedge mean weights (2nd half) ' + ' '.join(f'{k} {v:.3f}' for k, v in r['hedge_mean_w'].items()), flush=True)
    json.dump({k: {**v, 'beta_gamma_hedge': list(v['beta_gamma_hedge']), 'beta_gamma_x': list(v['beta_gamma_x'])} for k, v in res.items()},
              open(f'{O}/icl_hedge_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
