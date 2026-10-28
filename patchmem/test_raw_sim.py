import torch

import patchmem.eval_raw_sim as rs
from patchmem.test_error_path import synth

AGES = [192, 200, 230, 264]


def test_sums_match_bruteforce():
    data = synth(8, T=1400, C=2)
    data['origins'] = torch.arange(900, 1000)
    combos = [('cos', 0.0, 0.0), ('cos', 2.0, 1.0), ('l2', 4.0, 0.5)]
    acc = rs.sums(data, AGES, 'cpu', 37, combos)
    E, X, H, L = data['E'], data['X'], rs.H, rs.L
    for kind, te, ty in combos:
        tt = te_sum = ee = 0.0
        for t in data['origins'].tolist():
            for c in range(2):
                q, tg = E[t - H - L, :, c].double(), E[t - L, :, c].double()
                yq = torch.stack([X[t - 1 - k, c] for k in range(rs.W)]).double()
                sc, V = [], []
                for a in AGES:
                    s = t - a
                    k_ = E[s - H - L, :, c].double()
                    ys = torch.stack([X[s - 1 - k, c] for k in range(rs.W)]).double()
                    f = (lambda u, v: torch.nn.functional.cosine_similarity(u, v, dim=0)) if kind == 'cos' else \
                        (lambda u, v: -(u - v).square().mean())
                    sc.append(te * f(q, k_) + ty * f(yq, ys))
                    V.append(E[s - L, :, c].double())
                w = torch.softmax(torch.stack(sc), 0)
                eh = (w[:, None] * torch.stack(V)).sum(0)
                tt += float(tg.square().sum()); te_sum += float((tg * eh).sum()); ee += float(eh.square().sum())
        a = acc[kind, te, ty]
        assert abs(a[0] - tt) < 1e-6 * tt and abs(a[1] - te_sum) < 1e-6 * abs(te_sum) + 1e-6 and abs(a[2] - ee) < 1e-6 * ee
        s = 0.7
        assert abs(rs.mse_at(a, s) - (tt - 2 * s * te_sum + s * s * ee) / float(a[3])) < 1e-6
