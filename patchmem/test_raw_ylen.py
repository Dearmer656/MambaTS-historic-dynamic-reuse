import torch

import patchmem.eval_raw_sim as rs
import patchmem.eval_raw_ylen as ry
from patchmem.test_error_path import synth

AGES = [192, 200, 230, 264]


def test_e_y24_matches_raw_sim():
    data = synth(11, T=1600, C=2)
    data['origins'] = torch.arange(1000, 1060)
    a = ry.sums(data, AGES, 'cpu', 25, [('e+y24', (2.0, 4.0))])
    b = rs.sums(data, AGES, 'cpu', 25, [('cos', 2.0, 4.0)])
    assert torch.allclose(a['e+y24', (2.0, 4.0)], b['cos', 2.0, 4.0], rtol=1e-9)


def test_y192c_window_and_centering():
    """Two demos, tau on y192c only: e_hat weights follow softmax of the explicit centred cosine of y[t-H:t]."""
    data = synth(12, T=1600, C=2)
    t, ages = 1030, [200, 264]
    data['origins'] = torch.tensor([t])
    data['X'] = data['X'][:, :].clone()
    keep = {f: v for f, v in ry.FAMILIES.items()}
    ry.FAMILIES['probe'] = ('y192c',)
    try:
        r = ry.sums(data, ages, 'cpu', 8, [('probe', (3.0,))], None, 0)
    finally:
        ry.FAMILIES.clear(); ry.FAMILIES.update(keep)
    X, E, H, L = data['X'].double(), data['E'].double(), ry.H, ry.L
    want = 0.0
    for c in range(X.shape[1]):                                          # all_cases covers every channel
        yq = X[t - H:t, c]
        sims = []
        for a in ages:
            ys = X[t - a - H:t - a, c]
            sims.append(torch.nn.functional.cosine_similarity(yq - yq.mean(), ys - ys.mean(), dim=0))
        w = torch.softmax(3.0 * torch.stack(sims), 0)
        eh = sum(w[j] * E[t - a - L, :, c] for j, a in enumerate(ages))
        want += float((E[t - L, :, c] * eh).sum())
    got = float(r['probe', (3.0,)][1])
    assert abs(got - want) < 1e-4 * abs(want), (got, want)
