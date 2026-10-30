import torch

import patchmem.eval_raw_multistep as ms
import patchmem.eval_raw_ylen as ry
from patchmem.test_error_path import synth

AGES = [192, 200, 230, 264]


def test_k1_matches_one_step_attention():
    data = synth(14, T=1600, C=2)
    data['origins'] = torch.arange(1000, 1040)
    a = ms.sums(data, AGES, 'cpu', 16, {('self', 0.5, 1)})['self', 0.5, 1]
    b = ry.sums(data, AGES, 'cpu', 16, [('e+y96', (1.0, 4.0))])['e+y96', (1.0, 4.0)]
    assert torch.allclose(a[1], 0.5 * b[1], rtol=1e-5) and torch.allclose(a[2], 0.25 * b[2], rtol=1e-5)


def test_k2_explicit_kernel_gd():
    data = synth(15, T=1600, C=1)
    t, ages, eta = 1100, [200, 230, 264], 0.7
    data['origins'] = torch.tensor([t])
    E, X, H, L = data['E'].double(), data['X'].double(), ms.H, ms.L
    def key(o):
        return E[o - H - L, :, 0], X[o - 96:o, 0]
    def logit(o1, o2):
        e1, y1 = key(o1); e2, y2 = key(o2)
        cs = torch.nn.functional.cosine_similarity
        return cs(e1, e2, dim=0) + 4 * cs(y1, y2, dim=0)
    S = [t - a for a in ages]
    V = torch.stack([E[s - L, :, 0] for s in S])
    aq = torch.softmax(torch.stack([logit(t, s) for s in S]), 0)
    for mode in ('self', 'loo'):
        Ld = torch.stack([torch.stack([logit(si, sj) for sj in S]) for si in S])
        if mode == 'loo':
            Ld.fill_diagonal_(float('-inf'))
        ad = torch.softmax(Ld, 1)
        fq1, fd1 = eta * aq @ V, eta * ad @ V
        fq2 = fq1 + eta * aq @ (V - fd1)
        got = ms.sums(data, ages, 'cpu', 4, {(mode, eta, 2)})[mode, eta, 2]
        tg = E[t - L, :, 0]
        assert abs(float(got[1]) - float(tg @ fq2)) < 1e-4 * abs(float(tg @ fq2)), mode
