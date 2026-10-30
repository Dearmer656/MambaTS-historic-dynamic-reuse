import torch

import patchmem.eval_dense_multistep as dm
import patchmem.eval_raw_multistep as ms
from patchmem.test_error_path import synth

AGES = [192, 200, 230, 264]


def test_matches_explicit_recursion():
    """beta = 1: the u_k form equals eval_raw_multistep's explicit demo-update recursion for every K and mode."""
    data = synth(16, T=1600, C=2)
    data['origins'] = torch.arange(1000, 1030)
    for mode in ('self', 'loo'):
        for K in (1, 2, 3, 5):
            a = dm.sums(data, AGES, 'cpu', 8, {(mode, 1.0, 0.5, K)})[mode, 1.0, 0.5, K]
            b = ms.sums(data, AGES, 'cpu', 8, {(mode, 0.5, K)})[mode, 0.5, K]
            assert torch.allclose(a, b, rtol=1e-4), (mode, K, a, b)


def test_beta_scales_logits():
    """beta = 2 equals tau doubled: compare with the explicit recursion run on doubled temperatures."""
    data = synth(17, T=1600, C=2)
    data['origins'] = torch.arange(1000, 1020)
    a = dm.sums(data, AGES, 'cpu', 8, {('self', 2.0, 0.3, 3)})['self', 2.0, 0.3, 3]
    old = ms.TAU_E, ms.TAU_Y
    ms.TAU_E, ms.TAU_Y = 2 * old[0], 2 * old[1]
    try:
        b = ms.sums(data, AGES, 'cpu', 8, {('self', 0.3, 3)})['self', 0.3, 3]
    finally:
        ms.TAU_E, ms.TAU_Y = old
    assert torch.allclose(a, b, rtol=1e-4)


def test_pool_ages_causal():
    for pool in dm.POOLS:
        ages = dm.pool_ages(pool, 3504)
        assert min(ages) >= dm.H and max(ages) <= 3504 - dm.H - dm.L
        assert all(a % dm.POOLS[pool] == 0 for a in ages)
