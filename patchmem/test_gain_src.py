"""CPU tests for patchmem/gain_src.py."""
import torch
import patchmem.efc_cet as ec
import patchmem.gain_src as gs


def test_oracle_exact_linear():
    g = torch.Generator().manual_seed(0)
    F = torch.randn(400, 3, 4, generator=g) * 10; A = torch.randn(4, 4, generator=g)
    sh, W = gs.oracle(F, F @ A)
    assert sh > 0.999 and torch.allclose(W.float(), A, atol=1e-2)
    s = gs.w_struct(torch.eye(6, dtype=torch.float64))
    assert abs(s['diag'] - 1) < 1e-9 and abs(s['top1'] - 1 / 6) < 1e-9


def test_lag_finds_period():
    """Errors that repeat with period 6: softmax weights concentrate on lags that are multiples of 6."""
    g = torch.Generator().manual_seed(1)
    pat = torch.randn(6, 7, 2, generator=g)
    E = torch.stack([pat[t % 6] for t in range(300)]) + 0.01 * torch.randn(300, 7, 2, generator=g)
    st = gs.lag_stats(E, 3, 100, 290, 60, 6, 12)
    Er = torch.randn(300, 7, 2, generator=g)
    st2 = gs.lag_stats(Er, 3, 100, 290, 60, 6, 12)
    assert st['day_ratio'] > 1.3 and st2['day_ratio'] < 1.15, (st, st2)                  # soft1 is broad: ~1.5 even for exact period
    assert st['kr_day_ratio'] > st['day_ratio'] and st2['kr_day_ratio'] < 1.3, (st, st2)  # kernel ridge weights sharper


def test_analyse_runs_and_matches_win_row():
    g = torch.Generator().manual_seed(2)
    E = torch.randn(260, 7, 2, generator=g)
    o = gs.analyse(E, 3, 40, 100, 250, 'E', krexp=True)
    c, _ = ec.stream(E[:251], 3, 100, 3, True, 'row', 40)
    y = E[100:251, :3]
    assert abs(o['win_row'] - float((y - c[100:251]).square().mean())) < 1e-6
    assert abs(sum(o['lead_base']) / 3 - o['base']) < 1e-6 and len(o['var_win_row']) == 2


if __name__ == '__main__':
    test_oracle_exact_linear(); test_lag_finds_period(); test_analyse_runs_and_matches_win_row(); print('ok')
