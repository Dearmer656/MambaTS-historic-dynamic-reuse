"""CPU tests for the R1TeflAtt attention term (extra_delta, called on a stub: the backbone needs the CUDA scan) and the memory bank
indexing of patchmem/tefl_att.py."""
import math
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from patchmem.r1_tefl_att import R1TeflAtt
import patchmem.tefl_att as ta


def stub(variant='qk', n=5, din=4, C=2, hmax=3, P=4, r=2):
    s = SimpleNamespace(variant=variant, n_demo=n, r=r, mem_audit=None,
                        Wq=nn.Linear(din, r, bias=False), Wk=nn.Linear(din, r, bias=False),
                        pos=nn.Parameter(torch.zeros(n)), logt=nn.Parameter(torch.zeros(())), eta=nn.Parameter(torch.zeros(hmax)))
    g = torch.Generator().manual_seed(0)
    b = 3
    s.mem = dict(v=torch.randn(b, n, C, hmax * P, generator=g), eps=torch.randn(b, n, C, din, generator=g),
                 mask=torch.ones(b, n, dtype=torch.bool))
    delta, eps = torch.randn(b, C, hmax, P, generator=g), torch.randn(b, C, din, generator=g)
    stdev = torch.rand(b, 1, C, generator=g) + 0.5
    return s, delta, eps, stdev


def test_identity_at_init_and_off():
    s, d, e, sd = stub()
    assert torch.equal(R1TeflAtt.extra_delta(s, d, e, sd), d)                           # eta = 0
    s.eta.data.fill_(1.0); s.mem_audit = 'off'
    assert torch.equal(R1TeflAtt.extra_delta(s, d, e, sd), d)
    s.mem_audit = None; s.mem = None
    assert torch.equal(R1TeflAtt.extra_delta(s, d, e, sd), d)


def test_pos_window_selects_and_rescales():
    s, d, e, sd = stub(variant='pos')
    s.eta.data.fill_(1.0); s.pos.data[2] = 50.0                                         # all weight on demo 2
    out = R1TeflAtt.extra_delta(s, d, e, sd)
    v2 = s.mem['v'][:, 2] / sd.permute(0, 2, 1)                                         # (b, C, H) normalised by the window scale
    assert torch.allclose(out - d, v2.reshape(d.shape), atol=1e-5)


def test_mask_and_empty_memory():
    s, d, e, sd = stub(variant='pos')
    s.eta.data.fill_(1.0)
    s.mem['mask'][:] = False; s.mem['mask'][0, 4] = True                               # sample 0: only demo 4; others: none
    out = R1TeflAtt.extra_delta(s, d, e, sd)
    assert torch.allclose((out - d)[0], (s.mem['v'][0, 4] / sd[0].T).reshape(d.shape[1:]), atol=1e-5)
    assert torch.equal(out[1:], d[1:])
    out.sum().backward()                                                                # finite gradients with fully masked rows
    assert all(torch.isfinite(p.grad).all() for p in (s.pos, s.eta) if p.grad is not None)


def test_qk_content_selection():
    s, d, e, sd = stub(variant='qk', n=4, din=4, r=4)
    with torch.no_grad():
        s.Wq.weight.copy_(torch.eye(4) * 10); s.Wk.weight.copy_(torch.eye(4) * 10)
    s.eta.data.fill_(1.0)
    s.mem['eps'][:, 1] = e                                                              # demo 1 matches the query exactly
    s.mem['eps'][:, [0, 2, 3]] = -e[:, None]
    out = R1TeflAtt.extra_delta(s, d, e, sd)
    v1 = (s.mem['v'][:, 1] / sd.permute(0, 2, 1)).reshape(d.shape)
    assert torch.allclose(out - d, v1, atol=1e-3)


def test_bank_no_leak_and_mask():
    bk = ta.Bank(np.zeros((ta.LB + 20 * ta.H, 2), dtype=np.float32))
    bk.v = torch.arange(bk.n, dtype=torch.float32)[:, None, None].expand(bk.n, ta.H, 2).clone()
    bk.eps = torch.zeros(bk.n, 2, 3, dtype=torch.float16)
    ts = [ta.LB + ta.H + 5, ta.LB + 10 * ta.H]
    M = bk.mem(ts)
    s = np.asarray(ts)[:, None] - ta.DS[None]
    assert torch.equal(M['mask'].cpu(), torch.as_tensor(s >= ta.LB))
    ok = M['mask'].cpu()
    assert torch.equal(M['v'][..., 0, 0].cpu()[ok], torch.as_tensor(s - ta.LB, dtype=torch.float32)[ok])   # value of origin s
    assert (s + ta.H <= np.asarray(ts)[:, None]).all()


if __name__ == '__main__':
    test_identity_at_init_and_off(); test_pos_window_selects_and_rescales(); test_mask_and_empty_memory(); test_qk_content_selection()
    test_bank_no_leak_and_mask(); print('ok')
