"""Unit tests for patchmem/icl_att.py: attention indexing (keys e_i -> values e_{i+k}, observed queries only), null keeps values,
fusion SSE identity, end-to-end on a lag-2-periodic sequence."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import patchmem.icl_lag as IL
IL.dev = 'cpu'
import patchmem.icl_att as A
A.dev = 'cpu'
torch.manual_seed(0); dt = torch.float64
B, C, J, P = 2, 3, 10, IL.P
E = torch.randn(B, C, J, P, dtype=dt)
R = A.reps(E)
# with a near-zero temperature and the query equal to an existing key, attention returns that key's successor
for k in (1, 2):
    qpos = 4
    out = A.attend(R['raw'], E, k, qpos, 'raw', 'rbf', 1e-3)
    assert torch.allclose(out, E[:, :, qpos + k]), 'lag-k value of the matching key'
# keys only from e_0..e_{J-1-k}: the query e_{J-1} (lead 1, lag 1) never matches itself
out = A.attend(R['raw'], E, 1, J - 1, 'raw', 'rbf', 1e-3)
assert not torch.allclose(out, E[:, :, J - 1])
# null: permuting keys leaves the multiset of values and only re-pairs them
keys = torch.rand(B, C, J, dtype=dt); pm = A.perm_for(keys, J - 1)
assert sorted(pm[0, 0, :J - 1].tolist()) == list(range(J - 1)) and pm[0, 0, J - 1].item() == J - 1
o0 = A.attend(R['phi'], E, 1, J - 1, 'phi', 'dot', 1e9)                                  # tau -> inf: uniform mean of values
o1 = A.attend(R['phi'], E, 1, J - 1, 'phi', 'dot', 1e9, pm)
assert torch.allclose(o0, E[:, :, 1:].mean(2)) and torch.allclose(o1, o0)
# end to end: shapes A A B B A A ... -> lag-2 map deterministic (A->B, B->A), lag-1 ambiguous (A->A or B)
base = torch.randn(2, P, C, dtype=dt)
seq = torch.stack([base[(t // 2) % 2] * (1 + 0.01 * t) for t in range(30 + 7)], 0)[None].repeat(4, 1, 1, 1)
Hs = seq[:, :29].flip(1).numpy(); F = seq[:, 29:].reshape(4, -1, C).numpy()
S, Bs, ss = A.score(F, Hs, torch.Generator().manual_seed(0))
best = lambda nm: max((1 - v / ss[0], c) for (n, c), v in S[0].items() if n == nm)
print('att1', best('att1')[0], 'att2', best('att2')[0], 'fuse', best('att_fuse')[0], 'rdg2', best('rdg2')[0])
assert best('att2')[0] > 0.99 and best('att_fuse')[0] >= best('att2')[0] - 1e-9 and best('att1')[0] < 0.5
# fusion identity: beta = 0 equals att1, beta = 1 equals att2 for the same configs
c = 'raw_rbf_t0.25'
assert abs(S[0][('att_fuse', f'{c}|{c}|b0.0')] - S[0][('att1', c)]) < 1e-6 * ss[0]
assert abs(S[0][('att_fuse', f'{c}|{c}|b1.0')] - S[0][('att2', c)]) < 1e-6 * ss[0]
print('ALL PASS')
