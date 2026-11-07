"""PAT-317 checks for TimeAttention pos='path' (run: python -m pat317.test_path from the OpenLTM root).
1. beta = 0 -> 'path' == 'nope' exactly (same q/k/v).
2. C = 1: 'path' output == fla naive_path_attn (official PaTH reference, g = 0).
3. C = 3: path_scores == explicit Householder products, H_i ... H_{j+1} of the key variable, for every (query, key) pair.
4. 'rope' unchanged vs the untouched official formula; gradients reach w_proj / beta_proj.
"""
import sys

import torch

from layers.SelfAttention_Family import TimeAttention, path_scores

sys.path.insert(0, '../fla_clean_ref')
from fla.ops.path_attn.naive import naive_path_attn  # noqa: E402

torch.manual_seed(0)
dt = torch.float64
B, Hh, E = 2, 2, 8
D = Hh * E


def make(pos, C, N):
    a = TimeAttention(True, attention_dropout=0.0, d_model=D, num_heads=Hh, pos=pos).to(dt).eval()
    q, k, v = (torch.randn(B, C * N, Hh, E, dtype=dt) for _ in range(3))
    x = torch.randn(B, C * N, D, dtype=dt)
    return a, q, k, v, x


# 1
C, N = 3, 5
a, q, k, v, x = make('path', C, N)
torch.nn.init.zeros_(a.beta_proj.weight)
torch.nn.init.constant_(a.beta_proj.bias, -1e4)                                # 2 sigmoid -> 0
b = TimeAttention(True, attention_dropout=0.0, d_model=D, num_heads=Hh, pos='nope').to(dt).eval()
b.attn_bias.load_state_dict(a.attn_bias.state_dict())
o1, o2 = a(q, k, v, None, C, N, x=x)[0], b(q, k, v, None, C, N)[0]
print('1 beta=0 path vs nope max|diff|', float((o1 - o2).abs().max()))
assert (o1 - o2).abs().max() < 1e-12

# 2
C, N = 1, 12
a, q, k, v, x = make('path', C, N)
o1 = a(q, k, v, None, C, N, x=x)[0]
w = torch.nn.functional.normalize(a.w_proj(x).view(B, N, Hh, E), dim=-1)
beta = 2 * torch.sigmoid(a.beta_proj(x))
o2 = naive_path_attn(q, k, v, w, beta, torch.zeros(B, N, Hh, dtype=dt), scale=E ** -0.5, chunk_size=4).to(dt)
print('2 C=1 path vs fla naive max|diff|', float((o1 - o2).abs().max()))
assert (o1 - o2).abs().max() < 1e-5

# 3
C, N = 3, 6
qq, kk = torch.randn(1, 1, C * N, E, dtype=dt), torch.randn(1, 1, C * N, E, dtype=dt)
ww = torch.nn.functional.normalize(torch.randn(1, 1, C * N, E, dtype=dt), dim=-1)
bb = 2 * torch.rand(1, 1, C * N, dtype=dt)
S = path_scores(qq, kk, ww, bb, C, N)[0, 0]
err = 0.0
for c in range(C):
    for i in range(N):
        for c2 in range(C):
            for j in range(N):
                if j > i:
                    continue
                P = torch.eye(E, dtype=dt)
                for t in range(j + 1, i + 1):                                    # P = H_i ... H_{j+1}
                    u = ww[0, 0, c2 * N + t]
                    P = (torch.eye(E, dtype=dt) - bb[0, 0, c2 * N + t] * torch.outer(u, u)) @ P
                ref = qq[0, 0, c * N + i] @ P @ kk[0, 0, c2 * N + j]
                err = max(err, float((S[c * N + i, c2 * N + j] - ref).abs()))
print('3 C=3 path_scores vs explicit products max|diff|', err)
assert err < 1e-12

# 4
C, N = 2, 4
a, q, k, v, x = make('rope', C, N)
o = a(q, k, v, None, C, N)[0]
qr, kr = (t.permute(0, 2, 1, 3) for t in (q, k))
sid = torch.arange(C * N).expand(B, Hh, -1)
qr, kr = a.qk_proj(qr, kr, query_id=sid, kv_id=sid)
from einops import repeat  # noqa: E402
from utils.masking import TimerMultivariateMask  # noqa: E402
vid = repeat(repeat(torch.arange(C), 'C -> (C n)', n=N), 'L -> b h L', b=B, h=1)
m = a.attn_bias(vid, vid).masked_fill(TimerMultivariateMask(B, C, N).mask, float('-inf'))
ref = torch.einsum('bhls,bshd->blhd', torch.softmax(E ** -0.5 * (torch.einsum('bhle,bhse->bhls', qr, kr) + m), -1), v)
print('4 rope vs official formula max|diff|', float((o - ref).abs().max()))
assert (o - ref).abs().max() < 1e-12
a, q, k, v, x = make('rope_path', C, N)
a.train()
a(q, k, v, None, C, N, x=x)[0].square().sum().backward()
gw, gb = a.w_proj.weight.grad.abs().sum(), a.beta_proj.weight.grad.abs().sum()
print('4 rope_path grads w_proj', float(gw), 'beta_proj', float(gb))
assert gw > 0 and gb > 0
print('all ok')
