"""PAT-304 Gate 0: explicit recurrent selective scan (Mamba-1, conv-off) with per-patch-boundary state capture.
Same math as mamba_ssm selective_scan_ref (delta_softplus=True); the loop is split into chunks so each chunk can be checkpointed
and the SSM state is captured / injectable at chunk boundaries. Layout follows selective_scan_fn: u,delta,z (b,d,l); A (d,n); B,C (b,n,l)."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk(h, u, dt, Bm, Cm, A, D, z):
    dA = torch.exp(dt.unsqueeze(-1) * A[None, :, None, :])                 # (b,d,l,n)
    dBu = (dt * u).unsqueeze(-1) * Bm.transpose(1, 2).unsqueeze(1)         # (b,d,l,n)
    ys = []
    for t in range(u.shape[-1]):
        h = dA[:, :, t] * h + dBu[:, :, t]
        ys.append(torch.einsum('bdn,bn->bd', h, Cm[:, :, t]))
    y = torch.stack(ys, -1) + u * D[None, :, None]
    return h, y * F.silu(z)


def ref_scan(u, delta, A, B, C, D, z, delta_bias, boundaries=(), h0=None, ckpt=False, hook=None, return_read=False):
    """boundaries: increasing token counts (each in 1..l); the state after token boundaries[i] is captured.
    h0: (b,d,n) initial state or None. hook(i, h_minus) -> None | (d_read, d_post), each (b,d,n) or None, is called at every boundary (including the last token):
      d_read: read-time correction, h_read = h_minus + d_read is what a readout of patch i should see (acts on the current prediction only);
      d_post: post-label correction, the scan continues from h_minus + d_post (ignored at the last token, nothing follows).
    Returns y (b,d,l), last_state (b,d,n), states = prior h^- at each boundary (b,len(boundaries),d,n); with return_read also reads = h_read (same shape)."""
    b, d, l = u.shape
    dt = F.softplus(delta + delta_bias[None, :, None])
    h = torch.zeros(b, d, A.shape[1], dtype=u.dtype, device=u.device) if h0 is None else h0
    cuts = [0] + list(boundaries) + ([l] if not boundaries or boundaries[-1] != l else [])
    assert all(a < c for a, c in zip(cuts, cuts[1:])) and cuts[-1] == l
    ys, caps, reads = [], [], []
    for s, e in zip(cuts, cuts[1:]):
        args = (h, u[..., s:e], dt[..., s:e], B[..., s:e], C[..., s:e], A, D, z[..., s:e])
        h, y = checkpoint(_chunk, *args, use_reentrant=False) if ckpt else _chunk(*args)
        ys.append(y)
        if e in boundaries:
            caps.append(h)
            d_read, d_post = (hook(len(caps) - 1, h) if hook is not None else None) or (None, None)
            reads.append(h if d_read is None else h + d_read)
            if d_post is not None and e != l:
                h = h + d_post
    states = torch.stack(caps, 1) if caps else h.new_zeros(b, 0, d, A.shape[1])
    if return_read:
        return torch.cat(ys, -1), h, states, (torch.stack(reads, 1) if reads else states)
    return torch.cat(ys, -1), h, states


def _split_proj(m, x):
    x_dbl = m.x_proj(x)
    return torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)


def mamba_proj(m, hidden_states):
    """Conv-off Mamba input projections. hidden_states (b,L,dm) -> x (b,d,L), dt pre-softplus without bias (b,d,L), A (d,n), B,C (b,n,L), D, z, dt bias."""
    assert not m.use_casual_conv and not m.use_fast_path
    b, L, _ = hidden_states.shape
    x, z = m.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)        # (b,d,L); conv-off forward applies no activation to x
    dt, B, C = _split_proj(m, x.transpose(1, 2).reshape(b * L, -1))
    dt = (m.dt_proj.weight @ dt.t()).reshape(-1, b, L).permute(1, 0, 2)
    B = B.reshape(b, L, -1).transpose(1, 2).contiguous(); C = C.reshape(b, L, -1).transpose(1, 2).contiguous()
    return x, dt, -torch.exp(m.A_log.float()), B, C, m.D.float(), z, m.dt_proj.bias.float()


def mamba_forward_capture(m, hidden_states, boundaries=(), h0=None, ckpt=False, hook=None, return_read=False):
    """Conv-off, non-fast-path Mamba.forward (layers/mamba_ssm/mamba_simple.py) with ref_scan. hidden_states (b,L,dm).
    Returns out (b,L,dm), last_state, boundary prior states h^- (b,nb,d_inner,d_state) (+ read states h_read if return_read)."""
    x, dt, A, B, C, D, z, bias = mamba_proj(m, hidden_states)
    y, last, *rest = ref_scan(x, dt, A, B, C, D, z, bias, boundaries, h0, ckpt, hook, return_read)
    return (m.out_proj(y.transpose(1, 2)), last, *rest)


def mamba_step_noconv(m, hidden, state):
    """One-token step consistent with the conv-off forward. hidden (b,1,dm), state (b,d,n); returns out (b,1,dm), new state (not in place)."""
    assert not m.use_casual_conv
    x, z = m.in_proj(hidden.squeeze(1)).chunk(2, dim=-1)
    dt, B, C = _split_proj(m, x)
    dt = F.softplus(F.linear(dt, m.dt_proj.weight) + m.dt_proj.bias.to(dt.dtype))
    A = -torch.exp(m.A_log.float())
    state = state * torch.exp(torch.einsum('bd,dn->bdn', dt, A)) + x.unsqueeze(-1) * torch.einsum('bd,bn->bdn', dt, B)
    y = (torch.einsum('bdn,bn->bd', state, C) + m.D.to(x.dtype) * x) * F.silu(z)
    return m.out_proj(y).unsqueeze(1), state
