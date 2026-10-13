"""PAT-304 Gate 0 (part 1): ref_scan / mamba_forward_capture / mamba_step_noconv against the CUDA kernel.
usage (GPU node): python patchmem/test_ref_scan.py"""
import sys
from pathlib import Path
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from layers.mamba_ssm.mamba_simple import Mamba
from patchmem.ref_scan import ref_scan, mamba_forward_capture, mamba_step_noconv

dev = 'cuda'
torch.manual_seed(0)


def mk(b=3, d=24, n=8, l=40, dtype=torch.float32):
    g = lambda *s: torch.randn(*s, device=dev, dtype=dtype)
    u, delta, z = g(b, d, l), g(b, d, l), g(b, d, l)
    A = -torch.exp(g(d, n) * 0.3); B, C = g(b, n, l), g(b, n, l); D, db = g(d), g(d)
    return [t.requires_grad_() for t in (u, delta, A, B, C, D, z, db)]


def close(a, b, name, rtol=2e-3, atol=2e-4):
    err = (a - b).abs().max().item(); sc = b.abs().max().item() + 1e-9
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f'  {"ok  " if ok else "FAIL"} {name}: max abs err {err:.2e} (scale {sc:.2e})')
    assert ok, name


def grads(f, ins, w):
    for t in ins: t.grad = None
    out = f(*ins)
    sum((o * wi).sum() for o, wi in zip(out, w)).backward()
    return [t.grad.clone() for t in ins]


def test_vs_kernels():
    print('T1/T2 ref_scan vs selective_scan_ref and CUDA selective_scan_fn (y, last_state, all grads)')
    ins = mk(); u, delta, A, B, C, D, z, db = ins; bounds = (8, 16, 24, 40)
    wy = torch.randn_like(u); wh = torch.randn(u.shape[0], u.shape[1], A.shape[1], device=dev)
    f_ref = lambda *a: ref_scan(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], bounds)[:2]
    f_py = lambda *a: selective_scan_ref(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], True, True)
    f_cu = lambda *a: selective_scan_fn(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], True, True)
    outs = {k: f(*ins) for k, f in (('ref', f_ref), ('py', f_py), ('cuda', f_cu))}
    names = 'u delta A B C D z delta_bias'.split()
    for other in ('py', 'cuda'):
        close(outs['ref'][0], outs[other][0], f'y vs {other}'); close(outs['ref'][1], outs[other][1], f'last_state vs {other}')
    g_ref, g_py = grads(f_ref, ins, (wy, wh)), grads(f_py, ins, (wy, wh))
    for nm, a, b in zip(names, g_ref, g_py): close(a, b, f'grad {nm} (y + last_state loss) vs py', rtol=5e-3, atol=5e-4)
    wz = torch.zeros_like(wh)
    g_ref_y, g_cu_y = grads(f_ref, ins, (wy, wz)), grads(f_cu, ins, (wy, wz))
    for nm, a, b in zip(names, g_ref_y, g_cu_y): close(a, b, f'grad {nm} (y loss only) vs cuda', rtol=5e-3, atol=5e-4)
    g_cu_h = grads(f_cu, ins, (torch.zeros_like(wy), wh))
    assert all(g.abs().max().item() == 0 for g in g_cu_h), 'expected CUDA kernel to drop last_state gradients'
    print('  ok   CUDA selective_scan_fn drops the gradient through return_last_state (all grads exactly 0); ref_scan propagates it')


def test_boundaries_and_chunking():
    print('T3 captured boundary state == last_state of the CUDA kernel on the prefix;  T5 chunked (h0) == full scan')
    ins = mk(l=48); u, delta, A, B, C, D, z, db = ins; bounds = (12, 24, 36, 48)
    with torch.no_grad():
        y, last, st = ref_scan(*ins, bounds)
        for i, k in enumerate(bounds):
            _, hk = selective_scan_fn(u[..., :k], delta[..., :k], A, B[..., :k], C[..., :k], D, z[..., :k], db, True, True)
            close(st[:, i], hk, f'state after token {k}')
        close(st[:, -1], last, 'last boundary == last_state')
        y1, h1, _ = ref_scan(u[..., :24], delta[..., :24], A, B[..., :24], C[..., :24], D, z[..., :24], db)
        y2, h2, _ = ref_scan(u[..., 24:], delta[..., 24:], A, B[..., 24:], C[..., 24:], D, z[..., 24:], db, h0=h1)
        close(torch.cat([y1, y2], -1), y, 'chunk(h0)-continued y == full y'); close(h2, last, 'chunk(h0)-continued last state')


def test_causality():
    print('T4 causality: perturbing tokens >= k leaves y[:k] and states at boundaries <= k unchanged')
    ins = mk(l=48); u, delta, A, B, C, D, z, db = ins; bounds = (12, 24, 36, 48); k = 24
    with torch.no_grad():
        y, _, st = ref_scan(*ins, bounds)
        u2, d2, B2, C2, z2 = u.clone(), delta.clone(), B.clone(), C.clone(), z.clone()
        for t in (u2, d2, B2, C2, z2): t[..., k:] += torch.randn_like(t[..., k:])
        y2, _, st2 = ref_scan(u2, d2, A, B2, C2, D, z2, db, bounds)
        close(y2[..., :k], y[..., :k], 'y[:k]'); close(st2[:, :2], st[:, :2], 'states at 12, 24')
        assert (st2[:, 2] - st[:, 2]).abs().max() > 1e-3, 'perturbation should change later states'
        print('  ok   later states do change')


def test_module():
    print('T6 Mamba module (conv-off, official non-fast path): forward / gradients / checkpoint / step consistency')
    for dropout in (0.0,):
        m = Mamba(d_model=16, d_state=8, d_conv=2, expand=2, use_casual_conv=False, use_fast_path=False, layer_idx=0, dropout=dropout, n_vars=1).to(dev).eval()
        b, L = 3, 36; bounds = (9, 18, 27, 36)
        x = torch.randn(b, L, 16, device=dev, requires_grad=True); w = torch.randn(b, L, 16, device=dev)
        def run(f):
            m.zero_grad(); x.grad = None
            out = f(); (out * w).sum().backward()
            return out.detach(), x.grad.clone(), {n: p.grad.clone() for n, p in m.named_parameters()}
        o0, gx0, gp0 = run(lambda: m(x))
        o1, gx1, gp1 = run(lambda: mamba_forward_capture(m, x, bounds)[0])
        o2, gx2, gp2 = run(lambda: mamba_forward_capture(m, x, bounds, ckpt=True)[0])
        close(o1, o0, 'forward vs official'); close(gx1, gx0, 'grad x vs official', rtol=5e-3, atol=5e-4)
        for n in gp0: close(gp1[n], gp0[n], f'grad {n} vs official', rtol=5e-3, atol=5e-4)
        close(o2, o1, 'ckpt forward == no ckpt', rtol=1e-5, atol=1e-6); close(gx2, gx1, 'ckpt grad x == no ckpt', rtol=1e-4, atol=1e-5)
        for n in gp1: close(gp2[n], gp1[n], f'ckpt grad {n}', rtol=1e-4, atol=1e-5)
        with torch.no_grad():
            _, last, st = mamba_forward_capture(m, x, bounds)
            h = torch.zeros(b, m.d_inner, m.d_state, device=dev); outs = []
            for t in range(L):
                o, h = mamba_step_noconv(m, x[:, t:t + 1], h); outs.append(o)
                if t + 1 in bounds: close(h, st[:, bounds.index(t + 1)], f'step state after token {t + 1} == captured')
            close(torch.cat(outs, 1), o0, 'step-by-step outputs == official forward')


if __name__ == '__main__':
    test_vs_kernels(); test_boundaries_and_chunking(); test_causality(); test_module()
    print('ALL PASS')
