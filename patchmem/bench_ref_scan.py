"""PAT-304 Gate 0: speed / peak memory of ref_scan at ECL scale (one layer, fwd+bwd). usage: python patchmem/bench_ref_scan.py [b] [C] [npatch]"""
import sys, time
from pathlib import Path
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchmem.ref_scan import ref_scan
b, C, npatch = [int(a) for a in sys.argv[1:4]] if len(sys.argv) > 3 else (16, 321, 15)
d, n, l = 256, 16, C * npatch; dev = 'cuda'
g = lambda *s: torch.randn(*s, device=dev).requires_grad_()
u, delta, z = g(b, d, l), g(b, d, l), g(b, d, l); A = (-torch.exp(torch.randn(d, n, device=dev))).requires_grad_()
B, Cm = g(b, n, l), g(b, n, l); D, db = g(d), g(d)
bounds = tuple(C * (i + 1) for i in range(npatch))
def run(name, f):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated(); t = time.time()
    y = f(); y.square().mean().backward(); torch.cuda.synchronize()
    print(f'{name}: fwd+bwd {time.time() - t:.2f}s, peak extra mem {(torch.cuda.max_memory_allocated() - m0) / 2**30:.2f} GiB', flush=True)
print(f'b={b} d_inner={d} d_state={n} tokens={l} ({npatch} patches x {C} vars)')
run('CUDA selective_scan_fn', lambda: selective_scan_fn(u, delta, A, B, Cm, D, z, db, True))
run('ref_scan ckpt=True ', lambda: ref_scan(u, delta, A, B, Cm, D, z, db, bounds, ckpt=True)[0])
run('ref_scan ckpt=True (2nd, warm)', lambda: ref_scan(u, delta, A, B, Cm, D, z, db, bounds, ckpt=True)[0])
try: run('ref_scan ckpt=False', lambda: ref_scan(u, delta, A, B, Cm, D, z, db, bounds, ckpt=False)[0])
except torch.OutOfMemoryError as e: print('ref_scan ckpt=False: OOM')
