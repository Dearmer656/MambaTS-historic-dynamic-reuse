"""PAT-320 check: task embedding is a no-op at init, receives gradients, survives save -> Frozen reload, and is per-sample."""
import sys, tempfile, os; from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pat320.joint import build_model, set_task, L, TASK
from pat320.carry import Frozen
dev = torch.device('cuda'); torch.manual_seed(0)
m0 = build_model(dev).eval(); m1 = build_model(dev, True).eval()
m1.load_state_dict({**m0.state_dict(), 'task_emb.weight': m1.task_emb.weight}); x = torch.randn(4, L, 1, device=dev)
set_task(m1, 'ECL'); print('init no-op', (m0(x, None, None) - m1(x, None, None)).abs().max().item())
m1.train(); m1.task = torch.tensor([0, 1, 5, 6], device=dev); m1(x, None, None).square().mean().backward()
g = m1.task_emb.weight.grad; print('grad rows nonzero', (g.abs().sum(1) > 0).nonzero().view(-1).tolist(), '(expect [0, 1, 5, 6])')
with torch.no_grad(): m1.task_emb.weight.normal_()
m1.eval(); f = os.path.join(tempfile.mkdtemp(), 'te.pth'); torch.save(m1.state_dict(), f)
bb = Frozen(f, dev); print('Frozen task_emb', bb.task_emb)
for d in ('ETTh1', 'Traffic'):
    bb.set(d); set_task(m1, d); print(d, 'reload diff', (bb.m(x, None, None) - m1(x, None, None)).abs().max().item())
set_task(m1, 'ETTh1'); a = m1(x, None, None); set_task(m1, 'Traffic'); print('tasks differ', (a - m1(x, None, None)).abs().max().item() > 0)
bb0 = Frozen.__new__(Frozen); print('plain ckpt task_emb', Frozen(f.replace('te.pth', 'p.pth') if torch.save(m0.state_dict(), f.replace('te.pth', 'p.pth')) is None else f, dev).task_emb)
