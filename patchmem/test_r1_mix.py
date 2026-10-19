"""PAT-304 per-sample lookback / horizon (R1_MIX) checks for R1, R1Mem (Q8), R1Obs (O), R1Tefl (native, faithful):
(1) lens=None is the original code path: R1 and R1Obs outputs equal the pre-change modules (commit 6c6c724, read with git show) bit for bit;
(2) a right-aligned padded batch with lens gives, for every sample, the same forecast / valid history predictions as the sample's
unpadded window run alone (memory parameters randomised so every memory path is active), and the padded boundary states are exactly 0;
(3) gradients: the padded-batch loss (sum of per-sample losses) gives the same parameter gradients as the per-sample runs summed;
(4) the masked future loss and VPT cost equal the per-sample definitions."""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_mem import R1Mem
from patchmem.r1_obsmem import R1Obs
from patchmem.r1_tefl import R1Tefl
from patchmem.exp_r1 import Exp_R1


def load_bak(name, rev='6c6c724'):
    src = subprocess.run(['git', '-C', str(ROOT), 'show', f'{rev}:patchmem/{name}.py'], capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False); f.write(src); f.close()
    spec = importlib.util.spec_from_file_location(f'{name}_{rev}', f.name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
torch.manual_seed(0)
c = a.enc_in
P = a.patch_len
lens = torch.tensor([20 * P, 8 * P, 15 * P, 11 * P])
Lb, b, n_out = int(lens.max()), 4, 4
x = (torch.randn(b, Lb, c).cumsum(1) * 0.1 + torch.randn(b, 1, c) * 3).to(dev)
perm = torch.stack([torch.randperm(c) for _ in range(b)]).to(dev)
rel = lambda u, v: ((u - v).abs().max() / v.abs().max().clamp_min(1e-12)).item()

# (1) lens=None == pre-change code
old_r1 = load_bak('r1_model').R1
torch.manual_seed(1); m_new = R1(a).to(dev).eval()
torch.manual_seed(1); m_old = old_r1(a).to(dev).eval()
m_old.load_state_dict(m_new.state_dict())
with torch.no_grad():
    o1, o0 = m_new(x, n_out, perm=perm), m_old(x, n_out, perm=perm)
assert torch.equal(o1['pred'], o0['pred']) and torch.equal(o1['hist'], o0['hist']); print('[R1] lens=None == old code: exact')
old_obs = load_bak('r1_obsmem').R1Obs
torch.manual_seed(1); ob_new = R1Obs(a, qk_dim=8).to(dev).eval()
torch.manual_seed(1); ob_old = old_obs(a, qk_dim=8).to(dev).eval()
with torch.no_grad():
    ob_new.obs_net.W.normal_(0, 0.1); ob_new.obs_net.Wg.weight.normal_(0, 0.01)
ob_old.load_state_dict(ob_new.state_dict())
with torch.no_grad():
    assert torch.equal(ob_new(x, n_out, perm=perm)['pred'], ob_old(x, n_out, perm=perm)['pred'])
print('[R1Obs] lens=None == old code: exact')


def randomise(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith(('corr_net', 'mem_net', 'obs_net', 'W1', 'W2')):
                p.add_(torch.randn_like(p) * 0.05)


models = {
    'R1': lambda: R1(a),
    'R1Mem Q8': lambda: R1Mem(a, slot_keep=0.5, qk_dim=8),
    'R1Obs O': lambda: R1Obs(a, qk_dim=8, slot_keep=0.5),
    'R1Tefl native': lambda: R1Tefl(a, mode='native', hc=2),
    'R1Tefl faithful': lambda: R1Tefl(a, mode='faithful', hc=2),
}
for name, mk in models.items():
    torch.manual_seed(2); m = mk().to(dev).eval(); randomise(m)
    # (2) forward equivalence + zero padded states
    tr = {}
    with torch.no_grad():
        o = m(x, n_out, perm=perm, trace=tr, lens=lens)
        T = Lb // P
        for i in range(b):
            Li = int(lens[i]); t0 = T - Li // P
            oi = m(x[i:i + 1, -Li:], n_out, perm=perm[i:i + 1])
            ef, eh = rel(o['pred'][i:i + 1], oi['pred']), rel(o['hist'][i:i + 1, t0:], oi['hist'])
            assert ef < 1e-4 and eh < 1e-4, (name, i, ef, eh)
            if t0:
                assert tr['bstates'][i, :t0].abs().max().item() == 0.0, (name, i)
        print(f'[{name}] padded batch == per-sample runs (forecast + valid history), padded states 0: max rel err ok')
    # (3) gradients
    params = [p for p in m.parameters() if p.requires_grad]
    y = torch.randn(b, n_out * P, c, device=dev)
    m.zero_grad(); (m(x, n_out, perm=perm, lens=lens)['pred'] - y).square().mean((1, 2)).sum().backward()
    gb = [None if p.grad is None else p.grad.clone() for p in params]
    m.zero_grad()
    for i in range(b):
        Li = int(lens[i])
        (m(x[i:i + 1, -Li:], n_out, perm=perm[i:i + 1])['pred'] - y[i:i + 1]).square().mean().backward()
    worst = 0.0
    for p, g in zip(params, gb):
        if g is None and p.grad is None:
            continue
        assert g is not None and p.grad is not None
        if p.grad.abs().max() > 1e-8:
            worst = max(worst, rel(g, p.grad))
    print(f'[{name}] padded-batch grads vs summed per-sample grads: worst rel err {worst:.2e}'); assert worst < 1e-3, worst

# (4) loss and VPT cost
el = torch.rand(3, 10, c)
hv = torch.arange(10)[None, :] < torch.tensor([10, 4, 7])[:, None]
ref = torch.stack([el[i, :n].mean() for i, n in enumerate([10, 4, 7])]).mean()
assert abs(Exp_R1._future_loss(el, hv).item() - ref.item()) < 1e-6
torch.manual_seed(3); m = R1(a).train()
m.scan_order(3, 'cpu')
before = m.enc.adjacency_matrix.clone()
m.update_vpt_cost(el, hv)
cost = torch.stack([el[i, :n].mean() for i, n in enumerate([10, 4, 7])])
cost = cost / cost.std()
tt = m.enc.transition_tuple
exp_adj = before.clone()
for i in range(3):
    for k in range(c):
        exp_adj[tt[i, k, 0], tt[i, k, 1]] += cost[i]
assert torch.allclose(m.enc.adjacency_matrix, exp_adj, atol=1e-5); print('[loss / VPT] masked future loss and per-sample VPT cost: ok')
print('OK')
