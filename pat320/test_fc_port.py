"""PAT-320 check: pat320/fc.py with D = 24 reproduces MambaTS-pytorch/patchmem/eval_freshcast.py exactly (descriptors,
candidates, readouts, scorer output, untrained eps) on an hourly dataset (ETTh1), H = 720."""
import os, sys
from pathlib import Path
os.environ['ITR_H'] = '720'; os.environ['ITR_L'] = '672'
import torch
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'MambaTS-pytorch'))
import patchmem.eval_freshcast as ref
sys.path.insert(0, str(HERE))
from pat320 import fc
from pat320.carry import load_full
dev = torch.device('cuda')
x, b = load_full('ETTh1'); X = torch.tensor(x, device=dev)
cx = fc.Ctx(X, b[0], 24)
ws = ref.win_stats(X); sdesc = ref.series_desc(X[:b[0]]); order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
assert torch.equal(order, cx.order) and torch.allclose(sdesc, cx.sdesc, equal_nan=True)
g = torch.Generator().manual_seed(1)
t = torch.randint(ref.K_STEPS + ref.LQ, len(X) - 720 + 1, (64,), generator=g).to(dev); c = torch.randint(0, 7, (64,), generator=g).to(dev)
A, B = ref.build(X, ws, sdesc, order, t, c), fc.build(cx, t, c)
for k in A:
    print(k, 'equal' if torch.equal(A[k], B[k]) else f'maxdiff {(A[k].float() - B[k].float()).abs().max().item()}')
RA = ref.readout(X, ws, t, c, A['s'], A['p'], A['val']); RB = fc.readout(cx, t, c, B['s'], B['p'], B['val'])
print('readout maxdiff', (RA - RB).abs().max().item())
torch.manual_seed(0); m = ref.RKR().to(dev); mb = fc.RKR().to(dev); mb.load_state_dict(m.state_dict())
print('scorer maxdiff', (m(A['z'], A['u'], A['valid']) - mb(B['z'], B['u'], B['valid'])).abs().max().item())
vo = torch.arange(b[0], b[0] + 40, 10, device=dev)
e1, _ = ref.untrained_fit(X, ws, sdesc, order, vo); e2 = fc.untrained_eps(cx, vo)
print('eps maxdiff', (e1 - e2).abs().max().item())
