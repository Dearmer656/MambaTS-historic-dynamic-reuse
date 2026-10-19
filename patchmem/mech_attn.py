"""PAT-304: where does the Q8+drop50 rollout retrieval look, per future patch k? Eval-only on a trained R1Mem checkpoint, test split.
Every rollout retrieval call records, per window and head, the attention a over the M history slots. Slot position is reported as the
LAG from the end of the lookback (lag 0 = slot T-1, the most recent one), so the curves of different k are comparable (the age seen by
patch k is lag + k + 1). Per (L, H) and k:
  mass[k][lag]   window- and head-mean attention on the slot at that lag (the uniform reference is 1/M)
  entropy        per-window, per-head normalised entropy H(a)/log M (1 = uniform), its mean and quantiles
  eff_slots      exp(H(a)), effective number of slots read
  top1_lag       histogram of the per-window, per-head argmax lag
  gate           mean gate g
A smooth window-mean mass profile can come from sharp but varied per-window attention, so read it together with entropy and top1_lag.
usage (repo root): python patchmem/mech_attn.py --data E|W|T --seed 3047 [--grid 1440:336,720:336] [--stride 1]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA, Q8ENV, load


def hook(m, rec):
    orig = m._retrieve

    def f(h_minus, keys, vals, times=None, t_now=None, phase='hist', masks=None):
        if phase == 'roll' and rec.get('on'):
            _, a, g = m.mem_net(h_minus, torch.stack(keys, 1), torch.stack(vals, 1), None)   # (b, nh, M)
            M = a.shape[-1]
            k = t_now - times[-1] - 1
            lag = (times[-1] - torch.as_tensor(times)).numpy()                              # (M,)
            r = rec['k'].setdefault(k, dict(mass=np.zeros(M), n=0, ent=[], top1=np.zeros(M), gate=0.0))
            r['mass'][lag] += a.mean(1).sum(0).double().cpu().numpy()
            ent = (-(a * a.clamp_min(1e-12).log()).sum(-1) / np.log(M)) if M >= 2 else torch.ones_like(a[..., 0])
            r['ent'].append(ent.flatten().float().cpu())
            r['top1'] += np.bincount(lag[a.argmax(-1).flatten().cpu().numpy()], minlength=M)[:M]
            r['gate'] += g.mean().item() * a.shape[0]
            r['n'] += a.shape[0]
        return orig(h_minus, keys, vals, times, t_now, phase, masks)
    m._retrieve = f


@torch.no_grad()
def cell(e, rec, L, H, stride):
    m = e.model
    ds = e._ds('test', L, H)
    ids = list(range(0, len(ds), stride))
    rec.update(on=True, k={})
    for s0 in range(0, len(ids), 64):
        bx, _ = e._batch(ds, ids[s0:s0 + 64])
        m(bx.to(e.device), H // m.P)
    rec['on'] = False
    out = {}
    for k, r in sorted(rec['k'].items()):
        ent = torch.cat(r['ent'])
        M = len(r['mass'])
        out[k] = dict(M=M, mass=(r['mass'] / r['n']).tolist(), top1_frac=(r['top1'] / r['top1'].sum()).tolist(),
                      entropy_mean=ent.mean().item(), entropy_q=[ent.quantile(q).item() for q in (0.1, 0.5, 0.9)],
                      eff_slots_mean=(ent * np.log(M)).exp().mean().item(), gate=r['gate'] / r['n'], windows=r['n'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=list(DATA)); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default='1440:336,720:336'); ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    base, ck, _, dQ = DATA[a.data]
    e = load(base, ck, dQ.format(a.seed), a.seed, Q8ENV)
    rec = {}
    hook(e.model, rec)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        res[f'{L}/{H}'] = r = cell(e, rec, L, H, a.stride)
        for k, v in r.items():
            print(f'[attn {a.data} s{a.seed}] L={L} H={H} k={k + 1} M={v["M"]} | entropy {v["entropy_mean"]:.3f} '
                  f'(q10/50/90 {"/".join(f"{x:.2f}" for x in v["entropy_q"])}) eff slots {v["eff_slots_mean"]:.1f} gate {v["gate"]:.3f} | '
                  f'mass lag0..4 {" ".join(f"{x:.3f}" for x in v["mass"][:5])} | top1 lag0..4 {" ".join(f"{x:.2f}" for x in v["top1_frac"][:5])}',
                  flush=True)
    os.makedirs('patchmem/logs/mech', exist_ok=True)
    json.dump(dict(data=a.data, seed=a.seed, stride=a.stride, grid=res), open(f'patchmem/logs/mech/attn_{a.data}_s{a.seed}.json', 'w'))


if __name__ == '__main__':
    main()
