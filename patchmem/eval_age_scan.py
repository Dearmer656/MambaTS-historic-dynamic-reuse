"""PAT-315 diagnostic: similarity of the current case to demos at every hourly age, not only the same clock.

Query case (origin t, channel c); candidate demo at every age a = H .. history-H-L (s = t - a, value E_s fully
observed by t).  Per (case, age):
  se  cos(E_{t-H}, E_{s-H})          error key
  sy  cos(y[t-96:t], y[s-96:s])      input key
  lg  tau_e se + tau_y sy            retrieval logit (train-selected e+y96 tau of eval_raw_ylen for this L)
  cv  cos(E_s, E_t)                  how similar the demo's value is to the target error (oracle quantity)
Aggregates over a random sample of test cases:
  by age            mean se / sy / lg / cv, and single-demo gain_opt(a) = (sum t.e)^2 / (sum e^2 sum t^2)
                    (the mod-24 / mod-168 folds and the printed age groups AVERAGE gain_opt(a) over ages; this is
                    the mean single-demo value, not the gain of pooling those ages)
  by age mod 24 / mod 168
  14-demo sets      phase (the current same-clock demos) | top-k by logit over all ages | top-k by logit excluding
                    same clock | random | oracle top-k by cv; softmax(lg) weights inside the set (oracle: same);
                    gain_opt of the weighted error, and where the selected ages fall (same clock / same hour)
gain_opt is the best single-scale gain on these cases (information content, not a model result).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, demo_ages, load_split

TAUS = {96: (1.0, 4.0), 720: (1.0, 2.0)}
NK = 14
YL = 96


def cosv(a, b):
    """a: B d, b: B n d -> B n."""
    return F.normalize(b, dim=-1) @ F.normalize(a, dim=-1)[..., None]


@torch.no_grad()
def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    phase = demo_ages('phase', cell['history'], cell['period'], cell['stride'])
    data = load_split(exp, 'test', cell['history'], args.forecast_bs)
    del exp
    torch.cuda.empty_cache()
    E, X = data['E'], data['X']
    C = X.shape[1]
    ages = torch.arange(H, cell['history'] - H - L + 1)                 # all hourly ages
    A = len(ages)
    is_phase = torch.isin(ages, torch.as_tensor(phase)).to(device)
    same_clock = (ages % 168 == 0).to(device)
    same_hour = (ages % 24 == 0).to(device)
    te, ty = TAUS[L]
    g = torch.Generator().manual_seed(seed)
    n = args.cases
    t_all = data['origins'][torch.randint(len(data['origins']), (n,), generator=g)]
    c_all = torch.randint(C, (n,), generator=g)
    k = torch.arange(YL)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
    agg = dict(se=z(A), sy=z(A), lg=z(A), cv=z(A), tv=z(A), vv=z(A))
    sets = ('phase', 'topk', 'topk_noclock', 'random', 'oracle')
    sacc = {s: dict(te=z(()), ee=z(()), clock=z(()), hour=z(()), cv=z(())) for s in sets}
    tt = z(())
    for i in range(0, n, args.bs):
        t, c = t_all[i:i + args.bs], c_all[i:i + args.bs]
        B = len(t)
        s = t[:, None] - ages[None]
        cc = c[:, None]
        T = E[t - L, :, c].to(device).double()
        q, K = E[t - H - L, :, c].to(device), E[s - H - L, :, cc].to(device)
        V = E[s - L, :, cc].to(device).double()
        yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)
        ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)
        se, sy = cosv(q, K)[..., 0].double(), cosv(yq, ys)[..., 0].double()
        lg = te * se + ty * sy
        cv = cosv(T, V)[..., 0]
        tv = (V * T[:, None]).sum(-1)
        vv = V.square().sum(-1)
        for key, val in (('se', se), ('sy', sy), ('lg', lg), ('cv', cv), ('tv', tv), ('vv', vv)):
            agg[key] += val.sum(0)
        tt += T.square().sum()
        masks = {'phase': is_phase.expand(B, -1)}
        top = lg.topk(NK, -1).indices
        masks['topk'] = torch.zeros(B, A, dtype=torch.bool, device=device).scatter_(1, top, True)
        top = lg.masked_fill(same_clock, float('-inf')).topk(NK, -1).indices
        masks['topk_noclock'] = torch.zeros(B, A, dtype=torch.bool, device=device).scatter_(1, top, True)
        rnd = torch.rand(B, A, device=device).topk(NK, -1).indices
        masks['random'] = torch.zeros(B, A, dtype=torch.bool, device=device).scatter_(1, rnd, True)
        top = cv.topk(NK, -1).indices
        masks['oracle'] = torch.zeros(B, A, dtype=torch.bool, device=device).scatter_(1, top, True)
        for name, m in masks.items():
            w = torch.softmax(lg.masked_fill(~m, float('-inf')), -1)
            eh = (w[..., None] * V).sum(1)
            a = sacc[name]
            a['te'] += (eh * T).sum()
            a['ee'] += eh.square().sum()
            a['clock'] += (m & same_clock).sum()
            a['hour'] += (m & same_hour).sum()
            a['cv'] += (cv * m).sum()
    tt = float(tt)
    by_age = {key: (v / n).tolist() for key, v in agg.items() if key in ('se', 'sy', 'lg', 'cv')}
    by_age['gain_opt'] = (100 * agg['tv'] ** 2 / (agg['vv'] * tt)).tolist()
    res_sets = {}
    for name, a in sacc.items():
        res_sets[name] = dict(gain_opt=100 * float(a['te']) ** 2 / (float(a['ee']) * tt),
                              frac_same_clock=float(a['clock']) / (n * NK), frac_same_hour=float(a['hour']) / (n * NK),
                              mean_cv=float(a['cv']) / (n * NK))
        print(f'[set] {name:13s} gain_opt {res_sets[name]["gain_opt"]:.3f}% same_clock '
              f'{res_sets[name]["frac_same_clock"]:.2f} same_hour {res_sets[name]["frac_same_hour"]:.2f} '
              f'mean cos(e_j,target) {res_sets[name]["mean_cv"]:.4f}', flush=True)

    def fold(p):
        out = {}
        for key, vals in by_age.items():
            v = torch.tensor(vals)
            out[key] = [float(v[ages % p == r].mean()) for r in range(p)]
        return out
    for key in ('lg', 'cv', 'gain_opt'):
        v = torch.tensor(by_age[key])
        print(f'[age] {key:8s} same-clock {float(v[ages % 168 == 0].mean()):.4f} | same-hour other-day '
              f'{float(v[(ages % 24 == 0) & (ages % 168 != 0)].mean()):.4f} | other {float(v[ages % 24 != 0].mean()):.4f}',
              flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, taus=(te, ty), cases=n, nk=NK, ages=ages.tolist(), phase=phase,
               checkpoint=str(checkpoint), by_age=by_age, mod24=fold(24), mod168=fold(168), sets=res_sets)
    output = Path(args.out_dir) / f'{code}_H{H}_L{L}_agescan_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--cases', type=int, default=50000)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
