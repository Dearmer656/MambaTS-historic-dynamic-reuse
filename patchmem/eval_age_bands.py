"""PAT-315 mechanism: is the retrieved error a fixed state -> error map, or a slowly drifting one?

Training-free cos retrieval (eval_raw_sim) with the demo pool restricted to one age band at a time.
Bands of same-phase demos (multiples of 168 h), 4 demos each:
  weeks 2-5 | 6-9 | 10-13 | 14-17 | 18-21
Only out-of-sample errors are used: every demo key E_{s-H} and value E_{s} has its origin in the
val/test period (after the backbone's train split), and the query set is the same late-test
origins for every band (those whose oldest band still fits), so bands differ only in demo age.

Per band and family (mean, cos e, cos y, cos e+y; tau fixed to the train-selected values of
eval_raw_sim, see TAUS_FIXED):
  gain@s  : test gain at the train-selected s = 0.55
  gain_opt: gain at the best single scale on these cases, (sum t.e)^2 / (sum e^2 sum t^2)
            (diagnostic: information content of the retrieved error, not a model result)
Diagnostic: for every (case, demo) pair, cos(e_j, target) averaged within deciles of
sim_y and of sim_e (global decile edges from the first band) -> does a more similar state
come with a more similar error, and does that relation decay with age?
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, load_split
from patchmem.eval_raw_sim import W, sim

PERIOD = 168
BANDS = {'w02-05': (2, 3, 4, 5), 'w06-09': (6, 7, 8, 9), 'w10-13': (10, 11, 12, 13),
         'w14-17': (14, 15, 16, 17), 'w18-21': (18, 19, 20, 21)}
TAUS_FIXED = {'mean': (0.0, 0.0), 'e': (2.0, 0.0), 'y': (0.0, 4.0), 'e+y': (2.0, 2.0)}
S_FIXED = 0.55
NDEC = 10


@torch.no_grad()
def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    history = CELLS[code]['history']
    n_total = len(pd.read_csv(Path(exp.args.root_path) / exp.args.data_path))
    n_train, n_test = int(n_total * 0.7), int(n_total * 0.2)
    data = load_split(exp, 'test', history, args.forecast_bs)
    del exp
    torch.cuda.empty_cache()
    E, X = data['E'], data['X']
    assert len(X) == history + n_test
    oos = history - (n_total - n_test - n_train)                    # X index of the first val time step
    max_age = max(max(w) for w in BANDS.values()) * PERIOD
    t_q = data['origins'][data['origins'] - max_age - H >= oos]      # oldest key origin still out-of-sample
    C = X.shape[1]
    print(f'[setup] total {n_total} oos_start {oos} queries {len(t_q)} x {C} channels', flush=True)
    t_all = t_q.repeat_interleave(C)
    c_all = torch.arange(C).repeat(len(t_q))
    k = torch.arange(W)
    edges, out = None, {}
    for band, weeks in BANDS.items():
        ages = torch.tensor([w * PERIOD for w in weeks])
        acc = {f: torch.zeros(3, dtype=torch.float64, device=device) for f in TAUS_FIXED}
        tt = torch.zeros((), dtype=torch.float64, device=device)
        pairs = {'sy': [], 'se': [], 'cv': []}
        for i in range(0, len(t_all), args.bs):
            t, c = t_all[i:i + args.bs], c_all[i:i + args.bs]
            s = t[:, None] - ages[None]
            cc = c[:, None]
            assert bool((s - H >= oos).all())                          # key origin s - H: target y in val/test
            q = E[t - H - L, :, c].to(device)
            target = E[t - L, :, c].to(device).double()
            K = E[s - H - L, :, cc].to(device)
            V = E[s - L, :, cc].to(device).double()
            yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)
            ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)
            se, sy = sim(q, K, 'cos').double(), sim(yq, ys, 'cos').double()
            tt += target.square().sum()
            for f, (te, ty) in TAUS_FIXED.items():
                att = torch.softmax(te * se + ty * sy, -1)
                eh = (att[..., None] * V).sum(1)
                acc[f] += torch.stack([(target * eh).sum(), eh.square().sum(), tt.new_tensor(0.0)])
            pairs['sy'].append(sy.flatten().float().cpu())
            pairs['se'].append(se.flatten().float().cpu())
            pairs['cv'].append(F.cosine_similarity(V, target[:, None], dim=-1, eps=1e-8).flatten().float().cpu())
        tt = float(tt)
        res = {}
        for f, a in acc.items():
            te_, ee = float(a[0]), float(a[1])
            res[f] = dict(gain_at_s=100 * (2 * S_FIXED * te_ - S_FIXED ** 2 * ee) / tt,
                          gain_opt=100 * te_ ** 2 / (ee * tt), s_opt=te_ / ee)
        P = {kk: torch.cat(v) for kk, v in pairs.items()}
        if edges is None:
            qs = torch.linspace(0, 1, NDEC + 1)[1:-1]
            edges = {kk: torch.quantile(P[kk][torch.randperm(len(P[kk]))[:2_000_000]], qs) for kk in ('sy', 'se')}
        dec = {}
        for kk in ('sy', 'se'):
            b = torch.bucketize(P[kk], edges[kk])
            dec[kk] = [float(P['cv'][b == d].mean()) for d in range(NDEC)]
        res['cos_e_target_mean'] = float(P['cv'].mean())
        res['cos_e_target_by_simy_decile'] = dec['sy']
        res['cos_e_target_by_sime_decile'] = dec['se']
        out[band] = res
        print(f'[{band}] ' + ' | '.join(f'{f} {res[f]["gain_at_s"]:+.2f}/{res[f]["gain_opt"]:.2f}' for f in TAUS_FIXED)
              + f' | cos(e_j,target) {res["cos_e_target_mean"]:.3f} simy-dec {dec["sy"][0]:.3f}->{dec["sy"][-1]:.3f}'
              f' sime-dec {dec["se"][0]:.3f}->{dec["se"][-1]:.3f}', flush=True)
    result = dict(dataset=code, seed=seed, H=H, W=W, period=PERIOD, bands=BANDS, taus=TAUS_FIXED, s_fixed=S_FIXED,
                  n_queries=len(t_q), channels=C, oos_start=oos, checkpoint=str(checkpoint),
                  decile_edges={kk: v.tolist() for kk, v in edges.items()}, results=out)
    output = Path(args.out_dir) / f'{code}_H{H}_agebands_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=4096)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
