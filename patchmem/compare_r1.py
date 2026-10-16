"""PAT-304: paired comparison of two R1 / R1Mem arms from their diagnostics JSONL (patchmem/logs/diag/<model_id>_<des>_s<seed>.jsonl,
written by exp_r1). For every seed and (L, H): MSE / MAE of both arms, gain = 100 (MSE_a - MSE_b) / MSE_a, the per-rollout-patch
difference Delta_k = MSE_a,k - MSE_b,k (k = 1..H/P), then mean +- std over seeds and, per H, the least-squares slope of the gain against
L per seed. Also prints each run's evaluation scan order pi: different orders are a confound (rerun with a shared R1_EVAL_PERM).
usage: python patchmem/compare_r1.py --a ETTh2_r1_R1h0 --b ETTh2_r1_R1Mc1h0 --seeds 3047,3048,3049 [--flag test] [--dir patchmem/logs/diag]"""
import argparse, json, os
import numpy as np


def last_record(path, flag):
    rec = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('record') == 'test' and r.get('flag') == flag:
                rec = r
    if rec is None:
        raise SystemExit(f'no test record with flag={flag} in {path}')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='baseline run prefix <model_id>_<des without the s<seed> suffix>, e.g. ETTh2_r1_R1h0')
    ap.add_argument('--b', required=True, help='compared run prefix, e.g. ETTh2_r1_R1Mc1h0')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--flag', default='test')
    ap.add_argument('--dir', default=os.path.join('patchmem', 'logs', 'diag'))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    gains, deltas, maes = {}, {}, {}
    for s in seeds:
        ra = last_record(os.path.join(a.dir, f'{a.a}s{s}_s{s}.jsonl'), a.flag)
        rb = last_record(os.path.join(a.dir, f'{a.b}s{s}_s{s}.jsonl'), a.flag)
        print(f'== seed {s} | {a.flag} | order a {ra["order"]} | b {rb["order"]}{"" if ra["order"] == rb["order"] else "  <-- DIFFERENT pi"}')
        for key in ra['grid']:
            ga, gb = ra['grid'][key], rb['grid'][key]
            g = 100 * (ga['mse'] - gb['mse']) / ga['mse']
            dk = [x - y for x, y in zip(ga['patch_mse'], gb['patch_mse'])]
            gains.setdefault(key, []).append(g); deltas.setdefault(key, []).append(dk); maes.setdefault(key, []).append(ga['mae'] - gb['mae'])
            print(f'  {key:>9} | mse a {ga["mse"]:.5f} b {gb["mse"]:.5f} gain {g:+.2f}% | mae a {ga["mae"]:.5f} b {gb["mae"]:.5f} | '
                  'Delta_k ' + ' '.join(f'{v:+.5f}' for v in dk))
    print(f'== mean +- std over {len(seeds)} seeds (gain %, Delta_k = MSE_a,k - MSE_b,k; positive = b better)')
    for key in gains:
        g, d = np.array(gains[key]), np.array(deltas[key])
        print(f'  {key:>9} | gain {g.mean():+.2f} +- {g.std(ddof=1) if len(g) > 1 else 0:.2f}% | dMAE {np.mean(maes[key]):+.5f} | Delta_k ' +
              ' '.join(f'{m:+.5f}+-{sd:.5f}' for m, sd in zip(d.mean(0), d.std(0, ddof=1) if len(d) > 1 else np.zeros(d.shape[1]))))
    print('== gain vs lookback L per H (least-squares slope, % per 100 steps of L, per seed)')
    Hs = sorted({int(k.split('/')[1]) for k in gains})
    for H in Hs:
        ks = sorted([k for k in gains if int(k.split('/')[1]) == H], key=lambda k: int(k.split('/')[0]))
        Ls = np.array([int(k.split('/')[0]) for k in ks], dtype=float)
        if len(Ls) < 2:
            continue
        slopes = [100 * np.polyfit(Ls, [gains[k][i] for k in ks], 1)[0] for i in range(len(seeds))]
        print(f'  H={H:3d} L={list(map(int, Ls))} | slope per seed ' + ' '.join(f'{v:+.3f}' for v in slopes) +
              f' | mean {np.mean(slopes):+.3f}')


if __name__ == '__main__':
    main()
