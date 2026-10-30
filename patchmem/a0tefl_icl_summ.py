"""Summarise patchmem/a0tefl_icl.py grids -> JSON for the curves (half-2 MSE change vs TEFL, %).
Two step-size rules per family (W1+W2 / W2 / W1): 'cfg' = (a1, a2) chosen per (N, D) on half 1 (as printed by a0tefl_icl.py);
'fixed' = one (a1, a2) per family and dataset, chosen on half 1 by the mean half-1 MSE over all (N, D) cells (curves free of per-cell selection).
usage: python patchmem/a0tefl_icl_summ.py out.json"""
import glob, json, os, re, sys
import torch

FAM = {'W1+W2': lambda k: k[0] > 0 and k[1] > 0, 'W2': lambda k: k[0] == 0, 'W1': lambda k: k[1] == 0}


def main():
    out = {}
    for f in sorted(glob.glob('patchmem/logs/server/a0tefl_icl/*_H*_s*.pt')):
        D, H, s = re.search(r'/(\w)_H(\d+)_s(\d+)\.pt$', f).groups()
        r = torch.load(f)
        B = len(r['tefl']); h1, h2 = slice(0, B // 2), slice(B // 2, B)
        t1, t2 = float(r['tefl'][h1].mean()), float(r['tefl'][h2].mean())
        cells = r['grid']
        keys = next(iter(cells.values()))['cell'].keys()
        d = dict(tefl_all=float(r['tefl'].mean()), f_all=float(r['f'].mean()), tefl_h2=t2, f_h2=float(r['f'][h2].mean()), rows=[], fixed={})
        for fam, ok in FAM.items():
            ks = [k for k in keys if ok(k)]
            best = min(ks, key=lambda k: sum(float(c['cell'][k][h1].mean()) for c in cells.values()))
            d['fixed'][fam] = list(best)
        for (N, Dd), c in sorted(cells.items()):
            row = dict(N=N, D=Dd, n=c['n'])
            for fam in FAM:
                row[f'{fam}_cfg'] = 100 * (c[fam][1] / t2 - 1)
                row[f'{fam}_fixed'] = 100 * (float(c['cell'][tuple(d['fixed'][fam])][h2].mean()) / t2 - 1)
            d['rows'].append(row)
        out[f'{D}_H{H}_s{s}'] = d
        print(f'{D} H{H} s{s}: TEFL h2 {t2:.4f} f h2 {d["f_h2"]:.4f} | fixed steps {d["fixed"]} | best W1+W2 fixed ' +
              f'{min(x["W1+W2_fixed"] for x in d["rows"]):+.2f}%', flush=True)
    json.dump(out, open(sys.argv[1], 'w'))


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    main()
