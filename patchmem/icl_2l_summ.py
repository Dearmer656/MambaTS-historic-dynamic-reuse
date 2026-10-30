"""PAT-313 two-layer test-time update (icl2l_<a1>: W1 step + hyb's W2 step on the updated features; icl2lw1_<a1>: W1 step only,
TEFL readout), trained H192 hyb, seed 3047, L1440 test. a1 is chosen per family on the first half of the test windows (MSE),
reported on the second half against hyb (normal) and its TEFL part (icloff); full-window MSE for every a1 is printed too.
usage: python patchmem/icl_2l_summ.py [--seed 3047]"""
import argparse, glob, os, re
import torch

O = 'patchmem/logs/server/icl_2l'
FAMILY = lambda k: k.rsplit('_', 1)[0] + '_'                                                   # icl2l_<a1> -> icl2l_; icl2lz_<a1>_<a2> -> icl2lz_<a1>_


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', default='3047'); ap.add_argument('--dir', default=O); ap.add_argument('--H', default='192')
    a_ = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for D in ('E', 'W', 'T'):
        pre = f'{a_.dir}/{D}_s{a_.seed}_test_L1440H{a_.H}'
        if not os.path.exists(f'{pre}_normal.pt'):
            print(f'{D}: no dumps'); continue
        e = {'hyb': torch.load(f'{pre}_normal.pt')['err'].double(), 'tefl_part': torch.load(f'{pre}_icloff.pt')['err'].double()}
        for f in sorted(glob.glob(f'{pre}_icl2l*.pt')):
            e[re.search(r'_(icl2l[a-z0-9]*_[0-9.]+(?:_[0-9.]+)?)\.pt$', f).group(1)] = torch.load(f)['err'].double()
        N = e['hyb'].shape[0]
        h1, h2 = slice(0, N // 2), slice(N // 2, N)
        m = lambda k, sl: float(e[k][sl].square().mean())
        line = f'{D} s{a_.seed} | hyb all {m("hyb", slice(None)):.4f} half2 {m("hyb", h2):.4f} | TEFL part half2 {m("tefl_part", h2):.4f}'
        for fam in sorted({FAMILY(k) for k in e if k.startswith('icl2l')}):
            ks = [k for k in e if k.startswith(fam)]
            if not ks:
                continue
            best = min(ks, key=lambda k: m(k, h1))
            line += (f' || {fam[:-1]}: a1* {best.split("_")[-1]} (half 1) -> half2 {m(best, h2):.4f} '
                     f'({100 * (m(best, h2) / m("hyb", h2) - 1):+.2f}% vs hyb, {100 * (m(best, h2) / m("tefl_part", h2) - 1):+.2f}% vs TEFL part)')
        print(line, flush=True)
        print('   all windows: ' + ' '.join(f'{k} {m(k, slice(None)):.4f}' for k in sorted(k for k in e if k.startswith('icl2l'))), flush=True)


if __name__ == '__main__':
    main()
