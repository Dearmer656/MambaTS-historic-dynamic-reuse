"""PAT-307: does the trained hyb's in-context step weight different past errors on test than on train? Reads the R1_ICL_WDUMP files
(patchmem/logs/server/icl_wdump/{D}_s{seed}_{split}_L1440H96.pt, (N, n) GD-step weights of the normal pass, mean over variables;
demo j = in-window origin patch j, lag to the query origin = T-1-K-j patches; L1440 H96: T 30, K 2, n 26, lags 27..2).
Per dataset, seed, split: total weight sum_j w_j (= the weighted mean cosine), the mean normalised profile over lags, its effective
number of demos 1 / sum p^2 (per sample, / n), mass on the 3 most recent demos. Train vs test: Jensen-Shannon divergence of the mean
profiles, against a noise reference (JS between the two halves of the test split in time).
usage: python patchmem/icl_wdump_summ.py"""
import glob, os, sys
import torch

D_DIR = 'patchmem/logs/server/icl_wdump'


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log()).sum()
    return float((kl(p, m) + kl(q, m)) / 2)


def stats(w):
    tot = w.sum(1)
    p = w / tot.clamp_min(1e-12)[:, None]
    prof = p.mean(0)
    eff = (1 / p.square().sum(1)).mean() / w.shape[1]
    return dict(tot=float(tot.mean()), prof=prof, eff=float(eff), recent3=float(p[:, -3:].sum(1).mean()))


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for D in ('W', 'T'):
        for s in (3047, 3048, 3049):
            tr = torch.load(f'{D_DIR}/{D}_s{s}_train_L1440H96.pt').double()
            te = torch.load(f'{D_DIR}/{D}_s{s}_test_L1440H96.pt').double()
            a, b = stats(tr), stats(te)
            h = te.shape[0] // 2
            noise = js(stats(te[:h])['prof'], stats(te[h:])['prof'])
            n = tr.shape[1]
            lags = [n + 1 - j for j in range(n)]                                        # T-1-K-j with T-1-K = n+1
            top = lambda pr: ','.join(f'{lags[i]}' for i in pr.argsort(descending=True)[:3].tolist())
            print(f'{D} s{s} n={n} | N train {tr.shape[0]} test {te.shape[0]} | total weight train {a["tot"]:.3f} test {b["tot"]:.3f} | '
                  f'eff demos train {a["eff"]:.3f} test {b["eff"]:.3f} | recent-3 mass train {a["recent3"]:.3f} test {b["recent3"]:.3f} '
                  f'(uniform {3 / n:.3f}) | top lags train {top(a["prof"])} test {top(b["prof"])} | JS(train, test) {js(a["prof"], b["prof"]):.5f} '
                  f'vs JS(test 1st half, 2nd half) {noise:.5f}', flush=True)
            print('   profile by lag ' + ' '.join(f'{l}' for l in lags[::-1][:8]) + ' ... | train ' +
                  ' '.join(f'{v:.3f}' for v in a['prof'].flip(0)[:8].tolist()) + ' | test ' +
                  ' '.join(f'{v:.3f}' for v in b['prof'].flip(0)[:8].tolist()), flush=True)


if __name__ == '__main__':
    main()
