"""PAT-313 offline analysis of the trained hyb's in-context term (fixed 2026-10-26 before results; see the issue).
Inputs from patchmem/slurm/icl_case.sbatch: {D}_s{seed}_test_L1440H96_normal.pt (icl: Delta_t (N, C, 96) in RevIN-normalised units;
err: pred - y (N, 96, C), data scale), ..._icloff.pt (err with the ICL term removed), and the audit log lines (MSE per mode).
  A1/A2  share of the ICL gain kept by iclconst / iclswap<d> / iclproj_*: 1 - dMSE(mode) / dMSE(icloff) (dMSE vs normal);
         variance share of the case-specific part: sum |Delta - mean_t Delta|^2 / sum |Delta|^2.
  B3     shape dictionary: SVD of the (t, c) rows of Delta (explained energy of the top 5 shapes) and k-means (k = 4) of the unit-norm
         rows; each shape and centroid summarised by its energy in the const / linear / daily bases.
  B4     energy of Delta in the nested bases const, + linear, + daily (2 harmonics) and the rest; common (mean over variables) vs
         variable-specific.
  B5     R^2 of the top-3 shape scores on time of day (sin/cos) + day of week (one-hot); alignment cos(Delta_{t,c}, -err_icloff_{t,c})
         over the 96 leads (scale-free per (t, c): RevIN scaling cancels): mean and share > 0.
Window i of the test split predicts from date index border1 + i + L, border1 = len - int(0.2 len) - L (Dataset_Custom).
usage: python patchmem/icl_case.py [--data W,T]"""
import argparse, json, os, re
import numpy as np
import torch

O = 'patchmem/logs/server/icl_case'
CSV = {'W': 'dataset/weather/weather.csv', 'T': 'dataset/traffic32/traffic32.csv'}
PER = {'W': 144, 'T': 24}
L, H = 1440, 96


def basis(n, period):
    t = torch.arange(n, dtype=torch.float64)
    cols = [torch.ones(n), t - t.mean()]
    for h in (1, 2):
        w = 2 * torch.pi * h * t / period
        cols += [torch.sin(w), torch.cos(w)]
    Q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return Q[:, :1], Q[:, :2], Q                                                         # const, lin, daily (nested)


def energy_split(X, Bs):
    """X (M, n) rows; nested bases -> energy shares const, linear, daily, rest."""
    tot = X.square().sum()
    e = [(X @ B).square().sum() / tot for B in Bs]
    return dict(const=float(e[0]), linear=float(e[1] - e[0]), daily=float(e[2] - e[1]), rest=float(1 - e[2]))


def kmeans(X, k=4, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)                                                          # unit rows: cosine assignment
        for j in range(k):
            if (a == j).any():
                c = X[a == j].mean(0)
                C[j] = c / c.norm().clamp_min(1e-12)
    return C, a


def audits(D, s):
    txt = open(f'{O}/{D}_s{s}.txt').read()
    line = [l for l in txt.splitlines() if l.startswith('[R1Mem audit test] L=1440 H= 96')][-1]
    base = float(re.search(r'normal mse ([0-9.]+)', line).group(1))
    d = {k: float(v) for k, v in re.findall(r'\| (\w+) dMSE ([+-][0-9.]+)', line)}
    return base, d


def analyse(D, s):
    nrm = torch.load(f'{O}/{D}_s{s}_test_L{L}H{H}_normal.pt')
    off = torch.load(f'{O}/{D}_s{s}_test_L{L}H{H}_icloff.pt')
    icl = nrm['icl'].double()                                                            # (N, C, 96)
    N, C, n = icl.shape
    base, d = audits(D, s)
    gain = d['icloff']
    kept = {k: 1 - v / gain for k, v in d.items() if k.startswith('icl') and k != 'icloff'}
    mu = icl.mean(0, keepdim=True)
    var_case = float((icl - mu).square().sum() / icl.square().sum())
    Bs = basis(n, PER[D])
    X = icl.reshape(N * C, n)
    comp = energy_split(X, Bs)
    common = float(icl.mean(1, keepdim=True).expand_as(icl).square().sum() / icl.square().sum())
    U, S, Vt = torch.linalg.svd(X - 0, full_matrices=False)
    ev = (S.square() / S.square().sum())[:5].tolist()
    shapes = [dict(energy=ev[i], **energy_split(Vt[i:i + 1], Bs), mean_sign=float(np.sign(Vt[i].mean())),
                   profile=Vt[i][::12].tolist()) for i in range(3)]
    Xu = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sub = Xu[torch.randperm(Xu.shape[0], generator=torch.Generator().manual_seed(0))[:200000]].float()
    Cc, a = kmeans(sub)
    clusters = sorted([dict(share=float((a == j).float().mean()), **energy_split(Cc[j:j + 1].double(), Bs),
                            profile=Cc[j][::12].tolist()) for j in range(Cc.shape[0])], key=lambda z: -z['share'])
    # attributes: time of day / day of week of each window's first target
    with open(CSV[D]) as f:
        next(f)
        dates = np.array([ln.split(',', 1)[0] for ln in f], dtype='datetime64[m]')
    border1 = len(dates) - int(len(dates) * 0.2) - L
    t0 = dates[border1 + L: border1 + L + N]
    assert len(t0) == N, (len(t0), N)
    minutes = t0.astype('int64')
    tod = (minutes % 1440) / 1440.0
    dow = ((minutes // 1440) + 3) % 7                                                     # 1970-01-01 was a Thursday -> Monday = 0
    F = np.column_stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)] + [(dow == k).astype(float) for k in range(7)])
    scores = (X @ Vt[:3].T).reshape(N, C, 3).numpy()
    r2 = []
    for i in range(3):
        y = scores[:, :, i]
        Fx = np.repeat(F, C, 0); yy = y.reshape(-1)
        coef, *_ = np.linalg.lstsq(Fx, yy, rcond=None)
        r2.append(float(1 - ((yy - Fx @ coef) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()))
    eo = -off['err'].double().permute(0, 2, 1)                                           # (N, C, 96): what is left to correct
    cos = (icl * eo).sum(-1) / (icl.norm(dim=-1) * eo.norm(dim=-1)).clamp_min(1e-12)
    return dict(base=base, icloff_dmse=gain, kept=kept, var_case=var_case, components=comp, common_share=common,
                svd_top5=ev, shapes=shapes, clusters=clusters, r2_time=r2, align_mean=float(cos.mean()),
                align_pos=float((cos > 0).double().mean()), N=N, C=C)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='W,T')
    a_ = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    res = {}
    for D in a_.data.split(','):
        for s in (3047, 3048, 3049):
            r = analyse(D, s); res[f'{D}_s{s}'] = r
            k = r['kept']
            print(f'{D} s{s} | MSE {r["base"]:.4f}, ICL off +{100 * r["icloff_dmse"] / r["base"]:.1f}% | gain kept: ' +
                  ' '.join(f'{m[3:]} {v:+.2f}' for m, v in k.items()) + f' | case-specific energy {r["var_case"]:.2f}', flush=True)
            c = r['components']
            print(f'   components const {c["const"]:.2f} linear {c["linear"]:.2f} daily {c["daily"]:.2f} rest {c["rest"]:.2f} | common over '
                  f'variables {r["common_share"]:.2f} | SVD top5 ' + ' '.join(f'{e:.2f}' for e in r['svd_top5']) +
                  f' | R2(time) top3 ' + ' '.join(f'{e:.3f}' for e in r['r2_time']) + f' | align cos mean {r["align_mean"]:.3f}, >0 {r["align_pos"]:.2f}',
                  flush=True)
            for j, cl in enumerate(r['clusters']):
                print(f'   cluster {j}: share {cl["share"]:.2f} | const {cl["const"]:.2f} linear {cl["linear"]:.2f} daily {cl["daily"]:.2f} rest {cl["rest"]:.2f}'
                      f' | profile ' + ' '.join(f'{v:+.2f}' for v in cl['profile']), flush=True)
    json.dump(res, open(f'{O}/icl_case_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
