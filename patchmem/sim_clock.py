# Phase offsets of the FreshCast shape analogs (top-200 cosine) on test queries; no forecaster needed.
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0, '/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch')
import patchmem.eval_freshcast as fc
from sklearn.preprocessing import StandardScaler
for code, f in (('T', 'dataset/traffic/traffic.csv'), ('EL', 'dataset/electricity/electricity.csv')):
    df = pd.read_csv('/project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/' + f)
    v = df[df.columns[1:]].values
    T = len(v); n_tr, n_te = int(T * .7), int(T * .2); n_va = T - n_tr - n_te
    sc = StandardScaler().fit(v[:n_tr])
    X = torch.tensor(sc.transform(v), dtype=torch.float32).cuda()
    ws = fc.win_stats(X); sd = fc.series_desc(X[:n_tr])
    order = torch.randperm(T, generator=torch.Generator().manual_seed(0)).cuda()
    g = torch.Generator().manual_seed(1)
    t = torch.randint(n_tr + n_va, T - fc.H + 1, (4096,), generator=g).cuda()
    c = torch.randint(0, X.shape[1], (4096,), generator=g).cuda()
    offs, cs, best_same, n_above = [], [], [], []
    for i in range(0, 4096, 128):
        tt, cc = t[i:i+128], c[i:i+128]
        sp = fc.build(X, ws, sd, order, tt, cc)
        s = sp['s'][:, 2*fc.N_SEAS:]; cos = sp['z'][:, 2*fc.N_SEAS:, 3]
        d = (tt[:, None] - s)
        offs.append(d.cpu()); cs.append(cos.cpu())
        # best same-clock (daily-aligned) cosine among the full recent pool, from seasonal analog daily/weekly
        best_same.append(sp['z'][:, :fc.N_SEAS, 3].max(1).values.cpu())
    d = torch.cat(offs); cos = torch.cat(cs); bs = torch.cat(best_same)
    dd = d % 24; dist = torch.minimum(dd, 24 - dd)
    print(f'== {code} H{fc.H}: top-{fc.N_SHAPE} shape analogs of 4096 test queries')
    print(' share with daily phase offset 0 (same clock):', float((dist == 0).float().mean()))
    print(' share |offset| <=1h / <=2h / >=6h:', [round(float((dist <= k).float().mean()), 3) for k in (1, 2)], round(float((dist >= 6).float().mean()), 3))
    print(' share weekly-aligned (d%168==0):', float(((d % 168) == 0).float().mean()))
    print(' mean cos same-clock / offset 1-2h / offset >=3h:', [round(float(cos[m].mean()), 3) if m.any() else None for m in (dist == 0, (dist >= 1) & (dist <= 2), dist >= 3)])
    top1 = cos[:, 0]; top1_off = dist[:, 0]
    print(' top-1 analog: share same-clock', float((top1_off == 0).float().mean()), 'share offset>=3h', float((top1_off >= 3).float().mean()))
    print(' per query: #non-same-clock analogs with cos > best seasonal-analog cos (mean, median):', float(((cos > bs[:, None]) & (dist > 0)).sum(1).float().mean()), float(((cos > bs[:, None]) & (dist > 0)).sum(1).float().median()))
    print(' best seasonal-analog cos mean', float(bs.mean()), ' top-1 shape cos mean', float(top1.mean()))
    hist = torch.bincount(dist.flatten().long(), minlength=13)
    print(' offset histogram 0..12h (share):', [round(float(x), 3) for x in hist / hist.sum()])
