"""PAT-315 diag: per-channel train-tail std of the eval_decomp level / log-scale features (standardised series)."""
import sys
import numpy as np
import pandas as pd
import torch
import patchmem.eval_freshcast as fc

for name in sys.argv[1:]:
    df = pd.read_csv(name)
    v = df.values[:, 1:].astype(np.float64)
    T = len(v)
    n_tr = int(T * 0.7)
    mu, sd = v[:n_tr].mean(0), v[:n_tr].std(0)
    X = torch.tensor((v - mu) / sd, dtype=torch.float32)
    m, s = fc.win_stats(X)
    T0 = n_tr - 2016
    lev = (m[T0:n_tr - 192] - X[:n_tr].mean(0)) / X[:n_tr].std(0)
    sc = torch.log(s[T0:n_tr - 192].clamp_min(fc.SD_FLOOR) / X[:n_tr].std(0))
    sdl, sds = lev.std(0), sc.std(0)
    lev_all = (m[n_tr:T - 192] - X[:n_tr].mean(0)) / X[:n_tr].std(0)
    zmax = ((lev_all - lev.mean(0)) / sdl.clamp_min(1e-6)).abs().max(0).values
    print(name, 'C', X.shape[1], 'lev tail-std quantiles', np.quantile(sdl.numpy(), [0, .01, .1, .5]).round(4),
          'scale tail-std q', np.quantile(sds.numpy(), [0, .01, .1, .5]).round(4),
          'max |z level| val/test q50/q99/max', np.quantile(zmax.numpy(), [.5, .99, 1]).round(1))
