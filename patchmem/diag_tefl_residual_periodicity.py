"""Measure same-lead TEFL residual correlation at daily and weekly lags."""
import csv
from pathlib import Path

import numpy as np
import torch

from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model


PERIODS = {'E': (24, 168), 'M1': (96, 672), 'M2': (96, 672),
           'W': (144, 1008), 'T': (24, 168)}
OUT = Path('patchmem/logs/server/itr_tefl_icl/residual_periodicity.csv')


def lag_correlations(values, lag):
    a, b = values[lag:].astype(np.float64), values[:-lag].astype(np.float64)
    a -= a.mean(0)
    b -= b.mean(0)
    cov = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    corr = np.divide(cov, den, out=np.full_like(cov, np.nan), where=den > 1e-12)
    return {'mean_r': float(np.nanmean(corr)), 'median_r': float(np.nanmedian(corr)),
            'positive_frac': float(np.nanmean(corr > 0)),
            'valid_channels': int(np.isfinite(corr).sum())}


@torch.inference_mode()
def run(code, seed=3047):
    device = torch.device('cuda')
    exp, _ = load_model(code, seed, device, 'TEFL_paper_L96_E192')
    W1 = exp.model.W1.weight.detach().to(dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(dtype=torch.float64)
    rows = []
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, 2880, 16)
        residual = np.empty((len(origins), channels), dtype=np.float32)
        for start in range(0, len(origins), 32):
            t = origins[start:start + 32]
            q = E[t - 288].permute(0, 2, 1).to(device=device, dtype=torch.float64)
            target = E[t - 96, 191, :].to(device=device, dtype=torch.float64)
            pred = (torch.relu(q @ W1.T) @ W2.T)[:, :, 191]
            residual[start:start + len(t)] = (target - pred).cpu().numpy()
        for label, lag in zip(('day', 'week'), PERIODS[code]):
            stats = lag_correlations(residual, lag)
            row = {'dataset': code, 'seed': seed, 'split': split, 'lead': 192,
                   'period': label, 'lag': lag, 'windows': len(origins), **stats}
            rows.append(row)
            print(row, flush=True)
        del E
    return rows


def main():
    rows = []
    for code in PERIODS:
        rows.extend(run(code))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
