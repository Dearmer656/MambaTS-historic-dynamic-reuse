"""Decompose W2-only in-context gains into residual/correction alignment."""
import csv
import json
from pathlib import Path

import torch

from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model
from patchmem.eval_itransformer_tefl_icl_periodic import aligned_demo_ages


ROOT = Path('patchmem/logs/server/itr_tefl_icl')
OUT = ROOT / 'update_alignment.csv'
SETTINGS = {
    'E': [('fixed_1440_48', 1440, 48, None)],
    'M1': [('fixed_1440_48', 1440, 48, None), ('fixed_2880_96', 2880, 96, None)],
    'M2': [('fixed_1440_48', 1440, 48, None), ('fixed_2880_96', 2880, 96, None)],
    'W': [('fixed_1440_48', 1440, 48, None), ('fixed_2880_144', 2880, 144, None),
          ('daily_2880', 2880, 144, 'periodic')],
    'T': [('fixed_1440_48', 1440, 48, None), ('weekly_2880', 2880, 168, 'periodic')],
}


@torch.inference_mode()
def run_dataset(code, seed=3047):
    device = torch.device('cuda')
    exp, _ = load_model(code, seed, device, 'TEFL_paper_L96_E192')
    W1 = exp.model.W1.weight.detach().to(dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(dtype=torch.float64)
    fixed = json.loads((ROOT / f'{code}_TEFL_paper_L96_E192_s{seed}.json').read_text())
    periodic_path = ROOT / f'{code}_periodic_s{seed}.json'
    periodic = json.loads(periodic_path.read_text()) if periodic_path.exists() else None
    rows = []
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, 2880, 16)
        for label, N, stride, kind in SETTINGS[code]:
            if kind == 'periodic':
                ages = aligned_demo_ages(N, stride)
                record = periodic['validation' if split == 'val' else 'test'][f'N{N}_P{stride}']
            else:
                ages = list(range(192, N - 192 - 96 + 1, stride))
                record = fixed['validation' if split == 'val' else 'test'][f'N{N}_D{stride}']
            ages = torch.tensor(ages)
            squared = dot = energy = 0.0
            total = 0
            for start in range(0, len(origins), 8):
                t = origins[start:start + 8]
                s = t[:, None] - ages[None]
                q = E[t - 288].permute(0, 2, 1).to(device=device, dtype=torch.float64)
                target = E[t - 96].permute(0, 2, 1).to(device=device, dtype=torch.float64)
                K = E[s - 288].permute(0, 3, 1, 2).to(device=device, dtype=torch.float64)
                V = E[s - 96].permute(0, 3, 1, 2).to(device=device, dtype=torch.float64)
                hq = torch.relu(q @ W1.T)
                hk = torch.relu(K @ W1.T)
                residual = V - hk @ W2.T
                r = target - hq @ W2.T
                scores = (hk * hq[:, :, None]).sum(-1)
                den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
                add = (scores[..., None] * residual).sum(-2) / den[..., None]
                squared += float(r.square().sum())
                dot += float((r * add).sum())
                energy += float(add.square().sum())
                total += r.numel()
            mse, A, B = squared / total, dot / total, energy / total
            selected = (record['chosen'] if split == 'val' else record['arms'])['W2_only']
            a2 = selected['a2']
            gain = 100 * (2 * a2 * A - a2 * a2 * B) / mse
            row = dict(dataset=code,seed=seed,split=split,setting=label,demos=len(ages),
                       mse=mse,dot=A,correction_energy=B,
                       cosine=A / ((mse * B) ** .5) if B > 0 else 0.0,
                       optimal_step=A / B if B > 0 else 0.0,
                       oracle_gain_pct=100 * max(A, 0) ** 2 / (mse * B) if B > 0 else 0.0,
                       selected_step=a2,selected_gain_pct=gain)
            if split == 'test' and abs(gain - selected['gain_pct']) > 1e-4:
                raise AssertionError(f'Gain mismatch {code} {label}: {gain} vs {selected["gain_pct"]}')
            rows.append(row)
            print(row, flush=True)
        del E
    return rows


def main():
    rows = []
    for code in SETTINGS:
        rows.extend(run_dataset(code))
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
