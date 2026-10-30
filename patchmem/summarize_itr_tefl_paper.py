"""Summarize final paper-protocol TEFL checkpoints without mixing older runs."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PAPER = {'weather.csv': (0.211, 0.247),
         'ETTm1.csv': (0.357, 0.375),
         'ETTm2.csv': (0.239, 0.300)}
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'patchmem/logs/server/itr_tefl_icl/paper_reproduction.csv'


def main():
    data = defaultdict(list)
    for path in sorted((ROOT / 'results').glob('*_TEFL_paper_L96_E192_s*/tefl_metrics.json')):
        item = json.loads(path.read_text())
        assert (item['model'], item['L'], item['H'], item['L_error']) == (
            'iTransformer-TEFL', 96, 192, 192)
        data[item['dataset']].append(item)
    if not data:
        raise FileNotFoundError('No TEFL paper-protocol results')
    rows = []
    for dataset, reps in sorted(data.items()):
        for item in sorted(reps, key=lambda x: x['seed']):
            rows.append({'dataset': dataset, 'kind': 'seed', 'seed': item['seed'],
                         'n_seeds': '', 'mse': item['mse'], 'mae': item['mae'],
                         'paper_mse': PAPER.get(dataset, ('', ''))[0],
                         'paper_mae': PAPER.get(dataset, ('', ''))[1]})
        rows.append({'dataset': dataset, 'kind': 'mean', 'seed': '',
                     'n_seeds': len(reps),
                     'mse': statistics.mean(x['mse'] for x in reps),
                     'mae': statistics.mean(x['mae'] for x in reps),
                     'paper_mse': PAPER.get(dataset, ('', ''))[0],
                     'paper_mae': PAPER.get(dataset, ('', ''))[1]})
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(DEST)


if __name__ == '__main__':
    main()
