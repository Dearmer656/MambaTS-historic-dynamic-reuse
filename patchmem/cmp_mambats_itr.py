"""Test MSE of MambaTS ExpA0 (results/*/metrics.npy = [mae, mse, ...]) vs standalone iTransformer (base_metrics.json)."""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

mam = defaultdict(list)
for p in glob.glob('results/long_term_forecast_*_MambaTS_*_ExpA0s*_0/metrics.npy'):
    m = re.search(r'long_term_forecast_(\w+?)_(\d+)_(\d+)_MambaTS_.*_ExpA0s(\d+)_0', p)
    mam[(m.group(1), int(m.group(2)), int(m.group(3)))].append(float(np.load(p)[1]))
itr = defaultdict(list)
for p in glob.glob('results/iTransformer_base_*_L*_H*_s*/base_metrics.json'):
    m = re.search(r'iTransformer_base_(\w+?)_L(\d+)_H(\d+)_s(\d+)', p)
    itr[(m.group(1), int(m.group(2)), int(m.group(3)))].append(json.loads(Path(p).read_text())['mse'])
for k in sorted(mam):
    v = mam[k]
    print('MambaTS', *k, len(v), f'{np.mean(v):.4f} +- {np.std(v):.4f}')
for k in sorted(itr):
    v = itr[k]
    print('iTrans', *k, len(v), f'{np.mean(v):.4f} +- {np.std(v):.4f}')
