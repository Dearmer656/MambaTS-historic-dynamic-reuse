"""PAT-304 third dataset: Traffic restricted to 32 of its 862 sensors (the full set does not fit the local GPU budget). The sensors are drawn
uniformly at random without replacement with a fixed seed (numpy default_rng(0)) from all 862 value columns, kept in their original column
order, so the subset is reproducible and not picked by any statistic. Writes dataset/traffic32/traffic32.csv (date + 32 columns) and the list
of chosen columns to dataset/traffic32/sensors.txt. Rows, splits (70/10/20 by Dataset_Custom) and scaling are unchanged.
usage (repo root): python patchmem/make_traffic32.py"""
import os
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/traffic/traffic.csv')
cols = [c for c in df.columns if c != 'date']
pick = sorted(np.random.default_rng(0).choice(len(cols), 32, replace=False))
keep = [cols[i] for i in pick]
os.makedirs('dataset/traffic32', exist_ok=True)
df[['date'] + keep].to_csv('dataset/traffic32/traffic32.csv', index=False)
open('dataset/traffic32/sensors.txt', 'w').write(' '.join(keep) + '\n')
print(f'{len(df)} rows, {len(keep)} sensors (target column = last: {keep[-1]}): {keep}')
