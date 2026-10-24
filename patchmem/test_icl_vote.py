"""Unit test for patchmem/icl_vote.votes: mean / median over the right experts, agreement gate and fallback."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import patchmem.icl_lag as IL
IL.dev = 'cpu'
import patchmem.icl_vote as V
names = ['zero', 'mean', 'recent4', 'recent8', 'recent12', 'single1', 'single2', 'single3', 'pair2', 'pair3']
pr = torch.zeros(2, len(names), 1, 3)
pr[:, 3] = 7.0                                                                          # recent8
pr[0, 5:8, 0, 0] = torch.tensor([1., 2., 3.]); pr[0, 8:10, 0, 0] = torch.tensor([10., 20.])   # all positive -> agree
pr[0, 5:8, 0, 1] = torch.tensor([1., -2., 3.]); pr[0, 8:10, 0, 1] = torch.tensor([-1., -4.])  # 2+/3- -> a = 0.2
o = V.votes(names, pr)
assert o[('vmean', 'single')][0, 0, 0] == 2 and o[('vmean', 'all')][0, 0, 0] == 7.2
assert o[('vmedian', 'single')][0, 0, 0] == 2 and o[('vmedian', 'all')][0, 0, 0] == 3
assert o[('vagree', 'all_t0.9')][0, 0, 0] == 3, 'agree -> median'
assert o[('vagree', 'all_t0.5')][0, 0, 1] == 7, 'disagree -> recent8'
assert o[('vagree', 'single_t0.5')][0, 0, 1] == 7 and abs(((pr[0, 5:8, 0, 1] > 0).float().mean() * 2 - 1).abs() - 1 / 3) < 1e-6
print('ALL PASS')
