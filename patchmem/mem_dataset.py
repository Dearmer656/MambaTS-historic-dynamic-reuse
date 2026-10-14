"""Wraps an official Dataset_Custom/Dataset_ETT_hour instance to also return the R preceding non-overlapping input windows
(raw, un-normalised; normalisation happens inside MambaTSMem) needed by the retrieval memory. Valid indices start once R full
windows of history exist, matching the convention used in the PAT-304 training-free retrieval diagnostics (patchmem/exp_a0corr.py)."""
import numpy as np
import torch
from torch.utils.data import Dataset


class MemDataset(Dataset):
    def __init__(self, base_ds, R):
        self.base = base_ds
        self.R = R
        self.Lw = base_ds.seq_len
        self.offset = R * self.Lw

    def __len__(self):
        return max(0, len(self.base) - self.offset)

    def __getitem__(self, idx):
        i = idx + self.offset
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base[i]
        data = self.base.data_x
        mem = np.stack([data[i - r * self.Lw: i - r * self.Lw + self.Lw] for r in range(1, self.R + 1)]).astype(np.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, mem
