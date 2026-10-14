"""PAT-304: continue training a frozen-A0 checkpoint with the backbone UNFROZEN, with and without the retrieval memory
(patchmem/mem_model.py). Three arms, started from the SAME checkpoint with the SAME budget:
  none -- continue training the plain backbone + head, no memory module (control for "more training alone").
  mem  -- backbone + head + retrieval memory, trained jointly, real forecast MSE loss.
  shuf -- same architecture as mem, but memory keys/values come from a shuffled (unrelated) batch window.
Env: A0C_CKPT (required), A0C_MEM_MODE (none|mem|shuf), A0C_MEM_R (default 2), A0C_MEM_DK, A0C_MEM_LR (retrieval module lr,
default 1e-3), A0C_BACKBONE_LR (default 2e-5), A0C_MEM_EPOCHS (default 15), A0C_MEM_PATIENCE (default 3)."""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.MambaTS import Model
from patchmem.mem_model import MambaTSMem
from patchmem.mem_dataset import MemDataset
from utils.tools import EarlyStopping, set_seed
from utils.metrics import metric
from torch.utils.data import DataLoader


class Exp_Mem(Exp_Long_Term_Forecast):
    def _build_model(self):
        a = self.args
        base = Model(a).float()
        base.load_state_dict(torch.load(os.environ['A0C_CKPT'], map_location='cpu'))
        base.to(self.device)
        set_seed(a.seed)
        with torch.no_grad():
            base.encoder.eval()
            base.encoder(torch.zeros(1, base.encoder.n_vars * (a.seq_len // a.patch_len), a.d_model, device=self.device))
            base.encoder.train()
        mode = os.environ.get('A0C_MEM_MODE', 'mem')
        m = MambaTSMem(base, R=int(os.environ.get('A0C_MEM_R', '2')), dk=int(os.environ.get('A0C_MEM_DK', '0')) or None, mode=mode)
        print(f'A0C_MEM: ckpt {os.environ["A0C_CKPT"]} | mode {mode} | trainable params {sum(p.numel() for p in m.parameters() if p.requires_grad)}', flush=True)
        return m.float()

    def mem_loaders(self):
        a = self.args
        loaders = {}
        for flag in ('train', 'val', 'test'):
            ds, _ = self._get_data(flag)
            mds = MemDataset(ds, self.model.R)
            shuffle = (flag == 'train')
            loaders[flag] = DataLoader(mds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers, drop_last=False)
        return loaders

    def run_eval(self, loader, crit):
        self.model.eval(); tot = n = 0.0
        with torch.no_grad():
            for bx, by, _, _, mem in loader:
                bx = bx.float().to(self.device); by = by.float().to(self.device)[:, -self.args.pred_len:]
                mem_t = None if self.model.mode == 'none' else mem.float().to(self.device)
                pred = self.model(bx, mem_t)
                loss = crit(pred, by)
                tot += loss.item() * bx.shape[0]; n += bx.shape[0]
        return tot / n

    def train(self, setting):
        a = self.args; m = self.model
        loaders = self.mem_loaders()
        crit = nn.MSELoss()
        base_lr = float(os.environ.get('A0C_BACKBONE_LR', '2e-5')); mem_lr = float(os.environ.get('A0C_MEM_LR', '1e-3'))
        groups = [{'params': list(m.base.parameters()), 'lr': base_lr}]
        if m.retr is not None:
            groups.append({'params': list(m.retr.parameters()), 'lr': mem_lr})
        opt = torch.optim.Adam(groups)
        path = os.path.join(a.checkpoints, setting); os.makedirs(path, exist_ok=True)
        early = EarlyStopping(patience=int(os.environ.get('A0C_MEM_PATIENCE', '3')), verbose=True)
        v0 = self.run_eval(loaders['val'], crit); t0 = self.run_eval(loaders['test'], crit)
        print(f'[mem] pre-finetune (loaded checkpoint, mode={m.mode}): val {v0:.7f} test {t0:.7f}', flush=True)
        epochs = int(os.environ.get('A0C_MEM_EPOCHS', '15'))
        for epoch in range(epochs):
            m.train(); tl = []; ts = time.time()
            for i, (bx, by, _, _, mem) in enumerate(loaders['train']):
                opt.zero_grad()
                bx = bx.float().to(self.device); by = by.float().to(self.device)[:, -a.pred_len:]
                mem_t = None if m.mode == 'none' else mem.float().to(self.device)
                pred = m(bx, mem_t)
                loss = crit(pred, by)
                loss.backward(); opt.step(); tl.append(loss.item())
                if (i + 1) % 200 == 0:
                    g = (torch.sigmoid(m.retr.gate).item() * m.retr.max_scale) if m.retr is not None else 0.0
                    print(f'\titers {i + 1} mse {loss.item():.5f} gate_scale {g:.4f}', flush=True)
            vloss = self.run_eval(loaders['val'], crit); tloss = self.run_eval(loaders['test'], crit)
            print(f'Epoch: {epoch + 1} cost {time.time() - ts:.0f}s | Train {np.mean(tl):.7f} Vali {vloss:.7f} Test {tloss:.7f}', flush=True)
            early(vloss, m, path)
            if early.early_stop:
                print('Early stopping'); break
            for g in opt.param_groups:
                g['lr'] *= 0.5
        m.load_state_dict(torch.load(path + '/checkpoint.pth'))
        vfinal = self.run_eval(loaders['val'], crit); tfinal = self.run_eval(loaders['test'], crit)
        print(f'[mem FINAL best-val ckpt] val {vfinal:.7f} test {tfinal:.7f}', flush=True)
        return m

    def test(self, setting, test=0):
        """Overrides the official test() (whose signature/dataloader is incompatible with the memory module's extra windows);
        called unconditionally by run.py after train(). Uses the same best-val checkpoint train() already loaded."""
        if test:
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
        ds, _ = self._get_data('test')
        loader = DataLoader(MemDataset(ds, self.model.R), batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.model.eval(); preds = []; trues = []
        with torch.no_grad():
            for bx, by, _, _, mem in loader:
                bx = bx.float().to(self.device); by = by.float().to(self.device)[:, -self.args.pred_len:]
                mem_t = None if self.model.mode == 'none' else mem.float().to(self.device)
                pred = self.model(bx, mem_t)
                preds.append(pred.cpu().numpy()); trues.append(by.cpu().numpy())
        preds = np.concatenate(preds, 0); trues = np.concatenate(trues, 0)
        mae, mse, *_ = metric(preds, trues)
        print(f'test shape: {preds.shape} {trues.shape}')
        print(f'mse:{mse}, mae:{mae}')
