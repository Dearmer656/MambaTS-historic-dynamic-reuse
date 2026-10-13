"""PAT-304 A1 experiment: official run.py / data / seeding / early stopping / val+test protocol, with the model swapped for the time-major
patch-state model (memory OFF) and a per-patch-origin training loss. Evaluation (vali/test) uses the last origin only = the official forecast."""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.MambaTS import Model
from patchmem.tm_model import TMMambaTS, a1_loss, patch_targets
from utils.tools import EarlyStopping, adjust_learning_rate


class A1Model(TMMambaTS):
    def __init__(self, configs):
        super().__init__(Model(configs))

    def train_perm(self, b, dev):
        enc = self.base.encoder
        ids = torch.argsort(torch.rand(b, self.n_vars, device=dev), dim=1)          # same law as utils.masking.random_shuffle
        ids_ = torch.cat([ids[:, [0]], ids], dim=-1)
        enc.transition_tuple = torch.stack([ids_[:, :-1], ids_[:, 1:]], dim=-1); enc.ids_shuffle = None
        return ids

    def eval_perm(self, b, dev):
        enc = self.base.encoder
        if enc.ids_shuffle is None:                                                   # let the official encoder solve the ATSP order (SA) once
            with torch.no_grad():
                enc(torch.zeros(1, self.n_vars, self.base.encoder.d_model, device=dev))
        return enc.ids_shuffle[:1].expand(b, -1)

    def scan(self, x):
        perm = self.train_perm(x.shape[0], x.device) if self.training else self.eval_perm(x.shape[0], x.device)
        return super().forward(x, perm)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.scan(x_enc)['pred'][:, -1]

    def batch_update_state(self, cost): self.base.batch_update_state(cost)
    def reset_ids_shuffle(self): self.base.reset_ids_shuffle()


class Exp_A1(Exp_Long_Term_Forecast):
    def _build_model(self):
        return A1Model(self.args).float()

    def train(self, setting):
        a = self.args; H, P = a.pred_len, a.patch_len
        train_data, train_loader = self._get_data('train'); vali_data, vali_loader = self._get_data('val'); test_data, test_loader = self._get_data('test')
        path = os.path.join(a.checkpoints, setting); os.makedirs(path, exist_ok=True)
        early = EarlyStopping(patience=a.patience, verbose=True); opt = self._select_optimizer(); crit = self._select_criterion()
        assert not a.use_amp and a.features == 'M'
        for epoch in range(a.train_epochs):
            self.model.train(); t0 = time.time(); tl, per_ep = [], []
            for i, (bx, by, _, _) in enumerate(train_loader):
                opt.zero_grad(); bx = bx.float().to(self.device); yf = by.float().to(self.device)[:, -H:]
                pred = self.model.scan(bx)['pred']                                    # (b, n_patch, H, C)
                err = crit(pred, patch_targets(bx, yf, P, H))                         # reduction='none'
                self.model.batch_update_state(rearrange(err, 'b n h c -> b (n h) c'))  # ATSP cost: per-sample loss over all origins
                loss = err.mean(); per = err.detach().mean((0, 2, 3))
                tl.append(loss.item()); per_ep.append(per.cpu().numpy())
                if (i + 1) % 100 == 0: print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}', flush=True)
                loss.backward(); opt.step()
            print(f'Epoch: {epoch + 1} cost time: {time.time() - t0}')
            vloss = self.vali(vali_data, vali_loader, crit); tloss = self.vali(test_data, test_loader, crit)
            print(f'Epoch: {epoch + 1}, Steps: {len(train_loader)} | Train Loss: {np.average(tl):.7f} Vali Loss: {vloss:.7f} Test Loss: {tloss:.7f}')
            print('  per-origin train MSE:', np.round(np.mean(per_ep, 0), 4).tolist(), flush=True)
            early(vloss, self.model, path)
            if early.early_stop: print('Early stopping'); break
            if not a.no_lradj: adjust_learning_rate(opt, epoch + 1, a)
        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        return self.model
