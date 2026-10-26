"""PAT-313 TEFL (arXiv 2602.22520) reproduced on A0 (official MambaTS, direct), original setting, joint training from scratch.
Input window of length L + H: context x[:, :L] (the forecast issued one horizon ago), current input x[:, H:] (length L, as A0):
  eps = x[:, L:L+H] - f(x[:, :L])               (d x H, the full error of the forecast issued at t-H; error window = H)
  y'  = f(x[:, H:]) + ReLU(eps W1) W2            (W1: H -> r = 64, W2: r -> H, shared over variables, no bias)
Same f (same weights) for both calls, gradients through both (no stop-gradient: Algorithm 1 shows none).
Training (paper Sec. 3.3 / Alg. 1): warm-up TEFL_WARM = 3 epochs, f only, loss MSE + alpha * SF(y - f) (spectral flatness = geometric /
arithmetic mean of the residual power spectrum over time, per variable; alpha = TEFL_ALPHA = 1, "equal weights", App. D.1); then joint
training of f and g for TEFL_JOINT = 12 epochs with AdamW (lr continues the official schedule). Vali/test use y' (TEFL on).
Unspecified in the paper, our choices: W2 = 0 at init (joint training starts at f), eps in the dataset's scaled units (not RevIN'd),
early stopping (official patience) and best-vali checkpoint over the joint phase only. TEFL_OFF_TEST=1 (default) also reports the test MSE of
the jointly trained f alone (g removed).
Env: TEFL_WARM, TEFL_JOINT, TEFL_ALPHA, TEFL_RANK."""
import copy, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.MambaTS import Model
from utils.tools import EarlyStopping, adjust_learning_rate


def spectral_flatness(r, eps=1e-8):
    """r (b, H, C) -> mean over (b, C) of exp(mean_k log P_k) / mean_k P_k, P = |rfft_time(r)|^2."""
    P = torch.fft.rfft(r, dim=1).abs().square() + eps
    return (P.log().mean(1).exp() / P.mean(1)).mean()


class A0Tefl(nn.Module):
    def __init__(self, base, L, H, rank=64):
        super().__init__()
        self.base, self.L, self.H = base, L, H
        self.W1 = nn.Linear(H, rank, bias=False)
        self.W2 = nn.Linear(rank, H, bias=False)
        nn.init.zeros_(self.W2.weight)
        self.use_g = True

    def forward(self, x, x_mark=None, dec=None, dec_mark=None):
        L, H = self.L, self.H
        assert x.shape[1] == L + H, (x.shape, L, H)
        if self.use_g:
            yc = self.base(x[:, :L], None, None, None)                                  # forecast issued one horizon ago (own window)
            eps = x[:, L:L + H] - yc                                                     # (b, H, C)
            g = self.W2(torch.relu(self.W1(eps.transpose(1, 2)))).transpose(1, 2)
        y = self.base(x[:, H:], None, None, None)                                        # current forecast (called last: VPT state)
        return y + g if self.use_g else y

    def batch_update_state(self, cost):
        self.base.batch_update_state(cost)


class Exp_A0Tefl(Exp_Long_Term_Forecast):
    def _build_model(self):
        a = self.args
        L, H = a.seq_len, a.pred_len
        ab = copy.deepcopy(a)
        base = Model(ab).float()                                                         # f reads L steps
        a.seq_len = L + H                                                                # data windows carry the context
        m = A0Tefl(base, L, H, int(os.environ.get('TEFL_RANK', '64')))
        print(f'A0-TEFL: L {L} H {H} data seq_len {a.seq_len} rank {m.W1.out_features} warm {os.environ.get("TEFL_WARM", "3")} '
              f'joint {os.environ.get("TEFL_JOINT", "12")} alpha {os.environ.get("TEFL_ALPHA", "1")}', flush=True)
        return m

    def train(self, setting):
        a = self.args
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(a.checkpoints, setting); os.makedirs(path, exist_ok=True)
        warm, joint = int(os.environ.get('TEFL_WARM', '3')), int(os.environ.get('TEFL_JOINT', '12'))
        alpha = float(os.environ.get('TEFL_ALPHA', '1'))
        criterion = self._select_criterion()
        m = self.model
        opt = torch.optim.Adam(m.base.parameters(), lr=a.learning_rate)                 # warm-up: f only (official optimiser)
        early = EarlyStopping(patience=a.patience, verbose=True)
        for epoch in range(warm + joint):
            phase = 'warm' if epoch < warm else 'joint'
            if epoch == warm:                                                            # joint: f and g, AdamW, lr continues the schedule
                opt = torch.optim.AdamW(m.parameters(), lr=opt.param_groups[0]['lr'])
            m.use_g = phase == 'joint'
            m.train(); t0 = time.time(); tl = []
            for i, (bx, by, _, _) in enumerate(train_loader):
                opt.zero_grad()
                bx, by = bx.float().to(self.device), by[:, -a.pred_len:].float().to(self.device)
                out = m(bx)
                loss = criterion(out, by)
                if a.VPT_mode in [1]:
                    m.batch_update_state(loss)
                    loss = loss.mean()
                if phase == 'warm':
                    loss = loss + alpha * spectral_flatness(by - out)
                loss.backward(); opt.step(); tl.append(loss.item())
                if (i + 1) % 100 == 0:
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} ({phase}) | loss: {loss.item():.7f}', flush=True)
            m.use_g = phase == 'joint'
            vl, tsl = self.vali(vali_data, vali_loader, criterion), self.vali(test_data, test_loader, criterion)
            print(f'Epoch: {epoch + 1} ({phase}) cost {time.time() - t0:.1f}s | Train Loss: {np.average(tl):.7f} Vali Loss: {vl:.7f} '
                  f'Test Loss: {tsl:.7f}', flush=True)
            if phase == 'joint':
                early(vl, m, path)
                if early.early_stop:
                    print('Early stopping'); break
            if not a.no_lradj:
                adjust_learning_rate(opt, epoch + 1, a)
        m.load_state_dict(torch.load(path + '/checkpoint.pth'))
        m.use_g = True
        return m

    def test(self, setting, test=0):
        super().test(setting, test)
        if os.environ.get('TEFL_OFF_TEST', '1') == '1':                                  # the jointly trained f alone
            _, test_loader = self._get_data(flag='test')
            m = self.model; m.eval(); m.use_g = False; se = n = 0
            with torch.no_grad():
                for bx, by, _, _ in test_loader:
                    out = m(bx.float().to(self.device)).cpu()
                    se += (out - by[:, -self.args.pred_len:].float()).square().sum().item(); n += out.numel()
            m.use_g = True
            print(f'[A0-TEFL test] g removed (f alone) mse {se / n:.6f}', flush=True)
