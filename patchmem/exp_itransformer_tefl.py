"""Original TEFL training protocol on the iTransformer backbone.

The TEFL context has L steps and the error window has H steps.  The data
provider supplies L+H steps, with the final H steps containing the outcome
of the forecast issued from the first L steps.  Training is inherited from
Exp_A0Tefl: three warm-up epochs with MSE + spectral flatness, followed by
12 joint epochs using AdamW, with the best joint-validation checkpoint.
"""
import copy
import json
import os
from pathlib import Path

import torch

from patchmem.exp_a0tefl import A0Tefl, Exp_A0Tefl


class Exp_iTransformerTefl(Exp_A0Tefl):
    def _build_model(self):
        args = self.args
        L, H = args.seq_len, args.pred_len
        assert L > 0 and H > 0
        base_args = copy.deepcopy(args)
        base = self.model_dict['iTransformer'].Model(base_args).float()
        args.seq_len = L + H
        model = A0Tefl(base, L, H, int(os.environ.get('TEFL_RANK', '64')))
        print(f'iTransformer-TEFL: L {L} H {H} L_error {H} data seq_len {args.seq_len} '
              f'rank {model.W1.out_features} warm {os.environ.get("TEFL_WARM", "3")} '
              f'joint {os.environ.get("TEFL_JOINT", "12")} alpha {os.environ.get("TEFL_ALPHA", "1")}', flush=True)
        return model

    def test(self, setting, test=0):
        if test:
            setting = setting.replace(f'_sl{self.model.L}_', f'_sl{self.model.L + self.model.H}_')
            checkpoint = Path(self.args.checkpoints) / setting / 'checkpoint.pth'
            print(f'loading model {checkpoint}', flush=True)
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        _, loader = self._get_data(flag='test')
        model = self.model
        model.eval()
        squared = absolute = base_squared = base_absolute = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in loader:
                x = batch_x.float().to(self.device)
                x_mark = batch_x_mark.float().to(self.device)
                true = batch_y[:, -self.args.pred_len:].float().to(self.device)
                model.use_g = True
                corrected = model(x, x_mark)
                delta = corrected - true
                squared += delta.square().sum().item()
                absolute += delta.abs().sum().item()
                if os.environ.get('TEFL_OFF_TEST', '1') == '1':
                    model.use_g = False
                    base_delta = model(x, x_mark) - true
                    base_squared += base_delta.square().sum().item()
                    base_absolute += base_delta.abs().sum().item()
                count += true.numel()
        model.use_g = True
        assert count > 0
        metrics = {'dataset': self.args.data_path, 'model': 'iTransformer-TEFL',
                   'L': model.L, 'H': model.H, 'L_error': model.H,
                   'seed': self.args.seed, 'elements': count,
                   'mse': squared / count, 'mae': absolute / count}
        if os.environ.get('TEFL_OFF_TEST', '1') == '1':
            metrics.update(base_mse=base_squared / count, base_mae=base_absolute / count)
        output = Path(self.args.results) / setting
        output.mkdir(parents=True, exist_ok=True)
        (output / 'tefl_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(f'mse:{metrics["mse"]}, mae:{metrics["mae"]}', flush=True)
        print(f'TEFL metrics: {json.dumps(metrics)}', flush=True)
        return metrics
