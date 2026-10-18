"""PAT-304: one-model-for-all-horizons control for the original MambaTS (A0). A checkpoint trained once at pred_len = Hmax (run.py,
direct multi-step head) is evaluated on the official test split of every shorter horizon H by TRUNCATION: forecast Hmax steps, keep the
first H. Windows are those of the H test set (e.g. 2785 for H = 96), like R1's grid, so the numbers compare with R1 / R1Mem at the same
(L, H) and with the per-horizon numbers of the paper. Same eval protocol as exp.test: set_seed(seed), batch 1, the checkpoint's ATSP order
(one SA solve, shared by all H). H = Hmax reproduces exp.test (sanity check).
usage: python patchmem/a0_trunc_eval.py --setting <checkpoint dir name> --H 96,192,336 [--out f.json] -- <run.py model args of the run>"""
import argparse, json, os, sys, copy
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.tools import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True)
    ap.add_argument('--H', default='96,192,336')
    ap.add_argument('--out', default=None)
    a, rest = ap.parse_known_args()
    args = official_args([r for r in rest if r != '--'])
    exp = Exp_Long_Term_Forecast(args)
    m = exp.model
    m.load_state_dict(torch.load(os.path.join(args.checkpoints, a.setting, 'checkpoint.pth'), map_location=exp.device))
    m.eval()
    if args.VPT_mode in [1]:
        m.reset_ids_shuffle()
    set_seed(args.seed)
    P, res = args.patch_len, {}
    for H in [int(h) for h in a.H.split(',')]:
        assert H <= args.pred_len
        da = copy.copy(args); da.pred_len = H
        _, loader = data_provider(da, 'test')
        se, ae, n = np.zeros(H), np.zeros(H), 0
        with torch.no_grad():
            for bx, by, bxm, bym in loader:
                bx = bx.float().to(exp.device)
                dec = torch.zeros(bx.shape[0], args.label_len + args.pred_len, bx.shape[2], device=exp.device)
                out = m(bx, bxm.float().to(exp.device), dec, None)[:, :H].cpu().numpy()
                y = by[:, -H:].numpy()
                se += ((out - y) ** 2).mean(-1).sum(0); ae += np.abs(out - y).mean(-1).sum(0); n += bx.shape[0]
        se, ae = se / n, ae / n
        res[f'{args.seq_len}/{H}'] = dict(mse=float(se.mean()), mae=float(ae.mean()), windows=n,
                                          patch_mse=[float(se[i:i + P].mean()) for i in range(0, H, P)])
        print(f'[A0 trunc] {a.setting} | L {args.seq_len} H {H} (trained {args.pred_len}) | windows {n} | '
              f'mse {se.mean():.4f} mae {ae.mean():.4f} | patch mse ' + ' '.join(f'{v:.4f}' for v in res[f"{args.seq_len}/{H}"]["patch_mse"]),
              flush=True)
    if a.out:
        json.dump(dict(setting=a.setting, trained_pred_len=args.pred_len, grid=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
