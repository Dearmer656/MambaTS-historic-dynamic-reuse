"""PAT-320 stage 1: one official Timer-XL trained jointly on 7 benchmarks, channel independent (every variable is a univariate
series, so variable counts 7/21/321/862 need no padding).

Data: official splits and per-dataset StandardScaler fitted on train (data_provider.UnivariateDatasetBenchmark, reused as is).
Training: official teacher-forced next-token MSE (672 input, 7 tokens of 96, target shifted by one token), use_norm, AdamW-free
Adam as in OpenLTM, lr 5e-4 cosine over 10 epochs. Each epoch draws min(n_d, PER_EPOCH) training windows from every dataset d
(uniform over (variable, start)), so Traffic/ECL do not dominate. Selection: valid_last loss (last token) on a fixed val
subsample, equal weight per dataset. Test: official rolling protocol (feed predictions back, drop the oldest token),
MSE/MAE for H in 96/192/336/720 on every test window of length 672 + H, all variables.
usage: python pat320/joint.py --seed 2021 [--epochs 10] [--smoke]"""
import argparse, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from data_provider.data_loader import UnivariateDatasetBenchmark
from models.timer_xl import Model

L, P, HS = 672, 96, (96, 192, 336, 720)
DATA = {'ETTh1': 'ETT-small/ETTh1.csv', 'ETTh2': 'ETT-small/ETTh2.csv', 'ETTm1': 'ETT-small/ETTm1.csv',
        'ETTm2': 'ETT-small/ETTm2.csv', 'Weather': 'weather/weather.csv', 'ECL': 'electricity/electricity.csv',
        'Traffic': 'traffic/traffic.csv'}
CFG = dict(input_token_len=P, output_token_len=P, d_model=512, d_ff=2048, e_layers=5, n_heads=8, dropout=0.1,
           activation='relu', output_attention=False, covariate=False, flash_attention=False, pos='rope', use_norm=True)


def load_split(name, flag):
    """Scaled array [T, C] of one official split (val/test start L steps early, as in OpenLTM)."""
    root, f = os.path.split(DATA[name])
    ds = UnivariateDatasetBenchmark(root_path=str(ROOT / 'dataset' / root), data_path=f, flag=flag, size=[L, P, P])
    return ds.data_x.astype(np.float32)


def build_model(dev):
    return Model(SimpleNamespace(**CFG)).to(dev)


def gather(x, idx, length):
    """x: [T, C] tensor on device; idx: [n, 2] (variable, start) -> [n, length, 1]."""
    ar = torch.arange(length, device=x.device)
    return x[idx[:, 1:2] + ar, idx[:, 0:1]].unsqueeze(-1)


def all_idx(T, C, length, dev):
    n = T - length + 1
    v, s = torch.meshgrid(torch.arange(C, device=dev), torch.arange(n, device=dev), indexing='ij')
    return torch.stack([v.reshape(-1), s.reshape(-1)], 1)


@torch.no_grad()
def rollout(model, x0, H):
    """Official rolling forecast: x0 [B, L, 1] -> [B, H, 1]."""
    out, x = [], x0
    for _ in range(math.ceil(H / P)):
        y = model(x, None, None)[:, -P:]
        out.append(y); x = torch.cat([x[:, P:], y], 1)
    return torch.cat(out, 1)[:, :H]


@torch.no_grad()
def test_dataset(model, x, bs=4096):
    """MSE/MAE per H; each H uses every window of length L + H (as the official test with test_pred_len = H)."""
    T, C = x.shape
    idx = all_idx(T, C, L + HS[0], x.device)
    se = {H: 0.0 for H in HS}; ae = {H: 0.0 for H in HS}; cnt = {H: 0 for H in HS}
    for i0 in range(0, len(idx), bs):
        b = idx[i0:i0 + bs]
        pred = rollout(model, gather(x, b, L), HS[-1])       # the forecast uses only the input window
        for H in HS:
            ok = b[:, 1] <= T - L - H                          # windows whose H-step target exists
            if ok.any():
                bo = b[ok]
                y = gather(x, torch.stack([bo[:, 0], bo[:, 1] + L], 1), H)
                d = pred[ok, :H] - y
                se[H] += d.square().sum().item(); ae[H] += d.abs().sum().item(); cnt[H] += d.numel()
    return {H: dict(mse=se[H] / cnt[H], mae=ae[H] / cnt[H], n=cnt[H] // H) for H in HS}


@torch.no_grad()
def val_loss(model, xs, vidx, bs=4096):
    out = {}
    for d, x in xs.items():
        tot, n = 0.0, 0
        for i0 in range(0, len(vidx[d]), bs):
            b = vidx[d][i0:i0 + bs]
            w = gather(x, b, L + P)
            o = model(w[:, :L], None, None)[:, -P:]
            tot += (o - w[:, L:]).square().mean(dim=(1, 2)).sum().item(); n += len(b)
        out[d] = tot / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2021); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--per_epoch', type=int, default=200000); ap.add_argument('--val_n', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--data', default=','.join(DATA)); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'pat_logs' / 'p320'))
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.per_epoch, a.val_n = 1, 2000, 500
    names = a.data.split(',')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    xs = {f: {d: torch.tensor(load_split(d, f), device=dev) for d in names} for f in ('train', 'val', 'test')}
    for d in names:
        print(f'[data] {d}: C {xs["train"][d].shape[1]} | T train {len(xs["train"][d])} val {len(xs["val"][d])} '
              f'test {len(xs["test"][d])}', flush=True)
    vr = np.random.default_rng(0)                                           # fixed val subsample, independent of the seed
    vidx = {}
    for d in names:
        T, C = xs['val'][d].shape; n = (T - L - P + 1) * C
        pick = vr.choice(n, size=min(a.val_n, n), replace=False); pick.sort()
        vidx[d] = torch.tensor(np.stack([pick // (T - L - P + 1), pick % (T - L - P + 1)], 1), device=dev)
    model = build_model(dev)
    print(f'[model] params {sum(p.numel() for p in model.parameters())}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-8)
    tag = f'joint_s{a.seed}' + ('_smoke' if a.smoke else '')
    os.makedirs(a.out, exist_ok=True); ck = os.path.join(a.out, f'{tag}.pth')
    best = float('inf'); hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        parts = []
        for k, d in enumerate(names):
            T, C = xs['train'][d].shape; n_s = T - L - P + 1
            m = min(n_s * C, a.per_epoch)
            pick = rng.choice(n_s * C, size=m, replace=False)
            parts.append(np.stack([np.full(m, k), pick // n_s, pick % n_s], 1))
        allp = np.concatenate(parts); allp = allp[rng.permutation(len(allp))]
        tl = []
        for i0 in range(0, len(allp), a.bs):
            b = allp[i0:i0 + a.bs]
            ws = []
            for k in np.unique(b[:, 0]):
                bk = torch.tensor(b[b[:, 0] == k][:, 1:], device=dev)
                ws.append(gather(xs['train'][names[k]], bk, L + P))
            w = torch.cat(ws)
            loss = (model(w[:, :L], None, None) - w[:, P:]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        sched.step()
        model.eval(); v = val_loss(model, xs['val'], vidx); vm = float(np.mean(list(v.values())))
        hist.append(dict(epoch=ep + 1, train=float(np.mean(tl)), val=v, val_mean=vm))
        print(f'[ep {ep + 1}] {time.time() - t0:.0f}s | train {np.mean(tl):.4f} | val mean {vm:.4f} | '
              + ' '.join(f'{d} {x:.4f}' for d, x in v.items()), flush=True)
        if vm < best:
            best = vm; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    res = dict(seed=a.seed, cfg=CFG, per_epoch=a.per_epoch, bs=a.bs, lr=a.lr, hist=hist, best_val=best, test={})
    for d in names:
        r = test_dataset(model, xs['test'][d]); res['test'][d] = r
        print(f'[test] {d} | ' + ' | '.join(f'H{H} {r[H]["mse"]:.4f}/{r[H]["mae"]:.4f}' for H in HS)
              + f' | avg mse {np.mean([r[H]["mse"] for H in HS]):.4f}', flush=True)
    json.dump(res, open(os.path.join(a.out, f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
