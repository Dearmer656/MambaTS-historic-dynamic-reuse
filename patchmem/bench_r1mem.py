"""PAT-304 Stage 2: speed and GPU memory of every arm (A1 = R1, B0..B3 = R1Mem variants) at fixed lookbacks / horizons, real model
configs, random-walk data (timing does not depend on the values). Per (arm, L, H): training step (forward + backward + Adam step, the
arm's own losses: future MSE, + lambda_corr * L_corr where the writer loss is on), evaluation forward (no_grad, batch 64), peak memory.
usage: python patchmem/bench_r1mem.py [--data etth2|weather] [--L 384,1440] [--H 96,336] [--steps 20] [--out file.json]"""
import argparse, json, sys, time
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.r1_model import R1
from patchmem.r1_mem import R1Mem

CFG = dict(etth2=dict(enc_in=7, e_layers=5, d_model=16, dropout=0.3, lr=1e-3),       # patchmem/r1_etth2.sh
           weather=dict(enc_in=21, e_layers=3, d_model=32, dropout=0.2, lr=5e-4))    # patchmem/r1_weather.sh
ARMS = dict(A1=None, B0=dict(lam_c=0.0), B1=dict(value_detach=True), B2=dict(), B3=dict(retrieve=False))


def build(data, arm):
    c = CFG[data]
    cfg = Namespace(task_name='long_term_forecast', seq_len=1440, pred_len=336, patch_len=48, stride=48, d_model=c['d_model'],
                    e_layers=c['e_layers'], enc_in=c['enc_in'], dropout=c['dropout'], VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
    torch.manual_seed(0)
    spec = ARMS[arm]
    if spec is None:
        return R1(cfg).cuda(), 0.0, c['lr']
    kw = {k: v for k, v in spec.items() if k != 'lam_c'}
    m = R1Mem(cfg, **kw).cuda()
    lam_c = spec.get('lam_c', 1.0)
    m.corr_aux = lam_c > 0
    return m, lam_c, c['lr']


def timed(fn, steps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='etth2', choices=list(CFG))
    ap.add_argument('--L', default='384,1440')
    ap.add_argument('--H', default='96,336')
    ap.add_argument('--arms', default='A1,B0,B1,B2,B3')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--eval_bs', type=int, default=64)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(','):
        m, lam_c, lr = build(a.data, arm)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for L in map(int, a.L.split(',')):
            for H in map(int, a.H.split(',')):
                nH = H // 48
                x = torch.randn(a.bs, L, m.c, device='cuda').cumsum(1) * 0.1
                y = torch.randn(a.bs, H, m.c, device='cuda')
                xe = torch.randn(a.eval_bs, L, m.c, device='cuda').cumsum(1) * 0.1

                def train_step():
                    m.train()
                    opt.zero_grad()
                    out = m(x, nH)
                    loss = (out['pred'] - y).square().mean()
                    if lam_c > 0 and getattr(m, 'corr_loss', None) is not None:
                        loss = loss + lam_c * m.corr_loss
                    loss.backward()
                    opt.step()

                @torch.no_grad()
                def eval_fwd():
                    m.eval()
                    m(xe, nH)

                torch.cuda.reset_peak_memory_stats()
                ts = timed(train_step, a.steps)
                pk_tr = torch.cuda.max_memory_allocated() / 2 ** 20
                m.eval(); m.reset_ids_shuffle()
                with torch.no_grad():
                    m(xe[:1], nH)                                                       # solve / cache the eval order outside the timing
                torch.cuda.reset_peak_memory_stats()
                te = timed(eval_fwd, max(a.steps // 2, 3))
                pk_ev = torch.cuda.max_memory_allocated() / 2 ** 20
                r = dict(data=a.data, arm=arm, L=L, H=H, train_step_s=ts, train_peak_mb=pk_tr, eval_fwd_s=te, eval_peak_mb=pk_ev,
                         train_bs=a.bs, eval_bs=a.eval_bs, slots=L // 48 - 1 if arm != 'A1' else 0)
                rows.append(r)
                print(f'{a.data} {arm:3s} L={L:4d} H={H:3d} | train step {ts * 1e3:7.1f} ms (bs {a.bs}) peak {pk_tr:7.1f} MB | '
                      f'eval fwd {te * 1e3:7.1f} ms (bs {a.eval_bs}) peak {pk_ev:7.1f} MB | history slots {r["slots"]}', flush=True)
        del m, opt
        torch.cuda.empty_cache()
    base = {(r['L'], r['H']): r for r in rows if r['arm'] == 'A1'}
    if base:
        print('== relative to A1 (train step time / eval forward time)')
        for r in rows:
            if r['arm'] != 'A1':
                b = base[(r['L'], r['H'])]
                print(f'  {r["arm"]} L={r["L"]:4d} H={r["H"]:3d} | train x{r["train_step_s"] / b["train_step_s"]:.2f} | '
                      f'eval x{r["eval_fwd_s"] / b["eval_fwd_s"]:.2f}')
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
