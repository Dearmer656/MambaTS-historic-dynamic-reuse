"""PAT-315: what did the carry learn?  Reads eval_sr --dump files ({code}_H192_s{seed}_{arm}_dump.pt).
Per arm:
  rotation   theta vs its init (weekly harmonics): |d theta| in units of the harmonic spacing 2 pi / (PW q), period (h)
  memory     tau = -1 / log rho (h), init vs trained
  usage      mode weight |B_k| |C_k| (B in, C out): which modes carry signal, their period / memory
  T_t        P: beta, |beta u u^T z| / |z| (fraction of the state the Householder moves), cos(u_t, u_t-1);
             D: gate g;   readout rms / residual rms (how large the carry's output is)
"""
import argparse
import math
from pathlib import Path

import torch


def q3(v):
    q = torch.quantile(v.flatten().float()[:2_000_000], torch.tensor([0.1, 0.5, 0.9]))
    return '/'.join(f'{x:.3g}' for x in q.tolist())


def show(p):
    d = torch.load(p, map_location='cpu')
    st, q, PW = d['state'], d['q'], d['PW']
    print(f'== {p.name}  train_loss {d["train_loss"][0]:.4f} -> {d["train_loss"][-1]:.4f}')
    th, th0 = st['theta'], d['theta0']
    sp = 2 * math.pi / (PW * q)
    dth = (th - th0).abs() / sp
    print(f'  rotation |d theta| / spacing  q10/med/q90 {q3(dth)}  max {float(dth.max()):.3g}'
          f'  (trained mode k still at harmonic k iff < 0.5)')
    tau = lambda r: -1 / torch.log(torch.sigmoid(r)) / q
    t0, t1 = tau(d['rho0']), tau(st['rho_p'])
    print(f'  memory tau (h) init {q3(t0)}  trained {q3(t1)}  |log ratio| med {float((t1 / t0).log().abs().median()):.3g}')
    Bn = st['B'].view(-1, 2, st['B'].shape[1]).norm(dim=(1, 2))
    Cn = st['Cr'].norm(dim=(0, 2))
    w = Bn * Cn
    top = w.argsort(descending=True)[:8]
    per = lambda k: float('inf') if float(th[k]) == 0 else 2 * math.pi / float(th[k].abs()) / q
    share = float(w[top].sum() / w.sum().clamp_min(1e-30))
    print(f'  usage top-8 modes share {share:.2f}: ' +
          ', '.join(f'k{int(k)} P={per(k):.1f}h tau={float(t1[k]):.0f}h' for k in top))
    if 'beta' in d:
        print(f'  PaTH beta {q3(d["beta"])}  |bUUz|/|z| {q3(d["hh_rel"])}  cos(u_t,u_t-1) {q3(d["u_cos"])}'
              f'  |Wu| {float(st["Wu.weight"].norm()):.3g}  wb |w| {float(st["wb.weight"].norm()):.3g}'
              f' bias {float(st["wb.bias"]):.3g}')
    if 'gate' in d:
        print(f'  D gate {q3(d["gate"])}  |wd| {float(st["wd.weight"].norm()):.3g} bias {float(st["wd.bias"]):.3g}')
    r = d['m_rms'].mean() / d['rho_rms'].mean()
    print(f'  readout rms / residual rms {float(r):.3g}  gamma {float(st["gamma"].clamp(0, 2)):.3g}')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--dir', default='patchmem/logs/server/err_sr_dump')
    for p in sorted(Path(a.parse_args().dir).glob('*_dump.pt')):
        show(p)


if __name__ == '__main__':
    main()
