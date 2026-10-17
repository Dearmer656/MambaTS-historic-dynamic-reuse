"""PAT-304: why does the R1Mem (B2) gain turn negative late in the rollout? Eval-only diagnostics on trained checkpoints (no training).
Per (L, H) on the official split, all windows, the checkpoint's own deterministic eval order, per rollout patch k:
  modes (memory runs): normal | roll_off (memory used in the history only, no retrieval in the rollout) | from<k0> (rollout retrieval
  only for patches < k0) | win<m> (rollout retrieves over the m most recent history slots only) | keep<p> (rollout retrieves over a
  random p% of the history slots, one draw per batch) | tf (teacher forcing: the rollout is
  fed the TRUE previous patch instead of its own forecast) | tf_roll_off. Plain R1 runs: normal | tf.
  retrieval stats per call (history vs rollout steps): normalised attention entropy, max attention weight, gate, ||delta||/||h-||,
  best cosine q.k over the slots, and an out-of-distribution score of the query state: RMS z-score of sg h- against the per-sample mean /
  std of the history queries (the history value is leave-one-out for comparability).
usage: python patchmem/diag_rollout.py --setting <checkpoint dir> [--mem 1] [--L 384,720,1440] [--H 336] [--modes ...] [--out f.json]
       (the remaining model arguments are the r1_etth2.sh ones; R1_* env as in exp_r1.py)"""
import argparse, json, os, sys, textwrap
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.r1_mem import R1Mem
KEEP_GEN = torch.Generator().manual_seed(0)

ETTH2 = ('--task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 '
         '--c_out 7 --model_id ETTh2_r1 --model MambaTS --data ETTh2 --features M --seq_len 1440 --label_len 48 --pred_len 336 '
         '--e_layers 5 --d_layers 2 --factor 1 --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 '
         '--ATSP_solver SA --batch_size 8 --learning_rate 0.001').split()


def official_args(argv):
    """run.py's own argument parser (its source up to parse_args), so every default matches the training run."""
    src = (ROOT / 'run.py').read_text().splitlines()
    i0 = next(i for i, l in enumerate(src) if l.startswith("if __name__"))
    i1 = next(i for i, l in enumerate(src) if 'parser.parse_args()' in l)
    body = textwrap.dedent('\n'.join(src[i0 + 1:i1]))
    ns = {}
    exec('import argparse\n' + body, ns)
    a = ns['parser'].parse_args(argv)
    a.use_gpu = torch.cuda.is_available() and a.use_gpu
    return a


class Rec:
    """Wraps StateMem.forward: one record per retrieval call (history boundaries first, then the rollout steps)."""
    def __init__(self, mem):
        self.mem, self.calls, self.orig = mem, [], mem.forward
        mem.forward = self

    def __call__(self, h, keys, vals, mask=None):
        inj, a, g = self.orig(h, keys, vals, mask)
        mn = self.mem
        with torch.no_grad():
            M = keys.shape[1]
            hq = h.detach().flatten(1)
            q = mn.Wq(hq).reshape(h.shape[0], mn.nh, mn.dh)
            cos = torch.einsum('bhd,bmhd->bhm', torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(keys, dim=-1))
            ent = (-(a * a.clamp_min(1e-12).log()).sum(-1) / np.log(M)) if M >= 2 else torch.ones_like(a[..., 0])
            self.calls.append(dict(hq=hq, M=M, ent=ent.mean(1), amax=a.max(-1).values.mean(1), gate=g.flatten(1).mean(1),
                                   ratio=inj.flatten(1).norm(dim=1) / hq.norm(dim=1).clamp_min(1e-12), cos=cos.max(-1).values.mean(1)))
        return inj, a, g


def run(m, x, y, n_out, mode='normal', rec=None):
    """Forecast with the rollout variant `mode`. Mirrors R1.forward / R1Mem.forward exactly for mode == 'normal'."""
    b, L, c = x.shape
    P, T = m.P, L // m.P
    mem = isinstance(m, R1Mem)
    means = x.mean(1, keepdim=True)
    xc = x - means
    stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    xn = xc / stdev
    perm = m.scan_order(b, x.device)
    xp = xn.permute(0, 2, 1).reshape(b, c, T, P)
    xs = m.to_scan(xp.permute(0, 2, 1, 3), perm)
    tok = m.embed(xs).reshape(b, T * c, -1)
    if mem:
        m.mem_stats = {}
        bstates, states = m.history(tok, xs)
        keys, vals, times, _ = m.bank
        m.bank = None
    else:
        bstates, states = m.history(tok)
    yh = m.to_orig(m.read(bstates), perm)
    outs = [yh[:, -1]]
    ytrue = ((y - means) / stdev).reshape(b, n_out, P, c).permute(0, 1, 3, 2)          # (b, n_out, C, P) original order, normalised
    if rec is not None:
        rec.n_hist = len(rec.calls)
    for k in range(n_out - 1):
        inp = ytrue[:, k] if mode.startswith('tf') else outs[-1]
        states = m.step_time(m.embed(m.to_scan(inp.unsqueeze(1), perm)).squeeze(1), states)
        if mem and vals and m.retrieve:
            use, ks, vs, ts = True, keys, vals, times
            if mode in ('roll_off', 'tf_roll_off'):
                use = False
            elif mode.startswith('from'):
                use = k + 1 < int(mode[4:])                                             # output patch index k + 1 (0-based)
            elif mode.startswith('win'):
                w = int(mode[3:])
                ks, vs, ts = keys[-w:], vals[-w:], times[-w:]
            elif mode.startswith('keep'):                                               # random p% of the slots, one draw per forward
                if k == 0:
                    n = max(1, round(len(keys) * int(mode[4:]) / 100))
                    sel = sorted(torch.randperm(len(keys), generator=KEEP_GEN)[:n].tolist())
                ks, vs, ts = [keys[i] for i in sel], [vals[i] for i in sel], [times[i] for i in sel]
            if use:
                states[-1] = m._retrieve(states[-1], ks, vs, ts, T + k, phase='roll')
        outs.append(m.to_orig(m.read(states[-1]).unsqueeze(1), perm).squeeze(1))
    den = lambda v: v * stdev.unsqueeze(1) + means.unsqueeze(1)
    return den(torch.stack(outs, 1).transpose(-1, -2)).reshape(b, n_out * P, c)


def call_stats(rec):
    """Per retrieval call: batch-summed stats; plus the query OOD score against the per-sample history query distribution."""
    calls, nh = rec.calls, rec.n_hist
    H = torch.stack([cl['hq'] for cl in calls[:nh]], 1)                                # (b, n_hist, DN)
    out = []
    for i, cl in enumerate(calls):
        if i < nh:                                                                       # leave-one-out reference for history calls
            ref = torch.cat([H[:, :i], H[:, i + 1:]], 1)
        else:
            ref = H
        mu, sd = ref.mean(1), ref.std(1).clamp_min(1e-6)
        ood = ((cl['hq'] - mu) / sd).square().mean(1).sqrt()
        out.append({k: cl[k].sum().item() for k in ('ent', 'amax', 'gate', 'ratio', 'cos')} | dict(ood=ood.sum().item(), M=cl['M']))
    return out


@torch.no_grad()
def evaluate(exp, flag, L, H, modes, bs):
    m = exp.model
    P, nH = m.P, H // m.P
    ds = exp._ds(flag, L, H)
    res = {md: dict(se=torch.zeros(nH, dtype=torch.float64), n=0) for md in modes}
    stats = {}
    for s0 in range(0, len(ds), bs):
        bx, by = exp._batch(ds, range(s0, min(s0 + bs, len(ds))))
        bx, y = bx.to(exp.device), by[:, -H:].to(exp.device)
        for md in modes:
            rec = Rec(m.mem_net) if (isinstance(m, R1Mem) and md in ('normal', 'tf')) else None
            try:
                pred = run(m, bx, y, nH, md, rec)
            finally:
                if rec is not None:
                    m.mem_net.forward = rec.orig
            e = (pred - y).reshape(bx.shape[0], nH, P, -1)
            res[md]['se'] += e.square().sum((0, 2, 3)).double().cpu(); res[md]['n'] += bx.shape[0]
            if rec is not None:
                cs = call_stats(rec)
                st = stats.setdefault(md, dict(hist=[], roll=[], n=0))
                hist, roll = cs[:rec.n_hist], cs[rec.n_hist:]
                if not st['hist']:
                    st['hist'] = [dict.fromkeys(hist[0], 0.0) for _ in hist]; st['roll'] = [dict.fromkeys(hist[0], 0.0) for _ in roll]
                for acc, src in zip(st['hist'] + st['roll'], hist + roll):
                    for k, v in src.items():
                        acc[k] += v if k != 'M' else 0
                    acc['M'] = src['M']
                st['n'] += bx.shape[0]
    pm = {md: (r['se'] / (r['n'] * P * m.c)).tolist() for md, r in res.items()}
    for st in stats.values():
        for acc in st['hist'] + st['roll']:
            for k in acc:
                if k != 'M':
                    acc[k] /= st['n']
    return pm, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True)
    ap.add_argument('--mem', type=int, default=1)
    ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--flag', default='test')
    ap.add_argument('--L', default='384,720,1440')
    ap.add_argument('--H', default='336')
    ap.add_argument('--modes', default=None)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.environ['R1_MEM'] = str(a.mem)
    from patchmem.exp_r1 import Exp_R1
    args = official_args(ETTH2 + ['--seed', str(a.seed), '--des', 'diag'])
    exp = Exp_R1(args)
    m = exp.model
    ck = os.path.join(args.checkpoints, a.setting, 'checkpoint.pth')
    m.load_state_dict(torch.load(ck, map_location=exp.device))
    m.eval(); m.reset_ids_shuffle(); exp._fix_eval_order()
    print(f'checkpoint {ck} | {type(m).__name__} | eval order {exp.eval_order()}', flush=True)
    modes = a.modes.split(',') if a.modes else (['normal', 'roll_off', 'from2', 'from3', 'from4', 'win4', 'win8', 'win16', 'tf', 'tf_roll_off']
                                                 if a.mem else ['normal', 'tf'])
    out = {}
    for L in map(int, a.L.split(',')):
        for H in map(int, a.H.split(',')):
            pm, stats = evaluate(exp, a.flag, L, H, modes, a.bs)
            out[f'{L}/{H}'] = dict(patch_mse=pm, stats=stats)
            print(f'== {a.flag} L={L} H={H} per-patch MSE (mean)', flush=True)
            for md in modes:
                print(f'  {md:12s} ' + ' '.join(f'{v:.4f}' for v in pm[md]) + f' | mean {np.mean(pm[md]):.4f}', flush=True)
            for md, st in stats.items():
                print(f'  [{md}] retrieval stats: step (history last 3, then rollout) | slots ent amax gate ratio cos ood', flush=True)
                rows = [('h', i, r) for i, r in enumerate(st['hist'])][-3:] + [('r', i, r) for i, r in enumerate(st['roll'])]
                for ph, i, r in rows:
                    print(f'    {ph}{i:<3d} M {r["M"]:3d} | ent {r["ent"]:.3f} amax {r["amax"]:.3f} gate {r["gate"]:.3f} '
                          f'ratio {r["ratio"]:.3f} cos {r["cos"]:.3f} ood {r["ood"]:.3f}', flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
