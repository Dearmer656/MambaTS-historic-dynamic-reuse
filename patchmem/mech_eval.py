"""PAT-304: what does the memory do? Eval-only mechanism study on trained A1 / Q8+drop50 checkpoints (no training), test split
(and the val split for the one fitted constant of the naive error-feedback baseline). Per (L, H), every window (stride), batches built
from a FIXED random permutation of the windows so that batch neighbours are unrelated windows (needed by the swap mode).
  1. causal modes on the Q8 checkpoint: normal | zero (gate 0) | uniform (a = 1/M) | perm (key-value pairing broken) |
     swap (every retrieval reads the slot VALUES of another test window: own keys, own query, foreign corrections) |
     swapbank (keys and values of another window: a complete foreign memory)
  2. what the correction looks like: d = pred(normal) - pred(zero) per window, split into
       level   per (window, variable) mean of d over the horizon
       profile per (window, variable, position in patch) mean of d over the rollout patches, minus the level (a shape repeated every patch)
       rest    the remainder
     and the MSE of pred(zero) + level, + level + profile (how much of the memory's gain each part carries); alignment cos(d, y - pred(zero))
  3. naive error feedback on A1 (no learning): A1's own next-patch errors inside the lookback e_t = x_{t+1} - hist_t, b = mean of the
     last K history errors (level: also over the patch; profile: per position in the patch), forecast = pred(A1) + beta * b repeated over
     the horizon, beta one scalar fitted on val (least squares) and applied to test. How much of the gain does plain error feedback give?
     Same on the Q8 checkpoint's zero mode (its own backbone) and on Q8 itself (is the feedback already inside the memory's correction?). Plus corr(level of d, level of the Q8 model's own history errors).
  4. attention by slot age in the rollout (head- and window-mean mass on the slot written k patches before the query), entropy.
  5. per-window scalars (npz) for stratification: MSE of A1 / Q8 / zero, |level of A1 history error|, level shift lookback -> horizon.
usage (repo root): python patchmem/mech_eval.py --data E|W|T --seed 3047 [--mem Q8|U] [--grid 1440:336,384:96] [--stride 1] [--K 1,3,0]"""
import argparse, copy, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2

WT = ('--task_name long_term_forecast --is_training 0 --features M --seq_len 1440 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 2 '
      '--factor 1 --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA '
      '--batch_size 8 --learning_rate 0.0005 --model MambaTS --data custom').split()
DATA = {  # args, checkpoint setting, A1 des, Q8 des
    'E': (ETTH2, 'long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_{}_0',
          'R1h0s{}', 'R1MK0.5Q8c1h0s{}'),
    'W': (WT + '--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_r1'.split(),
          'long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_{}_0',
          'R1Wh0I1000s{}', 'R1WMK0.5Q8c1h0I1000s{}'),
    'T': (WT + ('--root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 --enc_in 32 --dec_in 32 --c_out 32 '
                '--model_id traffic32_r1').split(),
          'long_term_forecast_traffic32_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_{}_0',
          'R1Th0s{}', 'R1TMK0.5Q8c1h0s{}'),
}
Q8ENV = {'R1_MEM': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'}
MEMARMS = {'Q8': ('', Q8ENV), 'U': ('U', {'R1_MEM': '1', 'R1_MEM_UNIFORM': '1', 'R1_MEM_SLOT_KEEP': '0.5'})}   # U: uniform+drop50


def load(base, ck, des, seed, env):
    for k in ('R1_MEM', 'R1_MEM_QK_DIM', 'R1_MEM_SLOT_KEEP', 'R1_MEM_UNIFORM'):
        os.environ.pop(k, None)
    os.environ.update(env)
    from patchmem.exp_r1 import Exp_R1
    args = official_args(base + ['--seed', str(seed), '--des', des])
    exp = Exp_R1(args)
    m = exp.model
    m.load_state_dict(torch.load(os.path.join(args.checkpoints, ck.format(des), 'checkpoint.pth'), map_location=exp.device))
    m.eval(); m.reset_ids_shuffle(); exp._fix_eval_order()
    return exp


def patch_retrieve(m, state):
    """Wrap R1Mem._retrieve: swap modes read another window's bank (batch roll by one; batches are random windows); in the rollout
    record the attention mass by slot age."""
    orig = m._retrieve

    def f(h_minus, keys, vals, times=None, t_now=None, phase='hist', masks=None):
        mode = state['mode']
        if mode == 'swap':
            vals = [v.roll(1, 0) for v in vals]
        elif mode == 'swapbank':
            vals = [v.roll(1, 0) for v in vals]; keys = [k.roll(1, 0) for k in keys]
        if phase == 'roll' and state.get('rec') is not None:
            K, V = torch.stack(keys, 1), torch.stack(vals, 1)
            inj, a, g = m.mem_net(h_minus, K, V, None)
            Vf = V.flatten(2)                                                           # (b, M, D*N) slot values C_j
            coh = (Vf.mean(1).norm(dim=-1) / Vf.norm(dim=-1).mean(1).clamp_min(1e-12)).sum().item()   # ||mean_j C_j|| / mean_j ||C_j||
            ratio = (inj.flatten(1).norm(dim=1) / h_minus.flatten(1).norm(dim=1).clamp_min(1e-12)).sum().item()
            age = (t_now - torch.as_tensor(times, device=a.device)).cpu().numpy()        # (M,)
            mass = a.mean((0, 1)).double().cpu().numpy() * a.shape[0]                   # summed over windows
            M = a.shape[-1]
            ent = (-(a * a.clamp_min(1e-12).log()).sum(-1) / np.log(M)).mean().item() if M >= 2 else 1.0
            r = state['rec']
            for ag, v in zip(age, mass):
                r['mass'][int(ag)] = r['mass'].get(int(ag), 0.0) + float(v)
            r['n'] += a.shape[0]; r['ent'] += ent * a.shape[0]; r['gate'] += g.mean().item() * a.shape[0]
            r['coh'] = r.get('coh', 0.0) + coh; r['ratio'] = r.get('ratio', 0.0) + ratio
            r['C_rms'] = r.get('C_rms', 0.0) + Vf.square().mean((1, 2)).sqrt().sum().item()
        return orig(h_minus, keys, vals, times, t_now, phase, masks)
    m._retrieve = f


def hist_err(x, hist, P):
    """next-patch errors inside the lookback (b, T-1, P, C): target x[:, P:] (official raw scale)."""
    b, L, c = x.shape
    return x[:, P:].reshape(b, L // P - 1, P, c) - hist


def fb(e, K, kind):
    """error-feedback term from the last K history errors (K = 0: all): level (b, 1, C) or profile (b, P, C)."""
    e = e if K == 0 else e[:, -K:]
    return e.mean((1, 2))[:, None] if kind == 'level' else e.mean(1)


@torch.no_grad()
def run_cell(eA, eQ, flag, L, H, stride, Ks, rec_state, modes, fit=None):
    mA, mQ = eA.model, eQ.model
    P, dev = mA.P, eA.device
    ds = eA._ds(flag, L, H)
    ids = list(range(0, len(ds), stride))
    ids = [ids[i] for i in np.random.default_rng(0).permutation(len(ids))]
    n_out = H // P
    bs = int(os.environ.get('R1_EVAL_BS', '64'))
    S = {}                                                                              # summed squared errors
    add = lambda k, v: S.__setitem__(k, S.get(k, 0.0) + float(v))
    fbs = {}                                                                            # least squares sums for beta: <b, r>, <b, b>
    per = {k: [] for k in ('A1', 'Q8', 'zero', 'A1_lvl_abs', 'shift', 'A1_last1_lvl_abs')}
    corr_xy = []                                                                        # (level of d, level of Q8 hist error) pairs
    nel = 0
    for s0 in range(0, len(ids), bs):
        bx, by = eA._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        oA = mA(bx, n_out)
        preds = {}
        for md in modes:
            rec_state['mode'] = md
            mQ.set_audit(md if md in ('zero', 'uniform', 'perm') else None, 0)
            rec_state['rec'] = rec_state['attn'] if md == 'normal' else None
            o = mQ(bx, n_out)
            preds[md] = o['pred']
            if md == 'normal':
                histQ = o['hist']
            if md == 'zero':
                histZ = o['hist']
        mQ.set_audit(None); rec_state['mode'] = 'normal'; rec_state['rec'] = None
        pA = oA['pred']
        b, c = pA.shape[0], pA.shape[-1]
        nel += pA.numel()
        sq = lambda p: (p - y).square()
        add('A1', sq(pA).sum())
        for md, p in preds.items():
            add(md, sq(p).sum())
        # 2. decomposition of the memory's correction d = normal - zero
        pZ, pN = preds['zero'], preds['normal']
        d = (pN - pZ).reshape(b, n_out, P, c)
        lvl = d.mean((1, 2), keepdim=True)
        prof = d.mean(1, keepdim=True) - lvl
        rz = (y - pZ).reshape(b, n_out, P, c)
        add('zero+level', (rz - lvl).square().sum())
        add('zero+level+profile', (rz - lvl - prof).square().sum())
        dn, rn = d.flatten(1), rz.flatten(1)
        add('cos_num', (dn * rn).sum()); add('d_sq', dn.square().sum()); add('r_sq', rn.square().sum())
        add('frac_win_improved', ((pN - y).square().flatten(1).mean(1) < (pZ - y).square().flatten(1).mean(1)).float().sum())
        add('nwin', b)
        # 3. naive error feedback on A1 and on the Q8 checkpoint's zero mode (its own backbone, memory off in history too)
        eAh = hist_err(bx, oA['hist'], P)
        eZh = hist_err(bx, histZ, P)
        for tag, base, eh in (('A1', pA, eAh), ('zero', pZ, eZh), ('Q8', pN, hist_err(bx, histQ, P))):
            r = (y - base).reshape(b, n_out, P, c)
            for K in Ks:
                for kind in ('level', 'profile'):
                    bb = fb(eh, K, kind)[:, None].expand(b, n_out, P, c)
                    key = f'fb_{tag}_{kind}_K{K}'
                    fbs.setdefault(key, [0.0, 0.0, 0.0])
                    fbs[key][0] += (bb * r).sum().item(); fbs[key][1] += bb.square().sum().item()
                    if fit is not None:                                                 # val-fitted beta applied here
                        add(key, (r - fit[key] * bb).square().sum())
        eQh = hist_err(bx, histQ, P)
        corr_xy.append(torch.stack([lvl.flatten(), fb(eQh, 3, 'level').flatten()], 1).cpu())
        # 5. per-window scalars
        w = lambda p: (p - y).square().flatten(1).mean(1).cpu()
        per['A1'].append(w(pA)); per['Q8'].append(w(pN)); per['zero'].append(w(pZ))
        per['A1_lvl_abs'].append(fb(eAh, 0, 'level').abs().mean((1, 2)).cpu())
        per['A1_last1_lvl_abs'].append(fb(eAh, 1, 'level').abs().mean((1, 2)).cpu())
        per['shift'].append((y.mean(1) - bx[:, -P:].mean(1)).abs().mean(1).cpu())
    out = {k: v / nel for k, v in S.items() if k not in ('cos_num', 'd_sq', 'r_sq', 'frac_win_improved', 'nwin')}
    out['cos_d_resid'] = S['cos_num'] / (S['d_sq'] * S['r_sq']) ** 0.5
    out['frac_win_improved'] = S['frac_win_improved'] / S['nwin']
    out['windows'] = len(ids)
    beta = {k: v[0] / max(v[1], 1e-12) for k, v in fbs.items()}
    xy = torch.cat(corr_xy).double()
    out['corr_dlevel_vs_Q8histerr_level_K3'] = torch.corrcoef(xy.T)[0, 1].item()
    return out, beta, {k: torch.cat(v).numpy() for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=list(DATA)); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default='1440:336,384:96'); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--K', default='1,3,0'); ap.add_argument('--out', default=None)
    ap.add_argument('--mem', default='Q8', choices=list(MEMARMS), help='memory checkpoint: Q8 (Q8+drop50) | U (trained uniform+drop50)')
    a = ap.parse_args()
    base, ck, dA, dQ = DATA[a.data]
    eA = load(base, ck, dA.format(a.seed), a.seed, {})
    if a.mem == 'U':                                                                   # R1MK0.5Q8c1h0 -> R1MUK0.5c1h0
        dQ = dQ.replace('MK0.5Q8', 'MUK0.5')
    eQ = load(base, ck, dQ.format(a.seed), a.seed, MEMARMS[a.mem][1])
    rec_state = dict(mode='normal', rec=None)
    patch_retrieve(eQ.model, rec_state)
    Ks = [int(k) for k in a.K.split(',')]
    modes = ['normal', 'zero', 'uniform', 'perm', 'swap', 'swapbank'] if a.mem == 'Q8' else ['normal', 'zero', 'swap', 'swapbank']
    out = a.out or f'patchmem/logs/mech/{a.data}{MEMARMS[a.mem][0]}_s{a.seed}'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        rec_state['attn'] = dict(mass={}, n=0, ent=0.0, gate=0.0)
        _, beta, _ = run_cell(eA, eQ, 'val', L, H, a.stride, Ks, rec_state, ['normal', 'zero'])
        rec_state['attn'] = dict(mass={}, n=0, ent=0.0, gate=0.0)
        r, beta_test, per = run_cell(eA, eQ, 'test', L, H, a.stride, Ks, rec_state, modes, fit=beta)
        at = rec_state['attn']
        r['attn_mass_by_age'] = {k: v / at['n'] for k, v in sorted(at['mass'].items())}
        r['attn_entropy_norm'] = at['ent'] / at['n']; r['roll_gate'] = at['gate'] / at['n']
        r['value_coherence'] = at['coh'] / at['n']; r['inj_over_h'] = at['ratio'] / at['n']; r['C_rms'] = at['C_rms'] / at['n']
        r['beta_val'] = beta; r['beta_test_oracle'] = beta_test
        res[f'{L}/{H}'] = r
        np.savez(f'{out}_L{L}_H{H}_perwin.npz', **per)
        A = r['A1']
        print(f'[mech {a.data} s{a.seed}] L={L} H={H} windows {r["windows"]} | A1 {A:.4f} | ' +
              ' '.join(f'{k} {r[k]:.4f} ({100 * (A - r[k]) / A:+.1f}%)' for k in modes + ['zero+level', 'zero+level+profile']), flush=True)
        print(f'   cos(d, y-zero) {r["cos_d_resid"]:.3f} | windows improved {r["frac_win_improved"]:.3f} | corr(level d, Q8 hist err) '
              f'{r["corr_dlevel_vs_Q8histerr_level_K3"]:.3f} | roll gate {r["roll_gate"]:.3f} entropy {r["attn_entropy_norm"]:.3f} | '
              f'|inj|/|h-| {r["inj_over_h"]:.3f} C rms {r["C_rms"]:.4f} value coherence {r["value_coherence"]:.3f}', flush=True)
        print('   naive feedback (beta from val): ' + ' '.join(f'{k} {r[k]:.4f}(b={beta[k]:.2f})' for k in sorted(beta)), flush=True)
        print('   attention mass by slot age: ' + ' '.join(f'{k}:{v:.3f}' for k, v in r['attn_mass_by_age'].items()), flush=True)
        json.dump(dict(data=a.data, seed=a.seed, stride=a.stride, grid=res), open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
