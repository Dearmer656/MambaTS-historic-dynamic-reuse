"""PAT-307 task 6: where does the in-context error-correction gain come from? Training-free diagnostics on the test split (official
test origins), fixed 2026-10-25 before any result. Errors E[origin, lead, var] of base a1 | tefl | a0 exactly as patchmem/win_check.py.

Per cell (data, base, seed, L, H):
  lead     MSE per lead h for base / win_row (/ krexp_row on Traffic): is the gain early or late in the horizon?
  day      per-day (E, T: 24 origins, W: 144; buckets start at the first test origin, not calendar-aligned; partial last day dropped) gain: fraction of days improved, share of the net gain from the best 10% of days, corr(base day
           error, day gain): is the gain spread over time or a few episodes (shift / outages)?
  var      per-variable gain %: spread over channels or a few?
  oracle   in-sample (cheating) ridge on ALL test origins, lam 1, pooled over variables: fraction of the base MSE removable by a linear map
           from the row feature, from the ced feature, by a per-variable/lead mean (bias). An upper bound on what any windowed fit of
           that family can reach; compares bases (why A0 gains nothing).
  W        structure of the oracle row map W (H x H; feature i = error on y[t-H+i-1], output j = error on y[t+j-1]): energy share of the
           freshest 10% feature rows, of the diagonal (same lead), top singular value share: fresh-error carry vs lead transport vs offset.
  lag      softmax (soft1, row keys, as win_check --arms attn) weights at ~400 evenly spaced test origins: weight mass on context origins
           whose lag t - j is a multiple of a day / week, divided by the uniform expectation; effective neighbour share 1 / (m sum w^2):
           does selection retrieve the same time of day / week? Same for the krexp effective weights |k_q (K + lam I)^-1| (normalised).
           Unit test: an exactly 6-periodic error gives soft1 day ratio only ~1.5 (max ~6): the median-heuristic softmax is broad.
Pre-stated expectations (reported, not tuned): A0 oracle(row) << A1 oracle(row); Traffic day ratio above ETTh2 / Weather.
usage: python patchmem/gain_src.py --data E|W|T --base a1|tefl|a0 [--seeds ..] [--L 720,1440] [--H 96] [--smoke]"""
import argparse, json, os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import patchmem.efc_cet as ec
import patchmem.win_check as wc
dev = ec.dev
DAY = {'E': 24, 'W': 144, 'T': 24}
WEEK = {'E': 168, 'W': 1008, 'T': 168}


def oracle(F, Y):
    """F, Y (n, C, H): in-sample pooled ridge (lam 1) -> (removable share of sum Y^2, W)."""
    H = F.shape[-1]
    S = torch.einsum('nch,ncg->hg', F, F).double(); T = torch.einsum('nch,ncg->hg', F, Y).double()
    W = torch.linalg.solve(ec.LAM * torch.eye(H, device=F.device, dtype=torch.float64) + S, T)
    sse = (Y.double() - F.double() @ W).square().sum()
    return float(1 - sse / Y.double().square().sum().clamp_min(1e-30)), W


def w_struct(W):
    H = W.shape[0]; e = W.square(); k = max(1, H // 10)
    s = torch.linalg.svdvals(W).square()
    return dict(fresh10=float(e[-k:].sum() / e.sum()), diag=float(e.diagonal().sum() / e.sum()), top1=float(s[0] / s.sum()))


@torch.no_grad()
def lag_stats(E, H, first, last, L, day, week, n_t=400):
    """soft1 weights (row keys, unit, per variable, s2 median heuristic) over the window [t-L+H, t-H] as in win_check.stream_ctrl."""
    D, valid = ec.row_feat(E, H)
    F = D.permute(0, 2, 1).float()
    vcpu = valid.cpu()
    ts = torch.linspace(first, last, min(n_t, last - first + 1)).long().tolist()
    md = mw = ne = kd = kw = 0.0; ud = uw = 0.0; n = 0
    for t in ts:
        lo, hi = max(0, t - L + H), t - H
        if hi < lo or not bool(vcpu[t]):
            continue
        idx = torch.arange(lo, hi + 1)[vcpu[lo:hi + 1]]
        if len(idx) < H + 1:                                                                     # as stream_ctrl (nv = H)
            continue
        Kf = wc._unit(F[idx.to(F.device)]).transpose(0, 1); Kq = wc._unit(F[t:t + 1]).transpose(0, 1)
        s2 = wc._sigma2(Kf).double()
        A = (Kq @ Kf.transpose(1, 2)).double().sub(1) / s2
        w = torch.softmax(A, dim=-1)[:, 0]                                                       # (C, m)
        m = Kf.shape[1]
        Km = torch.exp((Kf @ Kf.transpose(1, 2)).double().sub(1) / s2) + ec.LAM * torch.eye(m, device=Kf.device, dtype=torch.float64)
        a = (torch.exp(A) @ torch.cholesky_inverse(torch.linalg.cholesky(Km)))[:, 0].abs()
        a = a / a.sum(1, keepdim=True).clamp_min(1e-300)
        lag = (t - idx).to(w.device)
        isd, isw = (lag % day == 0), (lag % week == 0)
        md += float(w[:, isd].sum(1).mean()); mw += float(w[:, isw].sum(1).mean())
        kd += float(a[:, isd].sum(1).mean()); kw += float(a[:, isw].sum(1).mean())
        ud += float(isd.float().mean()); uw += float(isw.float().mean())
        ne += float((1 / w.square().sum(1)).mean() / len(idx)); n += 1
    if n == 0:
        return None
    return dict(day_mass=md / n, day_uniform=ud / n, day_ratio=(md / max(ud, 1e-12)), week_mass=mw / n, week_uniform=uw / n,
                week_ratio=(mw / max(uw, 1e-12)), eff_share=ne / n, kr_day_ratio=kd / max(ud, 1e-12), kr_week_ratio=kw / max(uw, 1e-12),
                max_day_ratio=n / max(ud, 1e-12), n_t=n)


@torch.no_grad()
def analyse(E, H, L, first, last, data, krexp):
    sl = slice(first, last + 1)
    y = E[sl, :H]; Es = E[:last + 1]
    out = dict(n=last - first + 1)
    corrs = {'win_row': ec.stream(Es, H, first, H, True, 'row', L)[0]}
    if krexp:
        corrs['krexp_row'] = wc.stream_ctrl(Es, H, first, H, 'row', L, 'krexp')[0]
    b_lead = y.square().mean((0, 2)); b_t = y.square().mean((1, 2)); b_var = y.square().mean((0, 1))
    out['base'] = float(y.square().mean()); out['lead_base'] = b_lead.tolist()
    dd = DAY[data]; nd = y.shape[0] // dd
    for k, c in corrs.items():
        r = y - c[sl]
        out[k] = float(r.square().mean()); out[f'lead_{k}'] = r.square().mean((0, 2)).tolist()
        g = (b_t - r.square().mean((1, 2)))[:nd * dd].view(nd, dd).sum(1)
        net = float(g.sum()); gs = g.sort(descending=True).values
        bd = b_t[:nd * dd].view(nd, dd).sum(1)
        out[f'day_{k}'] = dict(n_days=nd, frac_improved=float((g > 0).float().mean()),
                               top10_share=float(gs[:max(1, nd // 10)].sum() / net) if net > 0 else None,
                               corr_base_gain=float(torch.corrcoef(torch.stack([bd, g]))[0, 1]))
        out[f'var_{k}'] = ((r.square().mean((0, 1)) / b_var.clamp_min(1e-12) - 1) * 100).tolist()
    Fr, _ = ec.row_feat(E, H); Fc, _ = ec.ced(E, H)
    Y = y.permute(0, 2, 1).float()
    out['oracle_row'], W = oracle(Fr[sl].permute(0, 2, 1).float(), Y)
    out['oracle_ced'], _ = oracle(Fc[sl].permute(0, 2, 1).float(), Y)
    out['oracle_bias'] = float(1 - (Y - Y.mean(0, keepdim=True)).square().sum() / Y.square().sum())
    out['W'] = w_struct(W)
    out['lag'] = lag_stats(E, H, first, last, L, DAY[data], WEEK[data])
    return out


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W', 'T'])
    ap.add_argument('--base', required=True, choices=['a1', 'tefl', 'a0'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='720,1440'); ap.add_argument('--H', default='96')
    ap.add_argument('--out', default='patchmem/logs/server/win_ck'); ap.add_argument('--smoke', action='store_true')
    a_ = ap.parse_args()
    if a_.smoke:
        a_.seeds, a_.L = a_.seeds.split(',')[0], '720'
    Hs = [int(v) for v in a_.H.split(',')]
    if a_.base == 'tefl' and wc.TEFL[a_.data][2]:
        Hs = [h for h in Hs if h == wc.TEFL[a_.data][2]]
    cf = CFG[a_.data]
    for k in [k for k in os.environ if k.startswith(('R1_TEFL', 'R1_ICL', 'R1_OBS', 'R1_MEM', 'R1_PATHRES', 'R1_HMAX'))]:
        os.environ.pop(k)
    if a_.base == 'tefl':
        os.environ['R1_TEFL'] = 'ms'; os.environ['R1_HMAX'] = str(wc.TEFL[a_.data][1])
    from patchmem.exp_r1 import Exp_R1
    import hashlib, subprocess
    md5 = hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:10]
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print(f'[version] gain_src.py md5 {md5} | HEAD {head} | data {a_.data} base {a_.base}', flush=True)
    res = {}
    for seed in [int(s) for s in a_.seeds.split(',')]:
        ex = None
        if a_.base != 'a0':
            des = ec.A1[a_.data].format(s=seed) if a_.base == 'a1' else wc.TEFL[a_.data][0].format(s=seed)
            ck = cf['r1ck'].replace(cf['r1des'], des)
            ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
            ex = Exp_R1(ar)
            ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
            ex.model.eval(); ex._fix_eval_order()
        else:
            ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des x --seed {seed}').split())
        for L in [int(v) for v in a_.L.split(',')]:
            X = split_x(ar, L + ec.WARM, 'test')
            n0 = len(X) - L - min(Hs) + 1
            E_all = ec.rollout_errors(ex.model, X, L, n0) if ex is not None else None
            for H in Hs:
                t0 = time.time()
                E = E_all
                if E is None:
                    E, src = wc.a0_errors(a_.data, L, H, seed, X, n0)
                    if E is None:
                        print(f'  s{seed} L={L} H={H} | skip: {src}', flush=True); continue
                first, last = ec.WARM, len(X) - H - L
                if a_.smoke:
                    last = first + 299
                o = analyse(E, H, L, first, last, a_.data, krexp=(a_.data == 'T'))
                res[f's{seed}|{L}|{H}'] = o
                ks = [k for k in ('win_row', 'krexp_row') if k in o]
                q = H // 4
                print(f'  s{seed} L={L:4d} H={H:3d} | base {o["base"]:.4f} | ' + ' | '.join(
                    f'{k} {(o[k] / o["base"] - 1) * 100:+.1f}% (lead q1 {(sum(o["lead_" + k][:q]) / sum(o["lead_base"][:q]) - 1) * 100:+.1f}% '
                    f'q4 {(sum(o["lead_" + k][-q:]) / sum(o["lead_base"][-q:]) - 1) * 100:+.1f}%; days improved {o["day_" + k]["frac_improved"]:.2f}, '
                    f'top10% days {o["day_" + k]["top10_share"] if o["day_" + k]["top10_share"] is None else round(o["day_" + k]["top10_share"], 2)} of net, '
                    f'corr(base,gain) {o["day_" + k]["corr_base_gain"]:+.2f}; vars improved {sum(v < 0 for v in o["var_" + k])}/{len(o["var_" + k])})'
                    for k in ks) + f' | oracle row {o["oracle_row"]:.3f} ced {o["oracle_ced"]:.3f} bias {o["oracle_bias"]:.3f} | W fresh10 '
                    f'{o["W"]["fresh10"]:.2f} diag {o["W"]["diag"]:.2f} top1 {o["W"]["top1"]:.2f} | lag ' + (
                    'n/a' if o['lag'] is None else f'soft1 day x{o["lag"]["day_ratio"]:.2f} week x{o["lag"]["week_ratio"]:.2f} eff {o["lag"]["eff_share"]:.2f} | krexp day '
                    f'x{o["lag"]["kr_day_ratio"]:.2f} week x{o["lag"]["kr_week_ratio"]:.2f} (max day x{o["lag"]["max_day_ratio"]:.1f})')
                    + f' | {time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(f'{a_.out}/gain_src_{a_.data}_{a_.base}{"_smoke" if a_.smoke else ""}.json', 'w'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
