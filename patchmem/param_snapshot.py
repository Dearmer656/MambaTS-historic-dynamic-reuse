"""PAT-304 (comment e18ad84b sec. 5): one statistics snapshot of every parameter and buffer, printed once per model-load event
(event='init' after construction, 'checkpoint_loaded' after every load_state_dict), never per batch or per forward.
Console: header, one row per named parameter, per-module subtotals, buffers section. Files: <out_dir>/<stamp>_<event>.jsonl (one JSON
record per parameter / buffer / module plus a header record) and .csv (parameter and buffer rows). Statistics are computed under
no_grad in float64 on the tensor's device; empty and non-finite tensors are handled (stats over the finite entries, counts reported)."""
import csv, datetime, json, os, subprocess
import torch

ENV_KEYS = ('R1_MEM', 'R1_CORR', 'R1_MEM_GROUP', 'R1_LAMBDA_HIST', 'R1_LAMBDA_CORR', 'R1_MEM_VALUE_DETACH', 'R1_MEM_RETRIEVE',
            'R1_CORR_SHUFFLE', 'R1_RO_LN', 'R1_LMIN', 'R1_LMAX', 'R1_HMIN', 'R1_HMAX', 'R1_EVAL_PERM')
COLS = ('name', 'shape', 'numel', 'requires_grad', 'dtype', 'mean', 'std', 'min', 'max', 'absmax', 'rms', 'l2', 'zero_fraction', 'nonfinite_count')


def _git(*args):
    try:
        return subprocess.run(['git', *args], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:
        return None


@torch.no_grad()
def tensor_stats(name, t, requires_grad=None):
    x = t.detach()
    row = dict(name=name, shape=list(x.shape), numel=x.numel(), requires_grad=requires_grad, dtype=str(x.dtype).replace('torch.', ''))
    if not (x.is_floating_point() or x.is_complex()):
        x = x.to(torch.float64)
    x = x.to(torch.float64).flatten()
    fin = torch.isfinite(x)
    nf = int((~fin).sum().item())
    xf = x[fin]
    row['nonfinite_count'] = nf
    if xf.numel() == 0:
        row.update({k: None for k in ('mean', 'std', 'min', 'max', 'absmax', 'rms', 'l2', 'zero_fraction')})
        return row, 0.0, 0
    sq = xf.square().sum().item()
    row.update(mean=xf.mean().item(), std=xf.std(unbiased=False).item(), min=xf.min().item(), max=xf.max().item(),
               absmax=xf.abs().max().item(), rms=(sq / xf.numel()) ** 0.5, l2=sq ** 0.5,
               zero_fraction=(xf == 0).double().mean().item())
    return row, sq, xf.numel()


def _module_of(name):
    """Group key: 'base.encoder.layers.3' for encoder layers, else the first two name components."""
    parts = name.split('.')
    if 'layers' in parts:
        i = parts.index('layers')
        return '.'.join(parts[:i + 2])
    return '.'.join(parts[:2]) if len(parts) > 2 else parts[0]


def _fmt(v, w=11):
    if v is None:
        return 'n/a'.rjust(w)
    if isinstance(v, float):
        return f'{v:{w}.4e}'
    return str(v).rjust(w)


@torch.no_grad()
def log_model_parameter_stats(model, event, out_dir, meta=None, extra_param_prefixes=('corr_net.', 'mem_net.'), print_fn=print):
    """event: 'init' | 'checkpoint_loaded'. meta: dict (checkpoint path / epoch / step / seed / setting ...). Returns the file stem."""
    meta = dict(meta or {})
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now().astimezone()
    stem = os.path.join(out_dir, f'{now.strftime("%Y%m%dT%H%M%S%f")}_{event}')
    p0 = next(model.parameters(), None)
    params = list(model.named_parameters())                                             # named_parameters de-duplicates shared tensors
    pids = {id(p) for _, p in params}
    buffers = [(n, b) for n, b in model.named_buffers() if id(b) not in pids]
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    extra = sum(p.numel() for n, p in params if n.startswith(tuple(extra_param_prefixes)))
    header = dict(record='header', timestamp=now.isoformat(), event=event, commit=_git('rev-parse', 'HEAD'),
                  branch=_git('rev-parse', '--abbrev-ref', 'HEAD'), dirty=bool(_git('status', '--porcelain', '--untracked-files=no')),
                  model_type=type(model).__name__, dtype=str(p0.dtype) if p0 is not None else None,
                  device=str(p0.device) if p0 is not None else None, total_params=total, trainable_params=trainable,
                  frozen_params=total - trainable, extra_params_vs_R1=extra, n_param_tensors=len(params), n_buffers=len(buffers),
                  env={k: os.environ.get(k) for k in ENV_KEYS}, **meta)
    prow, brow, mods = [], [], {}
    for n, p in params:
        row, sq, cnt = tensor_stats(n, p, p.requires_grad)
        prow.append(row)
        g = mods.setdefault(_module_of(n), dict(numel=0, sq=0.0, finite=0, nonfinite=0))
        g['numel'] += p.numel(); g['sq'] += sq; g['finite'] += cnt; g['nonfinite'] += row['nonfinite_count']
    for n, b in buffers:
        brow.append(tensor_stats(n, b)[0])
    mrow = [dict(record='module', module=k, numel=v['numel'], l2=v['sq'] ** 0.5, rms=(v['sq'] / v['finite']) ** 0.5 if v['finite'] else None,
                 nonfinite_count=v['nonfinite']) for k, v in mods.items()]

    with open(stem + '.jsonl', 'w') as f:
        f.write(json.dumps(header) + '\n')
        for r in prow:
            f.write(json.dumps(dict(record='param', **r)) + '\n')
        for r in mrow:
            f.write(json.dumps(r) + '\n')
        for r in brow:
            f.write(json.dumps(dict(record='buffer', **r)) + '\n')
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('kind',) + COLS)
        for kind, rows in (('param', prow), ('buffer', brow)):
            for r in rows:
                w.writerow((kind,) + tuple('x'.join(map(str, r['shape'])) if c == 'shape' else r[c] for c in COLS))

    out = [f'===== [param snapshot] event={event} | {header["timestamp"]} | commit {header["commit"]} ({header["branch"]}'
           f'{", dirty" if header["dirty"] else ""}) | {header["model_type"]} {header["dtype"]} {header["device"]}',
           '  meta: ' + json.dumps(meta, default=str),
           '  env: ' + ' '.join(f'{k}={v}' for k, v in header['env'].items() if v is not None),
           f'  params: total {total} | trainable {trainable} | frozen {total - trainable} | extra vs R1 (corr_net+mem_net) {extra} | '
           f'{len(params)} tensors, {len(buffers)} buffers | files {stem}.jsonl / .csv',
           '  ' + ' '.join(c.rjust(11) if i else c.ljust(48) for i, c in enumerate(('name', 'numel', 'grad', 'mean', 'std', 'min', 'max',
                                                                                  'absmax', 'rms', 'l2', 'zero_frac', 'nonfinite')))]
    for r in prow:
        out.append('  ' + r['name'].ljust(48) + ''.join(_fmt(r[k]) for k in ('numel', 'requires_grad', 'mean', 'std', 'min', 'max', 'absmax',
                                                                             'rms', 'l2', 'zero_fraction', 'nonfinite_count'))
                   + f'  {tuple(r["shape"])}')
    out.append('  -- module subtotals:')
    for r in mrow:
        rms = 'n/a' if r['rms'] is None else f'{r["rms"]:.4e}'
        bad = f' NONFINITE={r["nonfinite_count"]}' if r['nonfinite_count'] else ''
        out.append(f'  {r["module"].ljust(48)} numel {r["numel"]:>9} | l2 {r["l2"]:.4e} | rms {rms}{bad}')
    out.append(f'  -- buffers ({len(buffers)}; not parameters):')
    for r in brow:
        out.append('  ' + r['name'].ljust(48) + ''.join(_fmt(r[k]) for k in ('numel', 'mean', 'std', 'min', 'max', 'zero_fraction', 'nonfinite_count'))
                   + f'  {tuple(r["shape"])} {r["dtype"]}')
    out.append(f'===== [param snapshot end] event={event}')
    print_fn('\n'.join(out), flush=True)
    return stem
