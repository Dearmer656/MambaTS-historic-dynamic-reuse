"""Print per-epoch carry train loss / early-stop val MSE from eval_sr jsons and dumps."""
import json, sys, glob
import torch
for f in sorted(sys.argv[1:]):
    if f.endswith('.json'):
        r = json.load(open(f))
        items = [(k, v) for k, v in r.items() if isinstance(v, dict) and 'train_loss' in v]
        print(f'== {f}  base_sh val {r.get("base_sh", {}).get("val_mse")}')
    else:
        d = torch.load(f, map_location='cpu', weights_only=False)
        items = [(f.split('_s')[-1], d)]
        print(f'== {f}')
    for k, v in items:
        tl = v['train_loss']
        es = tl[-1] if tl and isinstance(tl[-1], dict) else None
        tr = [x for x in tl if not isinstance(x, dict)]
        print(f'  {k:10s} train ' + ' '.join(f'{x:.4f}' for x in tr))
        if es:
            print(f'  {"":10s} val   ' + ' '.join(f'{x:.5f}' for x in es['val']) + f'  best_ep {es["best_epoch"]}  test {v.get("test_mse")}')
