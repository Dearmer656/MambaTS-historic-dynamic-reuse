"""PAT-304 diagnostic: per-origin gradient geometry of the A1 multi-origin loss (same data / model / perm law as Exp_A1).
Env: GRAD_CKPTS = ';'-separated  name=path  (path 'init' = freshly initialised model), GRAD_NB = number of train batches, GRAD_OUT = json path."""
import os, sys, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.exp_a1 import Exp_A1
from patchmem.tm_model import TMMambaTS, patch_targets


def group_of(n):
    if n.startswith('G.'): return 'G'
    if 'value_embedding' in n: return 'embed'
    if 'encoder.layers' in n: return 'mixer'
    return 'other'


def naive_mse(bx, tgt, P):
    out_m, out_l = [], []
    for j in range(tgt.shape[1]):
        h = bx[:, :(j + 1) * P]
        out_m.append((tgt[:, j] - h.mean(1, keepdim=True)).square().mean().item())
        out_l.append((tgt[:, j] - h[:, -1:]).square().mean().item())
    return torch.tensor(out_m), torch.tensor(out_l)


class Exp_Grad(Exp_A1):
    def fwd(self, bx, perm):
        return TMMambaTS.forward(self.model, bx, perm)

    @torch.no_grad()
    def per_origin_mse(self, loader, nb=None):
        a = self.args; H, P = a.pred_len, a.patch_len; self.model.eval()
        se = torch.zeros(a.seq_len // P); nm = torch.zeros_like(se); nl = torch.zeros_like(se); n = 0
        for i, (bx, by, _, _) in enumerate(loader):
            if nb is not None and i >= nb: break
            bx = bx.float().to(self.device); yf = by.float().to(self.device)[:, -H:]
            perm = self.model.eval_perm(bx.shape[0], bx.device)
            pred = self.fwd(bx, perm)['pred']; tgt = patch_targets(bx, yf, P, H)
            se += (pred - tgt).square().mean((0, 2, 3)).cpu(); m_, l_ = naive_mse(bx, tgt, P); nm += m_; nl += l_; n += 1
        return (se / n).tolist(), (nm / n).tolist(), (nl / n).tolist()

    def analyse(self, name, loader, vloader, NB):
        a = self.args; H, P = a.pred_len, a.patch_len; K = a.seq_len // P; dev = self.device
        m = self.model; m.eval()
        params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        sizes = [p.numel() for _, p in params]; Ptot = sum(sizes)
        grp = torch.cat([torch.full((s,), ['G', 'embed', 'mixer', 'other'].index(group_of(n)), dtype=torch.long) for (n, _), s in zip(params, sizes)]).to(dev)
        groups = {'all': torch.ones(Ptot, dtype=torch.bool, device=dev), **{g: grp == i for i, g in enumerate(['G', 'embed', 'mixer'])}}
        sumA = {g: torch.zeros(K, int(mk.sum()), device=dev) for g, mk in groups.items()}; sumB = {g: torch.zeros_like(v) for g, v in sumA.items()}
        gram = {g: torch.zeros(K, K, device=dev, dtype=torch.float64) for g in groups}; cosb = {g: torch.zeros(K, K, device=dev, dtype=torch.float64) for g in groups}
        neg = {g: torch.zeros(K, device=dev) for g in groups}
        se = torch.zeros(K); n = 0
        for i, (bx, by, _, _) in enumerate(loader):
            if i >= NB: break
            torch.manual_seed(1000 + i)
            bx = bx.float().to(dev); yf = by.float().to(dev)[:, -H:]
            perm = m.train_perm(bx.shape[0], dev)
            pred = self.fwd(bx, perm)['pred']; tgt = patch_targets(bx, yf, P, H)
            lj = (pred - tgt).square().mean((0, 2, 3)); se += lj.detach().cpu(); n += 1
            Gm = torch.zeros(K, Ptot, device=dev)
            for j in range(K):
                gs = torch.autograd.grad(lj[j], [p for _, p in params], retain_graph=(j < K - 1), allow_unused=True)
                Gm[j] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, (_, p) in zip(gs, params)])
            for g, mk in groups.items():
                Gg = Gm[:, mk]; (sumA if i % 2 == 0 else sumB)[g] += Gg
                gr = (Gg @ Gg.T).double(); gram[g] += gr
                d = gr.diag().clamp_min(1e-30).sqrt(); cb = gr / (d[:, None] * d[None]); cosb[g] += cb; neg[g] += (cb[:, -1] < 0).float()
        res = {'name': name, 'NB': n, 'train_mse': (se / n).tolist()}
        for g in groups:
            tot = sumA[g] + sumB[g]; mean = tot / n
            ms = (gram[g].diag() / n).float()                                  # E ||g_j||^2
            nrm = ms.sqrt()
            snr = (mean.square().sum(1) / ms)
            hA, hB = sumA[g], sumB[g]
            split = torch.nn.functional.cosine_similarity(hA, hB, dim=1)       # reproducibility of the direction between even / odd batches
            cm = torch.nn.functional.normalize(mean, dim=1); cos_mean_last = cm @ cm[-1]
            proj = (mean @ mean[-1]) / mean[-1].square().sum()                 # component of mean grad_j along mean grad_last, in units of grad_last
            allm = mean.mean(0); cos_all_last = torch.nn.functional.cosine_similarity(allm, mean[-1], dim=0)
            res[g] = dict(rms_norm=nrm.tolist(), snr=snr.tolist(), split_half_cos=split.tolist(), cos_mean_to_last=cos_mean_last.tolist(), proj_on_last=proj.tolist(),
                          batch_cos_to_last=(cosb[g][:, -1] / n).tolist(), frac_batches_neg_to_last=(neg[g] / n).tolist(),
                          cos_equalweight_total_to_last=cos_all_last.item(), norm_equalweight_total=allm.norm().item(), norm_last_mean=mean[-1].norm().item())
        v_mse, v_nm, v_nl = self.per_origin_mse(vloader)
        res['val_mse'], res['val_naive_mean'], res['val_naive_last'] = v_mse, v_nm, v_nl
        return res

    def train(self, setting):
        train_data, train_loader = self._get_data('train'); _, vali_loader = self._get_data('val')
        NB = int(os.environ.get('GRAD_NB', 160)); out = []
        for item in os.environ['GRAD_CKPTS'].split(';'):
            name, path = item.split('=', 1)
            torch.manual_seed(self.args.seed)
            self.model = type(self.model)(self.args).float().to(self.device)
            if path != 'init': self.model.load_state_dict(torch.load(path, map_location=self.device))
            print('analysing', name, flush=True); out.append(self.analyse(name, train_loader, vali_loader, NB))
            r = out[-1]; K = len(r['train_mse']); f = lambda v: ' '.join(f'{x:8.3g}' for x in v)
            print('origin           ', ' '.join(f'{j:8d}' for j in range(K)))
            print('train mse        ', f(r['train_mse'])); print('val mse          ', f(r['val_mse'])); print('val naive mean   ', f(r['val_naive_mean'])); print('val naive last   ', f(r['val_naive_last']))
            for g in ('all', 'G', 'embed', 'mixer'):
                print(f'[{g}]'); [print(f'  {k:18s}', f(r[g][k])) for k in ('rms_norm', 'snr', 'split_half_cos', 'cos_mean_to_last', 'proj_on_last', 'batch_cos_to_last', 'frac_batches_neg_to_last')]
                print('  equal-weight total: cos to last', round(r[g]['cos_equalweight_total_to_last'], 3), '| |mean total|', round(r[g]['norm_equalweight_total'], 4), '| |mean last|', round(r[g]['norm_last_mean'], 4), flush=True)
            Path(os.environ['GRAD_OUT']).write_text(json.dumps(out))
        sys.exit(0)
