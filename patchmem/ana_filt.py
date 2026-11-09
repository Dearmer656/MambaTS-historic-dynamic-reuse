"""PAT-315: learned beta filter (wb) and fixed u of the PFw carries, across H, backbone seed and carry seed."""
import torch, numpy as np, math
R='patchmem/logs/server/err_srcx/'
d=torch.load(R+'cs0/T_H192_s3047_s1_PFw_dump.pt',map_location='cpu',weights_only=False)
print({k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in d.items()})
sd=next(v for k,v in d.items() if k in('state','state_dict','net','sd'))
print({k:tuple(v.shape) for k,v in sd.items()})
W=[];Us=[];L=[];B=[]
for H in (192,336):
  for s in (3047,2024,2025):
    for c in (0,1000):
      sd=next(v for k,v in torch.load(f'{R}cs{c}/T_H{H}_s{s}_s1_PFw_dump.pt',map_location='cpu',weights_only=False).items() if k in('state','state_dict','net','sd'))
      w=sd['wb.weight'][0].double().numpy(); W.append(w); Us.append(sd['uf'].double().numpy()); L.append((H,s,c)); B.append(float(sd['wb.bias']))
W=np.array(W);Us=np.array(Us)
Wn=W/np.linalg.norm(W,axis=1,keepdims=True)
C=Wn@Wn.T; iu=np.triu_indices(12,1)
print('filter cos: median %.3f min %.3f max %.3f'%(np.median(C[iu]),C[iu].min(),C[iu].max()))
print('same (H,s) cs0 vs cs1000 cos',[round(float(C[i,i+1]),3) for i in range(0,12,2)])
print('|w| norm',np.round(np.linalg.norm(W,axis=1),2),'bias',np.round(B,2),'sum w',np.round(W.sum(1),2))
m=Wn.mean(0); print('mean filter by lag (oldest -96 .. newest -1), every value x100:'); print(' '.join(f'{100*x:.0f}' for x in m))
F=np.abs(np.fft.rfft(Wn,axis=1))**2; F=F/F.sum(1,keepdims=True); per=[96/k if k else 0 for k in range(F.shape[1])]
top=np.argsort(-F.mean(0))[:6]; print('spectrum share by period(h):',[(round(per[k],1),round(float(F.mean(0)[k]),3)) for k in top])
print('energy share by age: last 6h %.3f last 24h %.3f 24-48h %.3f 48-96h %.3f'%tuple(float(x) for x in ((Wn[:,-6:]**2).sum(1).mean(),(Wn[:,-24:]**2).sum(1).mean(),(Wn[:,-48:-24]**2).sum(1).mean(),(Wn[:,:-48]**2).sum(1).mean())))
Un=Us/np.linalg.norm(Us,axis=1,keepdims=True); CU=np.abs(Un@Un.T); print('|cos| u: median %.3f same unit %s'%(np.median(CU[iu]),[round(float(CU[i,i+1]),3) for i in range(0,12,2)]))
p=(Un**2).reshape(12,64,2).sum(2); print('u participation ratio (of 64 modes)',np.round(1/(p**2).sum(1),1))

u0 = torch.randn(128, generator=torch.Generator().manual_seed(2)).double().numpy(); u0 /= np.linalg.norm(u0)
print('|cos(u, init)|', np.round(np.abs(Un @ u0), 3))
logit = lambda b: np.log(np.clip(b, 1e-4, 2 - 1e-4) / (2 - np.clip(b, 1e-4, 2 - 1e-4)))
def beta(f):
    return logit(torch.load(f, map_location='cpu', weights_only=False)['beta'].double().numpy())
def rep(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
def dev(a):                                                                       # minus channel x hour-of-week mean (test rows)
    n = a.shape[0]; h = np.arange(n) % 168; out = a.copy()
    for k in range(168):
        out[h == k] -= a[h == k].mean(0, keepdims=True)
    return out
for H in (192, 336):
    for s in (3047, 2024, 2025):
        a, b = beta(f'{R}cs0/T_H{H}_s{s}_s1_PFw_dump.pt'), beta(f'{R}cs1000/T_H{H}_s{s}_s1_PFw_dump.pt')
        pa, pb = beta(f'patchmem/logs/server/err_s2/T_H{H}_s{s}_s1_Pfix_dump.pt'), beta(f'patchmem/logs/server/err_gate/cs1000/T_H{H}_s{s}_s1_Pfix_dump.pt')
        v = lambda x: float(dev(x).var() / x.var())
        print(f'H{H} s{s} logit beta corr cs0 vs cs1000: PFw {rep(a, b):+.3f} (non-weekly part {rep(dev(a), dev(b)):+.3f}) Pfix {rep(pa, pb):+.3f} (non-weekly {rep(dev(pa), dev(pb)):+.3f}) | PFw vs Pfix {rep(a, pa):+.3f} / non-weekly {rep(dev(a), dev(pa)):+.3f} | non-weekly var share PFw {v(a):.2f} Pfix {v(pa):.2f}')
