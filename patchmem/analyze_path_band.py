"""PAT-315 Step 1 addendum: split the alignment term <r_inc, d> of each correction by frequency band (Parseval cross-spectrum along the real time axis, fixed lead) and report per-lead-range gains.  Reads eval_sr --mech npz."""
import numpy as np, glob
B={'<20h':(0,20),'daily 20-28h':(20,28),'28-140h':(28,140),'weekly 140-196h':(140,196),'>196h':(196,1e9)}
HARM=[24/k for k in range(1,13)]+[168/k for k in range(1,7)]           # daily + weekly harmonics (periods in h)
def harm_mask(per, tol=0.04):
    return np.any([np.abs(per/hp-1)<=tol for hp in HARM],axis=0)
for p in sorted(glob.glob('patchmem/logs/server/err_sr_mech/T_H192_L720_s*_mech.npz')):
    z=np.load(p); LH=z['leads'].tolist(); print('==',p.split('/')[-1])
    r=z['r_inc'].astype(np.float64); r=r-r.mean(0)
    T=r.shape[0]; f=np.fft.rfftfreq(T,1.0); per=np.where(f>0,1/np.maximum(f,1e-12),np.inf)
    R=np.fft.rfft(r,axis=0)
    for m in ('s1_P','s1_GF','S_e'):
        d=z[f'd_{m}'].astype(np.float64); d=d-d.mean(0); Dm=np.fft.rfft(d,axis=0)
        cr=(R*np.conj(Dm)).real; cr[1:]*=2   # cross-spectrum, sums to T*<r,d> (approx, ignoring Nyquist)
        rows=[]
        for i,h in enumerate(LH):
            tot=cr[:,:,i].sum()
            sh={k:cr[(per>=a)&(per<b),:,i].sum()/tot for k,(a,b) in B.items()}
            hm_=harm_mask(per); sh['HARMONIC(24/k,168/k +-4%)']=cr[hm_,:,i].sum()/tot; sh['non-harmonic']=cr[~hm_,:,i].sum()/tot
            g=(2*(r[:,:,i]*d[:,:,i]).sum()-(d[:,:,i]**2).sum())/(r[:,:,i]**2).sum()*100
            rows.append(f"h{h}: gain {g:+.2f}% | "+' '.join(f"{k} {v:+.2f}" for k,v in sh.items()))
        print(' ',m); [print('    ',x) for x in rows]
    lead_rr=z['rr_inc'].sum(0)
    for m in ('s1_P','s1_GF'):
        lg=100*(2*z[f'{m}_rd_inc'].sum(0)-z[f'{m}_dd_inc'].sum(0))/lead_rr
        print(f'  per-lead gain {m}: h0-23 {lg[:24].mean():.2f} h24-95 {lg[24:96].mean():.2f} h96-191 {lg[96:].mean():.2f}')
