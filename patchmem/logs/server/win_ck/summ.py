import json,glob,os,collections
C=['win_row','win_row_a1','bias','persist','scalar','shuffle','ridge_mean']
for f in sorted(glob.glob(os.path.dirname(__file__)+'/win_ck_*.json')):
    if 'smoke' in f: continue
    d=json.load(open(f)); print('==',os.path.basename(f))
    g=collections.defaultdict(dict)
    for k,v in d.items():
        s,L,H=k.split('|'); g[(int(L),int(H))][s]=v
    for (L,H) in sorted(g):
        r=g[(L,H)]; ss=sorted(r)
        base=' '.join(f"{r[s]['base']:.4f}" for s in ss)
        line=f"L{L}/H{H} base {base} |"
        for c in C: line+=f" {c} "+'/'.join(f"{100*(r[s][c]/r[s]['base']-1):+.1f}" for s in ss)
        act=min(r[s].get('win_row_active',1) for s in ss)
        beat=all(r[s]['win_row']<min(r[s][c] for c in C[2:]) for s in ss)
        gain=all(r[s]['win_row']<r[s]['base'] for s in ss)
        print(line+f" | act {act:.2f} gain3/3 {gain} beatAll3/3 {beat}")
