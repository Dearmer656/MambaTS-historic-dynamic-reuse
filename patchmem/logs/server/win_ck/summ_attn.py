import json,glob,os,collections,sys
A=['lin1','krlin','exp1','soft1','krexp']
for f in sorted(glob.glob(os.path.dirname(os.path.abspath(__file__))+'/win_attn_*.json')):
    if 'smoke' in f: continue
    d=json.load(open(f)); print('==',os.path.basename(f))
    g=collections.defaultdict(dict)
    for k,v in d.items():
        s,L,H=k.split('|'); g[(int(L),int(H))][s]=v
    for (L,H) in sorted(g):
        r=g[(L,H)]; ss=sorted(r)
        if all(abs(r[s]['win_row']-r[s]['base'])<1e-9 for s in ss): continue
        p=lambda k: '/'.join(f"{100*(r[s][k]/r[s]['base']-1):+.1f}" for s in ss)
        out=f"L{L}/H{H} n{len(ss)} base {'/'.join(f'{r[s][chr(98)+chr(97)+chr(115)+chr(101)]:.3f}' for s in ss)} | win_row {p('win_row')}"
        for ft in ('row','ced'):
            out+=f" || {ft}: "+' '.join(f"{a} {p(a+'_'+ft)}" for a in A)
            fl=lambda x,y: 'Y' if all(r[s][x+'_'+ft]<r[s][y+'_'+ft] for s in ss) else 'n'
            out+=f" [k1 {fl('exp1','lin1')} kr {fl('krexp','krlin')} sm {fl('soft1','exp1')}]"
        print(out)
