import json,glob,os,collections
for f in sorted(glob.glob(os.path.dirname(os.path.abspath(__file__))+'/gain_src_*.json')):
    if 'smoke' in f: continue
    d=json.load(open(f)); print('==',os.path.basename(f))
    g=collections.defaultdict(list)
    for k,v in d.items():
        s,L,H=k.split('|'); g[(int(L),int(H))].append(v)
    for (L,H),vs in sorted(g.items()):
        m=lambda fn: '/'.join(fn(v) for v in vs)
        q=H//4
        line=f"L{L}/H{H} n{len(vs)} base {m(lambda v:f'{v[chr(98)+chr(97)+chr(115)+chr(101)]:.3f}')}"
        for k in ('win_row','krexp_row'):
            if k not in vs[0]: continue
            line+=(f" | {k} {m(lambda v:f'{(v[k]/v[chr(98)+chr(97)+chr(115)+chr(101)]-1)*100:+.1f}')}%"
                   f" leadQ1 {m(lambda v:f'{(sum(v[chr(108)+chr(101)+chr(97)+chr(100)+chr(95)+k][:q])/sum(v[chr(108)+chr(101)+chr(97)+chr(100)+chr(95)+chr(98)+chr(97)+chr(115)+chr(101)][:q])-1)*100:+.0f}')}"
                   f" Q4 {m(lambda v:f'{(sum(v[chr(108)+chr(101)+chr(97)+chr(100)+chr(95)+k][-q:])/sum(v[chr(108)+chr(101)+chr(97)+chr(100)+chr(95)+chr(98)+chr(97)+chr(115)+chr(101)][-q:])-1)*100:+.0f}')}"
                   f" daysImp {m(lambda v:f'{v[chr(100)+chr(97)+chr(121)+chr(95)+k][chr(102)+chr(114)+chr(97)+chr(99)+chr(95)+chr(105)+chr(109)+chr(112)+chr(114)+chr(111)+chr(118)+chr(101)+chr(100)]:.2f}')}"
                   f" varsImp {m(lambda v:str(sum(x<0 for x in v[chr(118)+chr(97)+chr(114)+chr(95)+k]))+chr(47)+str(len(v[chr(118)+chr(97)+chr(114)+chr(95)+k])))}")
        line+=f" | oracle row {m(lambda v:f'{v[chr(111)+chr(114)+chr(97)+chr(99)+chr(108)+chr(101)+chr(95)+chr(114)+chr(111)+chr(119)]:.2f}')} ced {m(lambda v:f'{v[chr(111)+chr(114)+chr(97)+chr(99)+chr(108)+chr(101)+chr(95)+chr(99)+chr(101)+chr(100)]:.2f}')} bias {m(lambda v:f'{v[chr(111)+chr(114)+chr(97)+chr(99)+chr(108)+chr(101)+chr(95)+chr(98)+chr(105)+chr(97)+chr(115)]:.2f}')}"
        line+=f" | W fresh10 {m(lambda v:f'{v[chr(87)][chr(102)+chr(114)+chr(101)+chr(115)+chr(104)+chr(49)+chr(48)]:.2f}')} top1 {m(lambda v:f'{v[chr(87)][chr(116)+chr(111)+chr(112)+chr(49)]:.2f}')}"
        line+=f" | lag soft day {m(lambda v:f'{v[chr(108)+chr(97)+chr(103)][chr(100)+chr(97)+chr(121)+chr(95)+chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.2f}')} kr day {m(lambda v:f'{v[chr(108)+chr(97)+chr(103)][chr(107)+chr(114)+chr(95)+chr(100)+chr(97)+chr(121)+chr(95)+chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.2f}')} kr week {m(lambda v:f'{v[chr(108)+chr(97)+chr(103)][chr(107)+chr(114)+chr(95)+chr(119)+chr(101)+chr(101)+chr(107)+chr(95)+chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.2f}')}"
        print(line)
