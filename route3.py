import numpy as np
from route import hop_widest,_comp
def run_routing3(up,predfun,flows,T,Tu=2,Trd=2,policy='max-min',H=60,margin=None,life=None):
    margin=Tu if margin is None else margin
    """up: (T,N,N) true link state. predfun(t)-> NxN matrix of lifetime estimates (nan where no observed link)."""
    st=[dict(route=None,outage=0) for _ in flows]
    M=dict(sec=0,deliv=0,breaks=0,disc=0,disc_re=0,disc_pro=0,hops=0,hopsn=0,pc_n=0,pc_hit=0,lc_n=0,lc_hit=0,kept=0,kept_fail=0,slots=0)
    HB={h:[0,0,0,0] for h in ['1','2','3','4','5+']}  # path-cov n, hit, kept n, kept fail
    cache={}
    for t in range(20,T):
        upd=(t%Tu==0)
        if upd:
            W=predfun(t)
        for k,(s,d) in enumerate(flows):
            f=st[k]
            # connectivity check (truth) - only count seconds where S-D connected at all
            r=f['route']
            if r is not None:
                ok=all(up[t,r[i],r[i+1]] for i in range(len(r)-1))
                if not ok:
                    M['breaks']+=1; f['route']=None; f['outage']=Trd
            if f['route'] is None and f['outage']<=0:
                Wt=predfun(t)
                Wm=Wt if policy!='hop' else np.where(np.isnan(Wt),np.nan,1.0)
                p=hop_widest(Wm,s,d,slack=(0 if policy=='hop' else 2))
                if p is not None:
                    f['route']=p; M['disc']+=1; M['disc_re']+=1
            elif f['route'] is not None and upd and policy!='hop':
                r=f['route']; b=min(W[r[i],r[i+1]] if not np.isnan(W[r[i],r[i+1]]) else 0 for i in range(len(r)-1))
                if b<=margin:   # predicted to (possibly) break before next update -> proactive switch
                    p=hop_widest(W,s,d)
                    if p is not None and p!=r:
                        f['route']=p; M['disc']+=1; M['disc_pro']+=1
            # --- coverage on the links actually used for routing (selection effect)
            if life is not None and upd and policy!='hop' and f['route'] is not None and t<=T-H-1:
                r=f['route']; Ls=[W[r[q],r[q+1]] for q in range(len(r)-1)]
                if not any(np.isnan(Ls)):
                    Ys=[life[t,r[q],r[q+1]] for q in range(len(r)-1)]
                    M['lc_n']+=len(Ls); M['lc_hit']+=sum(y>=l for y,l in zip(Ys,Ls))
                    Lp=min(Ls); Yp=min(Ys); M['pc_n']+=1; M['pc_hit']+=int(Yp>=Lp)
                    hb=str(len(Ls)) if len(Ls)<5 else '5+'; HB[hb][0]+=1; HB[hb][1]+=int(Yp>=Lp)
                    if Lp>margin:
                        M['kept']+=1; M['kept_fail']+=int(Yp<Tu); HB[hb][2]+=1; HB[hb][3]+=int(Yp<Tu)
            # accounting
            M['slots']+=1
            reach=_comp(up,t)[s]==_comp(up,t)[d]
            if reach:
                M['sec']+=1
                if f['route'] is not None and all(up[t,f['route'][i],f['route'][i+1]] for i in range(len(f['route'])-1)):
                    M['deliv']+=1; M['hops']+=len(f['route'])-1; M['hopsn']+=1
            f['outage']-=1
    M['hopbins']=HB
    return M
