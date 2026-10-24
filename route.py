import numpy as np, time
from sim import *
def hop_widest(W, s, d, slack=2, Hmax=10):
    """W: NxN weights (nan = no link). maximise bottleneck among paths with hops <= hmin+slack."""
    N=W.shape[0]; A=~np.isnan(W)
    # BFS hops
    dist=np.full(N,99); dist[s]=0; fr=[s]
    while fr:
        nf=[]
        for u in fr:
            for v in np.where(A[u])[0]:
                if dist[v]==99: dist[v]=dist[u]+1; nf.append(v)
        fr=nf
    if dist[d]==99: return None
    hl=min(dist[d]+slack,Hmax)
    best=np.full((hl+1,N),-np.inf); best[0,s]=np.inf; pred=np.full((hl+1,N),-1)
    Wf=np.where(A,W,-np.inf)
    for h in range(1,hl+1):
        cand=np.minimum(best[h-1][:,None],Wf)   # u->v
        pred[h]=np.argmax(cand,axis=0); best[h]=cand[pred[h],np.arange(N)]
    # choose smallest h whose bottleneck >= 0.95*overall best (prefer fewer hops)
    tot=best[1:,d].max()
    h=1+np.where(best[1:,d]>=tot-1e-9)[0][0] if np.isfinite(tot) else None
    if h is None: return None
    path=[d]; v=d
    for hh in range(h,0,-1):
        v=pred[hh,v]; path.append(v)
    path=path[::-1]
    if path[0]!=s or len(set(path))!=len(path): return None
    return path
def run_routing(up,predfun,flows,T,Tu=2,Trd=2,policy='max-min',H=60,margin=None):
    margin=Tu if margin is None else margin
    """up: (T,N,N) true link state. predfun(t)-> NxN matrix of lifetime estimates (nan where no observed link)."""
    st=[dict(route=None,outage=0) for _ in flows]
    M=dict(sec=0,deliv=0,breaks=0,disc=0,hops=0,hopsn=0)
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
                    f['route']=p; M['disc']+=1
            elif f['route'] is not None and upd and policy!='hop':
                r=f['route']; b=min(W[r[i],r[i+1]] if not np.isnan(W[r[i],r[i+1]]) else 0 for i in range(len(r)-1))
                if b<=margin:   # predicted to (possibly) break before next update -> proactive switch
                    p=hop_widest(W,s,d)
                    if p is not None and p!=r:
                        f['route']=p; M['disc']+=1
            # accounting
            reach=_comp(up,t)[s]==_comp(up,t)[d]
            if reach:
                M['sec']+=1
                if f['route'] is not None and all(up[t,f['route'][i],f['route'][i+1]] for i in range(len(f['route'])-1)):
                    M['deliv']+=1; M['hops']+=len(f['route'])-1; M['hopsn']+=1
            f['outage']-=1
    return M
_rc={}
def _reach(A,s,d):
    N=A.shape[0]; seen=np.zeros(N,bool); seen[s]=True; fr=[s]
    while fr:
        nf=[]
        for u in fr:
            for v in np.where(A[u]&~seen)[0]:
                seen[v]=True; nf.append(v)
        fr=nf
    return seen[d]

from scipy.sparse.csgraph import connected_components
_cc={}
def _comp(up,t):
    k=(id(up),t)
    if k not in _cc: _cc[k]=connected_components(up[t],directed=False)[1]
    return _cc[k]
