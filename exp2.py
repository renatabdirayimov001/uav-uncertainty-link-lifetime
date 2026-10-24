import numpy as np, sys
from sim import *; from route import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
R=350.; H=60; HOLD=4
MOB='gm'
al=float(sys.argv[1]) if len(sys.argv)>1 else 0.1
def mk(seed,T,N=50,speed=(10,25),hetero=True):
    rng=np.random.default_rng(seed+100)
    if MOB=='group': P,_,_=group_trace(N=N,T=T,seed=seed)
    else: P=gm_trace(N=N,T=T,seed=seed,hetero=hetero,vmean=speed)
    D=np.linalg.norm(P[:,:,None,:]-P[:,None,:,:],axis=-1)
    life,up,valid=residual_life(D,R,H)
    import os
    STALE=os.environ.get('SCEN','base')=='stale'
    SG=np.random.default_rng(seed+7).uniform(2,20 if STALE else 10,N)
    Po=observe(P,SG,rng)
    cong=np.random.default_rng(seed+9).uniform(0,0.7 if STALE else 0.4,N)
    A=hello_age(D,R,np.random.default_rng(seed+11),cong=cong)
    return P,up,life,Po,A
def known(A,t): return A[t]<=HOLD
def ds(seed,T=1500,step=3,**kw):
    P,up,life,Po,A=mk(seed,T,**kw); N=P.shape[1]; iu=np.triu_indices(N,1); Xs=[];Ys=[]
    for t in range(20,T-H-1,step):
        m=known(A,t)[iu]; i=iu[0][m]; j=iu[1][m]
        X,_=features_age(Po,A,t,i,j,R,H); Xs.append(X); Ys.append(life[t,i,j])
    return np.vstack(Xs),np.concatenate(Ys)
Xtr,Ytr=ds(1); Xca,Yca=ds(2)
mp=HGB(max_iter=300,learning_rate=0.08).fit(Xtr,Ytr)
mq=HGB(loss='quantile',quantile=al,max_iter=300,learning_rate=0.08).fit(Xtr,Ytr)
ens=[]
for k in range(5):
    idx=np.random.default_rng(k).integers(0,len(Ytr),len(Ytr)); ens.append(HGB(max_iter=200,learning_rate=0.1,random_state=k).fit(Xtr[idx],Ytr[idx]))
n=len(Yca); q=np.quantile(mq.predict(Xca)-Yca,(1-al)*(1+1/n))
qp=np.quantile(mp.predict(Xca)-Yca,(1-al)*(1+1/n))
def est(name,X,let):
    if name in ('LET','HOP'): return let
    if name=='ML-point': return mp.predict(X)
    if name=='Ens-LB':
        pr=np.array([m.predict(X) for m in ens]); return pr.mean(0)-1.2816*pr.std(0)
    if name=='CP-point': return mp.predict(X)-qp
    if name=='CQR-LB': return mq.predict(X)-q
def coverage(seed=3):
    Xte,Yte=ds(seed)
    for nm in ['LET','ML-point','Ens-LB','CP-point','CQR-LB']:
        L=np.clip(est(nm,Xte,Xte[:,3]),0,H)
        print(f'{nm:9s} cov={np.mean(Yte>=L):.3f} meanL={L.mean():5.1f} meanY={Yte.mean():5.1f}')
    a=Xte[:,4]
    for lo,hi in [(0,0),(1,1),(2,4)]:
        m=(a>=lo)&(a<=hi); L=np.clip(est('CQR-LB',Xte[m],Xte[m,3]),0,H); Lt=Xte[m,3]
        print(f' age {lo}-{hi}: n={m.sum()} meanY={Yte[m].mean():.1f} LETcov={np.mean(Yte[m]>=Lt):.2f} CQRcov={np.mean(Yte[m]>=L):.2f} CQRmeanL={L.mean():.1f}')
def evaluate(test_seed,policies,T=900,nflows=15,speed=(10,25),Tu=2):
    P,up,life,Po,A=mk(test_seed,T,speed=speed)
    global LIFE; LIFE=life; N=P.shape[1]; iu=np.triu_indices(N,1)
    rng=np.random.default_rng(test_seed+5); flows=[]
    while len(flows)<nflows:
        s,d=rng.choice(N,2,replace=False)
        if np.linalg.norm(P[10,s]-P[10,d])>600: flows.append((s,d))
    feats={}
    def F(t):
        if t not in feats:
            m=known(A,t)[iu]; i=iu[0][m]; j=iu[1][m]; X,let=features_age(Po,A,t,i,j,R,H); feats[t]=(i,j,X,let)
        return feats[t]
    res={}
    for lab,(nm,margin) in policies.items():
        cache={}
        def pf(t,nm=nm,cache=cache):
            if t not in cache:
                i,j,X,let=F(t); w=np.clip(LIFE[t,i,j] if nm=='ORACLE' else est(nm,X,let),0,H)
                W=np.full((N,N),np.nan); W[i,j]=w; W[j,i]=w; cache[t]=W
            return cache[t]
        res[lab]=run_routing(up,pf,flows,T,policy='hop' if nm=='HOP' else 'max-min',Tu=Tu,margin=margin)
        res[lab]['mins']=nflows*(T-20)/60
    return res
if __name__=='__main__':
    coverage()
    pol={'HOP':('HOP',2)}
    for m in [2,4,6]:
        pol[f'LET m={m}']=('LET',m); pol[f'ML m={m}']=('ML-point',m)
    pol['Ens m=2']=('Ens-LB',2); pol['CP-point m=2']=('CP-point',2); pol['CQR m=2']=('CQR-LB',2)
    agg={}
    for sd in [7,8,9]:
        r=evaluate(sd,pol)
        for k,M in r.items():
            agg.setdefault(k,[]).append([M['deliv']/M['sec'],M['breaks']/M['mins'],M['disc']/M['mins'],M['hops']/max(1,M['hopsn'])])
        print('seed',sd,flush=True)
    for k,v in agg.items():
        v=np.array(v); print(f"{k:12s} PDR={v[:,0].mean():.3f}±{v[:,0].std():.3f} breaks={v[:,1].mean():.2f} disc={v[:,2].mean():.2f} hops={v[:,3].mean():.2f}",flush=True)
