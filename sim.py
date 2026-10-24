import numpy as np
# ---------------- mobility: 3-D Gauss-Markov swarm ----------------
def gm_trace(N=50,T=1200,dt=1.0,box=(1500,1500,200),zmin=100,vmean=(10,25),a=0.85,sig=4.0,seed=0,hetero=False,pman=0.02,mix=(0.4,0.4,0.2)):
    rng=np.random.default_rng(seed)
    if hetero:
        cls=rng.choice(3,N,p=list(mix))
        a=np.array([0.95,0.85,0.6])[cls][:,None]; sig=np.array([1.0,4.0,8.0])[cls][:,None]; pman=np.array([0.005,0.02,0.08])[cls]
    else:
        a=np.full((N,1),a); sig=np.full((N,1),sig); pman=np.full(N,pman)
    L=np.array(box,float)
    p=np.c_[rng.uniform(0,L[0],N),rng.uniform(0,L[1],N),rng.uniform(zmin,zmin+L[2],N)]
    sp=rng.uniform(*vmean,N); th=rng.uniform(0,2*np.pi,N)
    vbar=np.c_[sp*np.cos(th),sp*np.sin(th),np.zeros(N)]
    v=vbar.copy()
    P=np.zeros((T,N,3))
    lo=np.array([0,0,zmin]); hi=lo+L
    for t in range(T):
        P[t]=p
        # occasional mean-heading changes (manoeuvres)
        ch=rng.random(N)<pman
        th[ch]+=rng.normal(0,np.pi/2,ch.sum())
        vbar[:,0]=sp*np.cos(th); vbar[:,1]=sp*np.sin(th)
        v=a*v+(1-a)*vbar+np.sqrt(1-a*a)*sig*rng.normal(size=(N,3))*np.array([1,1,0.3])
        p=p+v*dt
        for k in range(3):
            m=p[:,k]<lo[k]; p[m,k]=2*lo[k]-p[m,k]; v[m,k]*=-1
            m=p[:,k]>hi[k]; p[m,k]=2*hi[k]-p[m,k]; v[m,k]*=-1
        # reflect mean heading at x/y walls
        bx=(p[:,0]<50)|(p[:,0]>L[0]-50); th[bx]=np.pi-th[bx]
        by=(p[:,1]<50)|(p[:,1]>L[1]-50); th[by]=-th[by]
    return P

def residual_life(D,R,H):
    """D: (T,N,N) true distances. returns lifetime (T,N,N) capped at H for links up at t."""
    T=D.shape[0]; up=D<=R
    life=np.full(D.shape,np.nan)
    nxt=np.full(D.shape[1:],H,float)  # time until first down, computed backwards
    # backward recursion: life[t]=0 if down; else 1+life[t+1]
    cur=np.zeros(D.shape[1:])
    out=np.zeros(D.shape)
    for t in range(T-1,-1,-1):
        cur=np.where(up[t],np.minimum(cur+1,H),0.0)
        out[t]=cur
    valid=np.arange(T)[:,None,None]<=T-1-H   # need full horizon to be uncensored by trace end
    return out,up,valid

def observe(P,sig_gps,rng):
    sg=np.asarray(sig_gps,float)
    if sg.ndim==1: sg=sg[None,:,None]
    return P+rng.normal(0,1,P.shape)*sg

def features(Po,t,i,j,R,H,dt=1.0):
    """features from noisy observations at t,t-1,t-2,t-3"""
    pi=Po[t,i]; pj=Po[t,j]
    vi=(Po[t,i]-Po[t-3,i])/(3*dt); vj=(Po[t,j]-Po[t-3,j])/(3*dt)
    dp=pj-pi; dv=vj-vi
    d=np.linalg.norm(dp,axis=-1)
    vr=(dp*dv).sum(-1)/np.maximum(d,1e-6)
    sv=np.linalg.norm(dv,axis=-1)
    # Su-Lee-Gerla LET (3-D): solve |dp+dv*t|=R
    a=(dv*dv).sum(-1); b=2*(dp*dv).sum(-1); c=(dp*dp).sum(-1)-R*R
    disc=np.maximum(b*b-4*a*c,0)
    let=np.where(a>1e-9,(-b+np.sqrt(disc))/(2*np.maximum(a,1e-9)),H)
    let=np.clip(let,0,H)
    d1=np.linalg.norm(Po[t-1,j]-Po[t-1,i],axis=-1); d3=np.linalg.norm(Po[t-3,j]-Po[t-3,i],axis=-1)
    acc=(d-2*d1+np.linalg.norm(Po[t-2,j]-Po[t-2,i],axis=-1))
    si=np.linalg.norm(vi,axis=-1); sj=np.linalg.norm(vj,axis=-1)
    vi2=(Po[t-3,i]-Po[t-6,i])/(3*dt); vj2=(Po[t-3,j]-Po[t-6,j])/(3*dt)
    mi=np.linalg.norm(vi-vi2,axis=-1); mj=np.linalg.norm(vj-vj2,axis=-1)
    X=np.c_[d/R,vr,sv,let,(d-d3)/3,acc,si,sj,np.abs(dp[...,2]),(R-d)/np.maximum(-vr,0.5),np.maximum(mi,mj),mi+mj]
    return X,let

def hello_age(D,R,rng,base=0.05,edge=0.6,cong=None,amax=10):
    """age (s) of the latest HELLO from j received at i; inf if never. loss grows near range edge and with sender congestion."""
    T,N,_=D.shape
    cong=np.zeros(N) if cong is None else cong
    age=np.full((N,N),amax,float); A=np.zeros((T,N,N))
    for t in range(T):
        pl=np.clip(base+edge*(D[t]/R)**4+cong[None,:],0,0.97)
        rec=(D[t]<=R)&(rng.random((N,N))>pl)
        age=np.where(rec,0.0,np.minimum(age+1,amax))
        A[t]=age
    return A.astype(int)

def features_age(Po,A,t,i,j,R,H,dt=1.0):
    """features computed from the stale HELLO of neighbour j (age a) and own fresh state of i"""
    a=A[t,i,j]; tj=t-a
    pi=Po[t,i]; vi=(Po[t,i]-Po[t-3,i])/3
    pj0=Po[tj,j]; vj=(Po[tj,j]-Po[tj-3,j])/3
    pj=pj0+vj*a[:,None]                  # dead-reckoning to now
    dp=pj-pi; dv=vj-vi
    d=np.linalg.norm(dp,axis=-1); vr=(dp*dv).sum(-1)/np.maximum(d,1e-6); sv=np.linalg.norm(dv,axis=-1)
    aa=(dv*dv).sum(-1); b=2*(dp*dv).sum(-1); c=(dp*dp).sum(-1)-R*R
    disc=np.maximum(b*b-4*aa*c,0)
    let=np.clip(np.where(aa>1e-9,(-b+np.sqrt(disc))/(2*np.maximum(aa,1e-9)),H),0,H)
    vi2=(Po[t-3,i]-Po[t-6,i])/3; vj2=(Po[tj-3,j]-Po[tj-6,j])/3
    mi=np.linalg.norm(vi-vi2,axis=-1); mj=np.linalg.norm(vj-vj2,axis=-1)
    X=np.c_[d/R,vr,sv,let,a,np.linalg.norm(vi,axis=-1),np.linalg.norm(vj,axis=-1),np.abs(dp[:,2]),np.maximum(mi,mj),(R-d)/np.maximum(-vr,0.5)]
    return X,let

def group_trace(N=50,G=3,T=1200,dt=1.0,box=(3000,3000,200),zmin=100,vmean=(12,20),spread=300.,seed=0,mix=(0.4,0.4,0.2),sigc=(1.5,4.0,8.0),pull=0.003,hdev=np.pi/8,gpull=0.002):
    """Reference-point group mobility: each sub-swarm follows a Gauss-Markov reference point (mission leg);
    members move around it with mean-reverting Gauss-Markov offsets whose agility depends on the UAV class."""
    rng=np.random.default_rng(seed)
    L=np.array(box,float); lo=np.array([0,0,zmin]); hi=lo+L
    g=np.arange(N)%G
    cls=rng.choice(3,N,p=list(mix))
    a=np.array([0.95,0.85,0.6])[cls][:,None]; sig=np.array(sigc)[cls][:,None]
    # references
    rp=np.c_[rng.uniform(300,L[0]-300,G),rng.uniform(300,L[1]-300,G),rng.uniform(zmin+50,zmin+L[2]-50,G)]
    sp=rng.uniform(*vmean,G); th0=rng.uniform(0,2*np.pi); dth=rng.normal(0,hdev,G); th=th0+dth
    rp=np.c_[L[0]/2+rng.normal(0,150,G),L[1]/2+rng.normal(0,150,G),rng.uniform(zmin+50,zmin+L[2]-50,G)]
    rv=np.c_[sp*np.cos(th),sp*np.sin(th),np.zeros(G)]
    off=rng.normal(0,spread/2,(N,3))*np.array([1,1,0.3]); ov=np.zeros((N,3))
    P=np.zeros((T,N,3))
    for t in range(T):
        P[t]=np.clip(rp[g]+off,lo,hi)
        if rng.random()<0.01: th0+=rng.normal(0,np.pi/3)
        ch=rng.random(G)<0.01; dth[ch]=rng.normal(0,hdev,ch.sum())
        th=th0+dth
        # keep the swarm inside: steer common heading towards centre when near walls
        c=rp.mean(0)
        if c[0]<500 or c[0]>L[0]-500 or c[1]<500 or c[1]>L[1]-500:
            th0=np.arctan2(L[1]/2-c[1],L[0]/2-c[0])+rng.normal(0,0.3)
        vb=np.c_[sp*np.cos(th),sp*np.sin(th),np.zeros(G)]
        c=rp.mean(0)
        rv=0.9*rv+0.1*vb+np.sqrt(1-0.81)*1.0*rng.normal(size=(G,3))*np.array([1,1,0.2])+gpull*(c-rp)
        rp=rp+rv*dt
        rp=np.clip(rp,lo,hi)
        # offsets: mean-reverting GM (pull towards formation slot)
        ov=a*ov+np.sqrt(1-a*a)*sig*rng.normal(size=(N,3))*np.array([1,1,0.3])-pull*off
        off=off+ov*dt
    return P,g,cls
