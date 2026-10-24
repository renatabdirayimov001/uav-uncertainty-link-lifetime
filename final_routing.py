import sys, os, json, time
seeds=[int(x) for x in sys.argv[1].split(',')]; scen=os.environ.get('SCEN','base')
sys.argv=['x','0.1']
import numpy as np, exp2
from exp2 import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
models={}
for a_ in [0.05,0.1,0.2,0.3,0.5]:
    m=HGB(loss='quantile',quantile=a_,max_iter=300,learning_rate=0.08).fit(Xtr,Ytr)
    models[a_]=(m,np.quantile(m.predict(Xca)-Yca,min(1,(1-a_)*(1+1/len(Yca)))))
old=exp2.est
def est2(name,X,let):
    if name.startswith('CQR'):
        m,qq=models[float(name.split('@')[1])]; return m.predict(X)-qq
    return old(name,X,let)
exp2.est=est2
TU=2
pol={'ORACLE':('ORACLE',TU),'HOP':('HOP',TU)}
for m in [2,4,6,9]: pol[f'LET m={m}']=('LET',m); pol[f'ML m={m}']=('ML-point',m)
for m in [2,4]: pol[f'Ens m={m}']=('Ens-LB',m)
pol['CP-point m=2']=('CP-point',2)
for a_ in [0.05,0.1,0.2,0.3,0.5]: pol[f'CQR a={a_}']=(f'CQR@{a_}',TU)
os.makedirs('results',exist_ok=True)
for sd in seeds:
    t0=time.time(); r=evaluate(sd,pol,T=900,nflows=15,Tu=TU)
    out={k:dict(PDR=M['deliv']/M['sec'],breaks=M['breaks']/M['mins'],disc=M['disc']/M['mins'],hops=M['hops']/max(1,M['hopsn'])) for k,M in r.items()}
    json.dump(out,open(f'results/route_{scen}_s{sd}.json','w'),indent=1)
    print(scen,'seed',sd,'done',round(time.time()-t0),'s',flush=True)
