"""Section 4.7 (Table 19): Monte Carlo sensitivity, 500 samples per case. Each carbon parameter, each liaison rating
(hence each disassembly time and cost), or both, multiplied by an independent factor drawn from U(0.8, 1.2); the five
budgets stay at their baseline NT$ values. Metrics: exact-front points, overlap (Jaccard), share of samples with all
five decisions unchanged, share of budget cases in which the baseline decision becomes infeasible, loss (95th
percentile, maximum) over the budget cases in which the baseline decision remains feasible. -> results/mc.json"""
import numpy as np, sens
rng=np.random.default_rng(20261111)
B=sens.BASE
def build(case):
    parts,lia,rule=sens.parse(case); SL=sens.states(parts,rule); N=len(parts)
    X=np.array([[(S>>(i-1))&1 for i in range(1,N+1)] for S in SL],dtype=float)
    Cut=np.array([[1.0 if ((S>>(a-1))&1 or (S>>(b-1))&1) else 0.0 for a,b,r in lia] for S in SL])
    return parts,lia,SL,X,Cut
def objs(parts,lia,X,Cut,P,rat):
    g,gb=sens.model(parts,lia,P); g=g[1:]; gb=gb[1:]
    c=np.array(rat)*4*196/3600
    return Cut@c, X@g+(1-X)@gb, c.sum()
def front(f1,f2):
    o=np.lexsort((-f2,f1)); out=[]; best=-1e18
    for i in o:
        if f2[i]>best+1e-9: out.append(i); best=f2[i]
    return set(out)

def mc(case, carbon, time, n=500):
    parts,lia,SL,X,Cut=build(case); rat0=[r for a,b,r in lia]
    f1b,f2b,f1m=objs(parts,lia,X,Cut,B,rat0); Fb=front(f1b,f2b)
    J=[];L=[];CH=[];INF=[];NP=[]
    for s in range(n):
        u=lambda v,on: [x*rng.uniform(0.8,1.2) if on else x for x in v]
        P=dict(OCF=u(B['OCF'],carbon),RCF=u(B['RCF'],carbon),PCF=u(B['PCF'],carbon),ETA=u(B['ETA'],carbon))
        rat=u(rat0,time)
        f1,f2,_=objs(parts,lia,X,Cut,P,rat); F=front(f1,f2); NP.append(len(F)); J.append(len(F&Fb)/len(F|Fb))
        ch=0; ml=0; inf=0
        for fr in (0.1,0.2,0.3,0.5,0.7):
            Bd=fr*f1m                                   # absolute budget (NT$) fixed at its baseline value
            ok=np.where(f1<=Bd+1e-9)[0]; i=ok[np.argmax(f2[ok])]
            okb=np.where(f1b<=Bd+1e-9)[0]; ib=okb[np.argmax(f2b[okb])]
            if f1[ib]>Bd+1e-9: inf+=1; ch+=1; continue
            loss=100*(f2[i]-f2[ib])/f2[i]; ml=max(ml,loss); ch+= loss>0.05
        L.append(ml); CH.append(ch); INF.append(inf)
    J=np.array(J);L=np.array(L);CH=np.array(CH);INF=np.array(INF)
    return dict(case=case,carbon=carbon,time=time,base_pts=len(Fb),pts=(min(NP),max(NP)),Jmed=np.median(J),Jp5=np.percentile(J,5),
                Jmin=J.min(),Lmed=np.median(L),Lp95=np.percentile(L,95),Lmax=L.max(),unch=100*np.mean(CH==0),
                chmean=CH.mean(),infshare=100*INF.sum()/(5*n))
import json
res=[]
for case in ['stapler','ceiling_fan']:
    for carbon,time in [(True,False),(False,True),(True,True)]:
        r=mc(case,carbon,time); res.append(r); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
json.dump(res,open('results/mc.json','w'),default=float)
