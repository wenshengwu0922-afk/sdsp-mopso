"""Section 4.7 (Table 18): Reach and NFE* of the six methods under each carbon-parameter scenario (out/sens_*_runs.csv);
MOPSO vs PSO on NFE* by Wilcoxon signed-rank tests, Holm-adjusted within scenario and case. -> results/sens_algs.csv"""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
SC=['base','eta08','eta12','etaUNEP','intlmat','proc05','proc15','allintl']
ALG=['MOPSO','PSO','KG_GA','NSGA2','TBGA','RS']
rows=[]
for s in SC:
  for c in ['stapler','fan']:
    bud=1500 if c=='stapler' else 20000
    d={a:pd.read_csv(f'out/{s}_{c}_{a}_runs.csv').sort_values('seed') for a in ALG}
    m=d['MOPSO']; mn=m.nfe_opt.where(m.nfe_opt>0,bud).values
    for a in ALG:
      x=d[a]; reach=(x.nfe_opt>0).sum(); nfe=x.nfe_opt[x.nfe_opt>0]
      p=np.nan
      if a!='MOPSO':
        xn=x.nfe_opt.where(x.nfe_opt>0,bud).values
        p=wilcoxon(mn,xn).pvalue if np.any(mn!=xn) else 1.0
      rows.append(dict(scen=s,case=c,alg=a,meanHV=x.HV.mean(),reach=reach,nfe_med=nfe.median() if len(nfe) else np.nan,p_nfe_raw=p))
R=pd.DataFrame(rows)
# Holm within scen/case
def holm(g):
    g=g.copy(); idx=g.index[g.alg!='MOPSO']; ps=g.loc[idx,'p_nfe_raw'].values; o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*ps[i]); adj[i]=min(1,run)
    g.loc[idx,'p_holm']=adj; return g
R=pd.concat([holm(g) for _,g in R.groupby(['scen','case'])])
R.to_csv('results/sens_algs.csv',index=False)
pd.set_option('display.width',200)
for c in ['stapler','fan']:
  print('=====',c)
  t=R[R.case==c].pivot_table(index='scen',columns='alg',values='reach').reindex(SC)[ALG]; print('reach\n',t)
  t=R[R.case==c].pivot_table(index='scen',columns='alg',values='nfe_med').reindex(SC)[['MOPSO','PSO']]; print('nfe med\n',t)
  t=R[R.case==c].pivot_table(index='scen',columns='alg',values='p_holm').reindex(SC)[ALG[1:]]; print('p holm\n',t.round(4))
for s in SC:
  for c in ['stapler','fan']:
    e=pd.read_csv(f'out/{s}_{c}_exact.csv'); print(s,c,len(e))
