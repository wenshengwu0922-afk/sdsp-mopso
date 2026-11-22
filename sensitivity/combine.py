"""Section 4.7 (Table 18): exact-front statistics (points, overlap, maximum loss over the five budgets) combined with
the algorithm results of ana.py. -> results/sens_table.csv"""
import json, pandas as pd, sys
sys.argv=['x']; import sens
M={'base':'Baseline','eta08':'eta x0.8','eta12':'eta x1.2','etaUNEP':'eta UNEP/ISSF','intlmat':'Intl. materials',
   'proc05':'Process x0.5','proc15':'Process x1.5','allintl':'All intl. + UNEP eta'}
scen=[(k,dict(sens.SCEN)[v]) for k,v in M.items()]
A=pd.read_csv('results/sens_algs.csv')
out=[]
for case,cn in [('ceiling_fan','fan'),('stapler','stapler')]:
    R=sens.run(case,scen)
    for k in M:
        r=R[k]; a=A[(A.scen==k)&(A.case==cn)].set_index('alg')
        ch=sum(1 for d in r['dec'] if d and d['loss']>0.05); ml=max(d['loss'] for d in r['dec'] if d)
        out.append(dict(scen=k,case=cn,n=r['n'],jac=r['jaccard'],changed=ch,maxloss=ml,
            m_reach=int(a.loc['MOPSO','reach']),m_nfe=a.loc['MOPSO','nfe_med'],p_reach=int(a.loc['PSO','reach']),
            p_nfe=a.loc['PSO','nfe_med'],p=a.loc['PSO','p_holm'],
            others_max_reach=int(a.loc[['KG_GA','NSGA2','TBGA','RS'],'reach'].max())))
pd.DataFrame(out).to_csv('results/sens_table.csv',index=False); print(pd.DataFrame(out).to_string())
