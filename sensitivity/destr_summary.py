"""Table 21: summary of the damage experiment by number of damaged parts (reads results/destr_rec.csv, destr_alg.csv)."""
import pandas as pd
R = pd.read_csv('results/destr_rec.csv'); A = pd.read_csv('results/destr_alg.csv')
rows = []
for q, g in R.groupby('q'):
    a = A[A.q == q]
    rows.append(dict(q=q, n=len(a), jmed=g.jac.median(), jp5=g.jac.quantile(.05), changed=100 * g.changed.mean(),
                     dk=g.dk.mean(), back=100 * (g.dk > 0).mean(), fwd=100 * (g.dk < 0).mean(),
                     dkmin=g.dk.min(), dkmax=g.dk.max(), drop=g['drop'].mean(), dropmax=g['drop'].max(),
                     infeas=100 * (~g.feas).mean(), loss=g.loss.mean(), lossp95=g.loss.quantile(.95),
                     lossmax=g.loss.max(), reachmin=a.reach.min(), reachmean=a.reach.mean(), nfe=a.nfe.median()))
pd.DataFrame(rows).to_csv('results/destr_summary.csv', index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
