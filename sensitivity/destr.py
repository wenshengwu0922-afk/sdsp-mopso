"""Section 4.8 (Tables 20-21): effect of component damage on the stopping point, ceiling fan. A damaged part is
downgraded from RU/RM to RC and its liaisons are removed destructively (Table 20; SSDSP_DESTR in src/problem.hpp).
All 10 + 45 + 120 combinations of 1-3 damaged RU/RM parts; MOPSO (Table 11 parameters), E_max = 20,000, 30 runs
(seeds 7001-7030). Metrics per run and budget (10/20/30/50/70 % of the undamaged f1max, kept at their NT$ values):
overlap (Jaccard index of non-dominated stopping states), change of the best stopping point dk, and loss (carbon
recovery forgone if the decision planned for undamaged parts is executed). Summary: destr_summary.py."""
import itertools, subprocess, os, json, numpy as np, pandas as pd, sens
from concurrent.futures import ThreadPoolExecutor
parts, lia, rule = sens.parse('ceiling_fan'); N = len(parts)
ELIG = [pid for pid, m, w, q, s in parts if s in ('RU', 'RM')]
DMAP = {2.2: 1.8, 2.1: 1.8, 1.8: 1.6, 1.6: 1.4, 1.4: 1.2, 1.3: 1.4, 1.2: 1.0, 2.0: 2.0}
def objs_set(S, D):
    """S: set of removed part ids, D: destroyed parts -> (f1, f2) under the scenario"""
    P = sens.BASE; f1 = 0.0; f2 = 0.0
    for a, b, r in lia:
        if a in S or b in S:
            rr = DMAP[r] if (a in D or b in D) else r
            f1 += rr * 4 * 196 / 3600
    for pid, m, w, q, s in parts:
        o, rc, phi = P['OCF'][m] * w, P['RCF'][m] * w, P['PCF'][q] * w
        st = 'RC' if (pid in D and s in ('RU', 'RM')) else s
        g = {'RU': o + phi, 'RM': o, 'RC': o - rc, 'T': 0.0}[st]
        f2 += g if pid in S else P['ETA'][m] * (o - rc)
    return f1, f2
F1MAX = sum(r * 4 * 196 / 3600 for a, b, r in lia)
BUD = [f * F1MAX for f in (0.1, 0.2, 0.3, 0.5, 0.7)]
def run_mopso(D, tag):
    env = dict(os.environ, SSDSP_DESTR=','.join(map(str, D)))
    subprocess.run([os.path.join('..', 'ssdsp'), '--alg', 'MOPSO', '--case', 'fan', '--budget', '20000', '--seeds', '7001-7030',
                    '--set', 'w=0,c1=3.5,c2=4.5', '--front', '--threads', '1', '--out', 'out/ds_' + tag],
                   env=env, check=True, capture_output=True)
    runs = pd.read_csv('out/ds_%s_runs.csv' % tag); fr = pd.read_csv('out/ds_%s_front.csv' % tag)
    fronts = {}
    for seed, g in fr.groupby('seed'):
        fronts[seed] = [frozenset(map(int, r.sequence.split()[:int(r.stopping_point)])) for r in g.itertuples()]
    return runs, fronts
def decide(front_sets, D, B):
    best = None
    for S in front_sets:
        f1, f2 = objs_set(S, D)
        if f1 <= B + 1e-9 and (best is None or f2 > best[2] + 1e-12): best = (S, f1, f2)
    return best
# baseline (no damage)
runs0, fr0 = run_mopso([], 'base')
base_sets = set(fr0[7001]); base_dec = [decide(base_sets, set(), B) for B in BUD]
scen = [c for q in (1, 2, 3) for c in itertools.combinations(ELIG, q)]
def one(D):
    tag = 'q%d_' % len(D) + '_'.join(map(str, D))
    runs, frs = run_mopso(list(D), tag)
    rec = []
    for seed, sets in frs.items():
        S = set(sets); jac = len(S & base_sets) / len(S | base_sets)
        for bi, B in enumerate(BUD):
            nb = decide(S, set(D), B); bs = base_dec[bi]
            f1o, f2o = objs_set(bs[0], set(D))            # baseline decision executed under the damage scenario
            feas = f1o <= B + 1e-9
            rec.append(dict(q=len(D), D=tag, seed=seed, B=bi, jac=jac, k_new=len(nb[0]), k_base=len(bs[0]),
                            dk=len(nb[0]) - len(bs[0]), changed=nb[0] != bs[0], drop=100 * (bs[2] - nb[2]) / bs[2],
                            feas=feas, loss=100 * (nb[2] - f2o) / nb[2] if feas else np.nan))
    reach = int((runs.nfe_opt > 0).sum()); nfe = runs.nfe_opt[runs.nfe_opt > 0].median()
    return rec, dict(q=len(D), D=tag, reach=reach, nfe=nfe)
with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(one, scen))
R = pd.DataFrame([r for rec, _ in out for r in rec]); A = pd.DataFrame([a for _, a in out])
R.to_csv('results/destr_rec.csv', index=False); A.to_csv('results/destr_alg.csv', index=False)
print('eligible', ELIG, 'scenarios', len(scen), 'baseline reach', int((runs0.nfe_opt > 0).sum()))
for q, g in R.groupby('q'):
    a = A[A.q == q]
    print('q=%d n_scen=%d | Jaccard med %.2f p5 %.2f | changed %.1f%% | dk mean %.2f, back %.1f%% fwd %.1f%% | '
          'drop mean %.2f%% max %.2f%% | base infeasible %.1f%% | loss mean %.2f%% p95 %.2f%% max %.2f%% | '
          'MOPSO reach min %d mean %.1f NFE med %.0f' % (
          q, a.shape[0], g.jac.median(), g.jac.quantile(.05), 100 * g.changed.mean(), g.dk.mean(),
          100 * (g.dk > 0).mean(), 100 * (g.dk < 0).mean(), g.drop.mean(), g.drop.max(),
          100 * (~g.feas).mean(), g.loss.mean(), g.loss.quantile(.95), g.loss.max(), a.reach.min(), a.reach.mean(),
          a.nfe.median()))
