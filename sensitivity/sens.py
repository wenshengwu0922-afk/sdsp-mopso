"""Section 4.7: exact Pareto front under the carbon-parameter scenarios of Table 18 (exact enumeration of all feasible
stopping states, as in Table 8). Model and data are parsed from src/problem.hpp so that both implementations agree.
Scenarios: eta x0.8 / x1.2; global EOL recycling rates as eta; international material factors; process factors x0.5 /
x1.5; international factors + global eta. Budgets: 10/20/30/50/70 % of f1max."""
import re, itertools, json, sys, os
import numpy as np
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'problem.hpp')).read()
MATS = ['IRON', 'STEEL', 'ALU', 'SUS', 'CU']; PROCS = ['STAMP', 'EXTRUDE', 'MACHINE']
BASE = dict(OCF=[3.27, 3.27, 9.85, 3.562, 3.65], RCF=[0.657, 0.678, 0.844, 2.831, 0.697],
            PCF=[0.662, 0.912, 0.262], ETA=[0.47, 0.47, 0.51, 0.50, 0.34])

def parse(fn):
    body = SRC[SRC.index('inline Instance %s()' % fn):]
    body = body[:body.index('\n}\n')]
    parts = [(int(a), MATS.index(b), float(c), PROCS.index(d), e) for a, b, c, d, e in
             re.findall(r'\{(\d+), (\w+), ([\d.]+), (\w+), (\w+)\}', body)]
    tri = re.findall(r'\{(\d+), (\d+), ([\d.]+)\}', body)
    lia = [(int(a), int(b), float(c)) for a, b, c in tri]
    pairs = re.findall(r'\{(\d+), (\d+)\}', body.split('}},')[-1] if False else body[body.rindex('{{'):])
    rule = [(int(a), int(b)) for a, b in pairs]
    return parts, lia, rule

def states(parts, rule):
    N = len(parts); pred = [0] * (N + 1)
    for a, b in rule: pred[b] |= 1 << (a - 1)
    # transitive closure not needed for ideal generation (direct predecessors suffice)
    seen = {0}; st = [0]
    while st:
        S = st.pop()
        for i in range(1, N + 1):
            bit = 1 << (i - 1)
            if not S & bit and (pred[i] & ~S) == 0 and S | bit not in seen:
                seen.add(S | bit); st.append(S | bit)
    seen.discard(0)
    return sorted(seen)

def model(parts, lia, P):
    N = len(parts); g = np.zeros(N + 1); gb = np.zeros(N + 1)
    for pid, m, w, q, s in parts:
        o, r, phi = P['OCF'][m] * w, P['RCF'][m] * w, P['PCF'][q] * w
        g[pid] = {'RU': o + phi, 'RM': o, 'RC': o - r, 'T': 0.0}[s]
        gb[pid] = P['ETA'][m] * (o - r)
    return g, gb

def evaluate(parts, lia, S_list, P):
    N = len(parts); g, gb = model(parts, lia, P)
    cost = {}
    f1 = []; f2 = []
    c = [(a, b, r * 4.0 * 196.0 / 3600.0) for a, b, r in lia]
    for S in S_list:
        x = sum(cc for a, b, cc in c if (S >> (a - 1)) & 1 or (S >> (b - 1)) & 1)
        y = sum(g[i] if (S >> (i - 1)) & 1 else gb[i] for i in range(1, N + 1))
        f1.append(x); f2.append(y)
    return np.array(f1), np.array(f2), sum(cc for *_, cc in c), g.sum()

def front(f1, f2):
    idx = sorted(range(len(f1)), key=lambda i: (f1[i], -f2[i])); out = []; best = -1e18
    for i in idx:
        if f2[i] > best + 1e-9: out.append(i); best = f2[i]
    return out

def hv(f1, f2, fr, f1m, f2m, ref=1.1):
    pts = sorted((f1[i] / f1m, 1 - f2[i] / f2m) for i in fr); h = 0; prev = ref
    for x, y in sorted(pts, key=lambda t: t[0], reverse=False):
        pass
    # standard 2-D HV for minimization of (x, y)
    pts.sort(); h = 0.0; ycur = ref
    for x, y in pts:
        if y < ycur: h += (ref - x) * (ycur - y); ycur = y
    return h

def run(case, scen):
    parts, lia, rule = parse(case)
    SL = states(parts, rule)
    out = {}
    base = None
    for name, P in scen:
        f1, f2, f1m, f2m = evaluate(parts, lia, SL, P)
        fr = front(f1, f2)
        sets = {SL[i] for i in fr}
        rec = dict(n=len(fr), hv=hv(f1, f2, fr, f1m, f2m), sets=sets, f1=f1, f2=f2, fr=fr)
        if base is None: base = rec
        jac = len(sets & base['sets']) / len(sets | base['sets'])
        dec = []
        for B in [fr_ * f1m for fr_ in (0.1, 0.2, 0.3, 0.5, 0.7)]:
            ok = [i for i in range(len(SL)) if f1[i] <= B + 1e-9]
            if not ok: dec.append(None); continue
            i = max(ok, key=lambda i: (f2[i], -f1[i]))
            ib = max(ok, key=lambda i: (base['f2'][i], -f1[i]))
            dec.append(dict(B=B, same=SL[i] == SL[ib], k=bin(SL[i]).count('1'), kb=bin(SL[ib]).count('1'),
                            loss=100 * (f2[i] - f2[ib]) / f2[i] if f2[i] else 0.0))   # carbon lost if the baseline decision is kept
        out[name] = dict(n=rec['n'], hv=rec['hv'], jaccard=jac, nstates=len(SL), dec=dec)
    return out

def mk(**kw):
    P = json.loads(json.dumps(BASE)); 
    for k, v in kw.items(): P[k] = v
    return P

UNEP = [0.72, 0.72, 0.57, 0.85, 0.48]
SCEN = [
    ('Baseline', mk()),
    ('eta x0.8', mk(ETA=[e * 0.8 for e in BASE['ETA']])),
    ('eta x1.2', mk(ETA=[e * 1.2 for e in BASE['ETA']])),
    ('eta UNEP/ISSF', mk(ETA=UNEP)),
    ('eta = 0', mk(ETA=[0] * 5)),
    ('Stainless 5.3/1.6', mk(OCF=[3.27, 3.27, 9.85, 5.3, 3.65], RCF=[0.657, 0.678, 0.844, 1.6, 0.697])),
    ('Intl. materials', mk(OCF=[2.66, 2.66, 14.8, 5.3, 3.965], RCF=[0.71, 0.71, 0.52, 1.6, 0.697])),
    ('Process x0.5', mk(PCF=[p * 0.5 for p in BASE['PCF']])),
    ('Process x1.5', mk(PCF=[p * 1.5 for p in BASE['PCF']])),
    ('Virgin x0.8', mk(OCF=[o * 0.8 for o in BASE['OCF']])),
    ('Virgin x1.2', mk(OCF=[o * 1.2 for o in BASE['OCF']])),
    ('Recycled x0.8', mk(RCF=[r * 0.8 for r in BASE['RCF']])),
    ('Recycled x1.2', mk(RCF=[r * 1.2 for r in BASE['RCF']])),
    ('All intl. + UNEP eta', mk(OCF=[2.66, 2.66, 14.8, 5.3, 3.965], RCF=[0.71, 0.71, 0.52, 1.6, 0.697], ETA=UNEP)),
]
if __name__ == '__main__':
    for case in ['stapler', 'ceiling_fan']:
        R = run(case, SCEN)
        print('==', case, 'states', R['Baseline']['nstates'])
        for n, r in R.items():
            d = ' '.join('%.1f:%s(k%d/%d,%.1f%%)' % (x['B'], 'Y' if x['same'] else 'N', x['k'], x['kb'], x['loss']) for x in r['dec'] if x)
            print('%-22s n=%2d HV=%.4f J=%.2f | %s' % (n, r['n'], r['hv'], r['jaccard'], d))
