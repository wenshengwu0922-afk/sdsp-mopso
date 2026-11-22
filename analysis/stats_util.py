"""Shared statistics and table helpers (Section 4.3: Friedman, Wilcoxon signed-rank with Holm correction, effect sizes).

Comparison of k >= 3 methods on the same seeds: Friedman test (blocks = seeds) and Friedman mean ranks; if
p < 0.05, two-sided Wilcoxon signed-rank tests (paired by seed) of the reference method against every other
method, Holm-adjusted. Effect size: matched-pairs rank-biserial correlation r = (R+ - R-) / (R+ + R-), where
positive values favour the reference method (higher HV).
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

ALG_ORDER = ["MOPSO", "PSO", "KG_GA", "NSGA2", "TBGA", "RS"]
ALG_LABEL = {"MOPSO": "MOPSO", "PSO": "PSO", "KG_GA": "K&G-GA", "NSGA2": "NSGA-II", "TBGA": "TBGA",
             "RS": "Random search"}
PARAMS = {"MOPSO": ["w", "c1", "c2", "pm", "Q", "beta1"], "PSO": ["w", "c1", "c2"], "KG_GA": ["pc", "pm"],
          "NSGA2": ["pc", "pm"], "TBGA": ["team_ratio", "delta", "nm_ratio"], "RS": []}
PARAM_LABEL = {"Q": "Q (number of layers)", "w": "w (inertia weight)", "c1": "c1 (cognitive coefficient)",
               "c2": "c2 (social coefficient)", "pm": "p_m (mutation probability)",
               "beta1": "beta1 (clustering coefficient of sigma1; beta2 = 1 - beta1)",
               "pc": "p_c (crossover probability)", "team_ratio": "team size / population",
               "delta": "delta (team convergence threshold)", "nm_ratio": "N_M / N (MHM moves)"}
TOL = 1e-9


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "---"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[idx]))
        adj[idx] = run
    return adj


def wilcoxon_paired(x, y):
    """Two-sided Wilcoxon signed-rank test of x vs y (paired). Returns (W, p, r_rb, n_nonzero)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d[np.abs(d) < TOL] = 0.0
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0, 1.0, 0.0, 0
    ranks = stats.rankdata(np.abs(nz))
    rp, rm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = stats.wilcoxon(nz, alternative="two-sided")
    return float(min(rp, rm)), float(res.pvalue), float((rp - rm) / (rp + rm)), int(len(nz))


def block_matrix(df, group_col, groups, value="HV"):
    """Seeds x groups matrix (rows aligned by seed); raises if a seed is missing for some group."""
    piv = df.pivot_table(index="seed", columns=group_col, values=value, aggfunc="first")
    piv = piv[[g for g in groups if g in piv.columns]]
    if piv.isna().any().any():
        raise SystemExit(f"unpaired data: some seeds missing for {group_col}")
    return piv


def friedman_ranks(piv):
    """Friedman test over the columns of a seeds x groups matrix (higher value = better = rank 1)."""
    vals = piv.to_numpy()
    ranks = np.vstack([stats.rankdata(-row) for row in np.round(vals, 12)])
    mean_rank = pd.Series(ranks.mean(axis=0), index=piv.columns)
    if np.all(np.abs(vals - vals[:, :1]) < TOL):
        return np.nan, np.nan, mean_rank, True                 # all groups identical on every seed: no test
    chi2, p = stats.friedmanchisquare(*[vals[:, j] for j in range(vals.shape[1])])
    if np.isnan(p):
        p = 1.0
    return float(chi2), float(p), mean_rank, False


def compare_to_reference(piv, ref, require_friedman=True):
    """Friedman test, then Wilcoxon (ref vs each other column) with Holm adjustment.
    Returns (summary dict, DataFrame with one row per column)."""
    chi2, pf, mean_rank, identical = friedman_ranks(piv) if piv.shape[1] >= 3 else (np.nan, np.nan, None, False)
    rows = []
    others = [c for c in piv.columns if c != ref]
    do_post = not identical and (not require_friedman or (not np.isnan(pf) and pf < 0.05) or piv.shape[1] < 3)
    raw = []
    for c in others:
        W, p, r, n = wilcoxon_paired(piv[ref], piv[c]) if do_post else (np.nan, np.nan, np.nan, 0)
        raw.append(p)
        rows.append({"group": c, "W": W, "p_raw": p, "r_rb": r, "n_nonzero": n})
    out = pd.DataFrame(rows)
    if do_post and len(out):
        out["p_holm"] = holm(out["p_raw"].to_numpy())
    else:
        out["p_holm"] = np.nan
    return {"chi2": chi2, "p_friedman": pf, "mean_rank": mean_rank, "identical": identical,
            "post_hoc": do_post}, out


# ---------------------------------------------------------------------------------------------- exact front
def load_exact(analysis_dir, case):
    f = pathlib.Path(analysis_dir) / f"exact_{case}.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found (run 00_build.ps1)")
    e = pd.read_csv(f)
    return e, float(e["f1max"].iloc[0]), float(e["f2max"].iloc[0])


def hv_from_points(f1, f2, f1max, f2max, ref=1.1):
    x = np.asarray(f1, float) / f1max
    y = 1.0 - np.asarray(f2, float) / f2max
    keep = (x < ref) & (y < ref)
    pts = sorted(zip(x[keep], y[keep]))
    area, ybest = 0.0, ref
    for xi, yi in pts:
        if yi < ybest - 1e-12:
            area += (ref - xi) * (ybest - yi)
            ybest = yi
    return area


def exact_points_found(f1, f2, exact):
    """Number of exact Pareto points that appear (within 1e-9 relative tolerance) among (f1, f2)."""
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    cnt = 0
    for e1, e2 in zip(exact["cost"].to_numpy(), exact["carbon"].to_numpy()):
        if np.any((np.abs(f1 - e1) <= 1e-9 * max(1.0, abs(e1))) & (np.abs(f2 - e2) <= 1e-9 * max(1.0, abs(e2)))):
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------------------------- tuning table
def write_tuning_table(grid_dir, best, stage2_note=None):
    """Reviewer-ready parameter table (levels, number of sets, selected values, mean HV of the selected set)."""
    grid_dir = pathlib.Path(grid_dir)
    budget, seeds = "?", "?"
    for alg in ALG_ORDER:
        f = grid_dir / f"grid_{alg}_fan_runs.csv"
        if f.exists():
            df0 = pd.read_csv(f, usecols=["budget", "seed"])
            budget, seeds = f"{int(df0['budget'].iloc[0]):,}", f"{df0['seed'].min()}-{df0['seed'].max()}"
            break
    lines = [f"# Parameter tuning (ceiling fan, E_max = {budget} evaluations, 30 runs per set, seeds {seeds})\n",
             "Selection rule: highest mean HV over the 30 runs; ties (equal mean HV) broken by the lower mean number "
             "of evaluations needed to reach the exact optimum (unreached runs counted as the budget), then by the "
             "lower standard deviation.\n",
             "| Algorithm | Parameter | Candidate levels | Selected |", "|---|---|---|---|"]
    notes = []
    for alg in ALG_ORDER:
        f = grid_dir / f"grid_{alg}_fan_runs.csv"
        if alg == "RS" or not f.exists() or alg not in best:
            continue
        df = pd.read_csv(f)
        varied = [p for p in PARAMS[alg] if df[p].nunique() > 1]
        nsets = df.groupby(varied).ngroups
        for p in varied:                       # parameters fixed in the grid (ablation switches) are not listed
            lv = sorted(df[p].unique())
            lv_s = ", ".join(f"{v:g}" for v in lv)
            lines.append(f"| {ALG_LABEL[alg]} | {PARAM_LABEL[p]} | {lv_s} | {best[alg][p]:g} |")
        notes.append(f"- {ALG_LABEL[alg]}: {nsets} parameter sets in the full-factorial grid (Section 4.2); selected set "
                     f"mean HV = {best[alg]['_mean']:.6f} (std {best[alg]['_std']:.6f}).")
    lines.append("")
    lines += notes
    if stage2_note:
        lines.append(stage2_note)
    (grid_dir / "tuning_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_best(grid_dir, best):
    clean = {a: {k: (int(v) if k == "Q" else float(v)) for k, v in d.items() if not k.startswith("_")}
             for a, d in best.items()}
    (pathlib.Path(grid_dir) / "best_params.json").write_text(json.dumps(clean, indent=2))
    (pathlib.Path(grid_dir) / "best_params_detail.json").write_text(json.dumps(best, indent=2))


def load_best_detail(grid_dir):
    return json.loads((pathlib.Path(grid_dir) / "best_params_detail.json").read_text())
