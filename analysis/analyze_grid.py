"""Parameter setting (Section 4.2): full-factorial grids, selected parameters (Table 11), factorial ANOVA of the
MOPSO grid (Table 12), Kruskal-Wallis check and tuning-robustness check.

usage: python analyze_grid.py results_v5/grid

Outputs (in the grid folder):
  grid_summary_<ALG>.csv     mean / std / min / max / median HV of every parameter set (30 runs)
  best_params.json           selected parameter set per algorithm (highest mean HV; tie (sets reaching HV* in all
                             30 runs): fewer mean evaluations to reach HV*, unreached runs counted as E_max)
  best_params_detail.json    the same with mean / std of the selected set
  anova_<ALG>.csv            factorial ANOVA (main effects + two-way interactions): df, F, p, partial eta^2 (Table 12)
  kruskal_<ALG>.csv          robustness check: Kruskal-Wallis per factor, epsilon^2
  oat_<ALG>.csv / .png       one-factor-at-a-time profile around the selected set
  tuning_effort.csv          share of all MOPSO parameter sets whose mean HV exceeds each baseline's best-tuned set
  tuning_table.md            parameter table (Table 11)
  report_stage1.md           summary of all of the above (Section 4.2)
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stats_util import ALG_LABEL, ALG_ORDER, PARAMS, fmt_p, save_best, write_tuning_table  # noqa: E402

# natural limits of each parameter: a selected value on such a limit is not a reason to extend the grid
# (team_ratio = 1: one team = whole population; delta >= 1: every team counts as converged, since HV_pt < 1.21)
HARD = {"Q": (1, 100), "w": (0, None), "c1": (0, None), "c2": (0, None), "pm": (0, 1), "pc": (0, 1),
        "beta1": (0, 1), "team_ratio": (0.03, 1), "delta": (0, 1), "nm_ratio": (0, None)}


def summarize(df, ps):
    """Per-set statistics, sorted by the selection rule: highest mean HV; ties (equal mean HV, e.g. sets that
    reach the exact optimum in every run) are broken by the lower mean number of evaluations needed to reach
    the exact optimum (runs that never reach it count as the full budget), then by the lower std."""
    df = df.copy()
    budget = df["budget"].iloc[0]
    nf = df["nfe_opt"] if "nfe_opt" in df else pd.Series(-1, index=df.index)
    df["speed"] = np.where(nf > 0, nf, budget)
    s = df.groupby(ps).agg(mean=("HV", "mean"), std=("HV", "std"), min=("HV", "min"), max=("HV", "max"),
                           median=("HV", "median"), count=("HV", "size"), reach=("speed", lambda v: int((v < budget).sum())),
                           mean_nfe=("speed", "mean")).reset_index()
    s["mean_r"] = s["mean"].round(10)
    s = s.sort_values(["mean_r", "mean_nfe", "std"], ascending=[False, True, True], kind="mergesort")
    return s.drop(columns="mean_r").reset_index(drop=True)


def factorial_anova(df, ps):
    """Balanced full-factorial ANOVA with all main effects and two-way interactions (higher-order terms pooled
    into the error term)."""
    y = df["HV"].to_numpy()
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    rows, main = [], {}
    for a in ps:
        m = df.groupby(a)["HV"].transform("mean").to_numpy()
        main[a] = m - grand
        rows.append((a, df[a].nunique() - 1, ((m - grand) ** 2).sum()))
    fitted = grand + sum(main.values())
    for a, b in itertools.combinations(ps, 2):
        cell = df.groupby([a, b])["HV"].transform("mean").to_numpy()
        eff = cell - main[a] - main[b] - grand
        rows.append((f"{a} x {b}", (df[a].nunique() - 1) * (df[b].nunique() - 1), (eff ** 2).sum()))
        fitted = fitted + eff
    df_err = len(y) - 1 - sum(r[1] for r in rows)
    ss_err = sst - sum(r[2] for r in rows)
    ms_err = ss_err / df_err
    out = [{"source": s, "df": d, "SS": ss, "F": (ss / d) / ms_err, "p": stats.f.sf((ss / d) / ms_err, d, df_err),
            "partial_eta2": ss / (ss + ss_err)} for s, d, ss in rows]
    out.append({"source": "error", "df": df_err, "SS": ss_err, "F": np.nan, "p": np.nan, "partial_eta2": np.nan})
    return pd.DataFrame(out), y - fitted


def assumption_checks(df, ps, resid):
    rng = np.random.default_rng(0)
    sample = resid if len(resid) <= 5000 else rng.choice(resid, 5000, replace=False)
    w, p_sw = stats.shapiro(sample)
    groups = [g["HV"].to_numpy() for _, g in df.groupby(ps)]
    lev, p_lev = stats.levene(*groups, center="median")
    return {"shapiro_W": w, "shapiro_p": p_sw, "brown_forsythe_F": lev, "brown_forsythe_p": p_lev}


def kruskal_per_factor(df, ps):
    out = []
    for a in ps:
        groups = [g["HV"].to_numpy() for _, g in df.groupby(a)]
        k, n = len(groups), len(df)
        H, p = stats.kruskal(*groups)
        out.append({"factor": a, "levels": k, "H": H, "p": p, "epsilon2": (H - k + 1) / (n - k)})
    return pd.DataFrame(out)


def oat(summary, ps, best):
    out = []
    for a in ps:
        mask = np.ones(len(summary), bool)
        for b in ps:
            if b != a:
                mask &= np.isclose(summary[b], best[b])
        for _, r in summary[mask].sort_values(a).iterrows():
            out.append({"factor": a, "level": r[a], "mean": r["mean"], "std": r["std"]})
    return pd.DataFrame(out)


def plot_oat(o, alg, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fs = list(o["factor"].unique())
    fig, axes = plt.subplots(1, len(fs), figsize=(3.0 * len(fs), 2.8), squeeze=False)
    for ax, f in zip(axes[0], fs):
        s = o[o["factor"] == f]
        ax.errorbar(s["level"], s["mean"], yerr=s["std"], marker="o", capsize=3, color="#1f4e79")
        ax.set_xlabel(f)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mean HV (30 runs)")
    fig.suptitle(f"{ALG_LABEL[alg]}: one-factor-at-a-time profile around the selected set", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def boundary_warnings(df, ps, best):
    warn = []
    for p in ps:
        lv = sorted(df[p].unique())
        if len(lv) < 3:
            continue
        lo, hi = HARD.get(p, (None, None))
        if best[p] == lv[0] and (lo is None or lv[0] > lo + 1e-12):
            warn.append(f"{p} = {best[p]:g} (lowest level)")
        if best[p] == lv[-1] and (hi is None or lv[-1] < hi - 1e-12):
            warn.append(f"{p} = {best[p]:g} (highest level)")
    return warn


def main(folder):
    folder = pathlib.Path(folder)
    best, report = {}, ["# Grid analysis (ceiling fan, 30 runs per set)\n"]
    summaries = {}
    for alg in ALG_ORDER:
        if alg == "RS":
            continue
        f = folder / f"grid_{alg}_fan_runs.csv"
        if not f.exists():
            print(f"[skip] {alg}: no grid file")
            continue
        df = pd.read_csv(f)
        if (df["nfe"] != df["budget"]).any():
            raise SystemExit(f"{alg}: some runs did not use the full budget")
        ps = [p for p in PARAMS[alg] if df[p].nunique() > 1]          # factors varied in the grid
        fixed = {p: float(df[p].iloc[0]) for p in PARAMS[alg] if p not in ps}
        counts = df.groupby(ps).size()
        if counts.nunique() != 1:
            raise SystemExit(f"{alg}: unbalanced grid (runs per set differ)")
        summ = summarize(df, ps)
        summaries[alg] = summ
        summ.to_csv(folder / f"grid_summary_{alg}.csv", index=False)
        b = {p: float(summ.loc[0, p]) for p in ps}
        b.update(fixed)
        best[alg] = {p: b[p] for p in PARAMS[alg]}
        best[alg]["_mean"] = float(summ.loc[0, "mean"])
        best[alg]["_std"] = float(summ.loc[0, "std"])
        sel = ", ".join(f"{p} = {b[p]:g}" for p in PARAMS[alg])
        report.append(f"## {ALG_LABEL[alg]}\n\nparameter sets: {len(summ)}, runs per set: {counts.iloc[0]}\n")
        report.append(f"selected: {sel};  mean HV = {summ.loc[0, 'mean']:.6f} (std {summ.loc[0, 'std']:.6f}; "
                      f"reached the exact optimum in {int(summ.loc[0, 'reach'])}/{int(summ.loc[0, 'count'])} runs, "
                      f"mean NFE to reach it {summ.loc[0, 'mean_nfe']:,.0f} with unreached runs counted as the budget); "
                      f"worst set mean = {summ['mean'].min():.6f}; sets with mean HV equal to the best: "
                      f"{int((summ['mean'].round(10) == round(summ.loc[0, 'mean'], 10)).sum())}\n")
        warn = boundary_warnings(df, ps, b)
        if warn:
            msg = "WARNING: selected value on a grid edge that is not a natural limit: " + "; ".join(warn)
            report.append(msg + "\n")
            print(f"[{alg}] {msg}")
        an, resid = factorial_anova(df, ps)
        an.to_csv(folder / f"anova_{alg}.csv", index=False)
        chk = assumption_checks(df, ps, resid)
        kw = kruskal_per_factor(df, ps)
        kw.to_csv(folder / f"kruskal_{alg}.csv", index=False)
        o = oat(summ, ps, b)
        o.to_csv(folder / f"oat_{alg}.csv", index=False)
        plot_oat(o, alg, folder / f"oat_{alg}.png")
        report.append("| source | df | F | p | partial eta^2 |\n|---|---|---|---|---|")
        for _, r in an.iterrows():
            if r["source"] != "error":
                report.append(f"| {r['source']} | {int(r['df'])} | {r['F']:.3f} | {fmt_p(r['p'])} | {r['partial_eta2']:.3f} |")
        report.append(f"| error | {int(an.iloc[-1]['df'])} | | | |\n")
        report.append(f"assumptions: Shapiro-Wilk on residuals (5,000 sampled) W = {chk['shapiro_W']:.3f} "
                      f"(p {fmt_p(chk['shapiro_p'])}); Brown-Forsythe F = {chk['brown_forsythe_F']:.3f} "
                      f"(p {fmt_p(chk['brown_forsythe_p'])})\n")
        report.append("Kruskal-Wallis per factor: " + "; ".join(
            f"{r['factor']}: H = {r['H']:.2f}, p {fmt_p(r['p'])}, eps2 = {r['epsilon2']:.3f}" for _, r in kw.iterrows()) + "\n")
    if not best:
        raise SystemExit("no grid results found")
    save_best(folder, best)

    if "MOPSO" in summaries:
        ms = summaries["MOPSO"]["mean"]
        rows = []
        report.append("## Tuning-effort check (all MOPSO parameter sets vs best-tuned baselines)\n")
        report.append("| baseline | baseline best-set mean HV | share of MOPSO sets with higher mean HV | "
                      "MOPSO median-set mean HV |\n|---|---|---|---|")
        for alg in ("PSO", "KG_GA", "NSGA2", "TBGA"):
            if alg in best:
                bb = best[alg]["_mean"]
                share = float((ms > bb).mean() * 100)
                rows.append({"baseline": alg, "baseline_best": bb, "share_mopso_better": share,
                             "mopso_median_set": float(ms.median())})
                report.append(f"| {ALG_LABEL[alg]} | {bb:.6f} | {share:.1f} % | {ms.median():.6f} |")
        pd.DataFrame(rows).to_csv(folder / "tuning_effort.csv", index=False)
        report.append("")
    write_tuning_table(folder, best)
    (folder / "report_stage1.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nselected parameters written to {folder / 'best_params.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v5/grid")
