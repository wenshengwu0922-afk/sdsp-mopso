"""Comparison with other algorithms (Section 4.3, Tables 13-14), ablation (Section 4.4, Table 15) and sensitivity
to the HV reference coordinate (Section 4.5, Table 16): tables and tests (figures: figures.py).

usage: python analyze_results.py results_v5

Statistics (Section 4.3):
  comparison / rho : Friedman test (blocks = seeds) and mean ranks; if p < 0.05, two-sided Wilcoxon signed-rank
                     tests (paired by seed) of the reference (MOPSO / rho = 1.1) against each other group,
                     Holm-adjusted; effect size = matched-pairs rank-biserial r (> 0 favours the reference).
  ablation         : Wilcoxon signed-rank test of each variant against the full MOPSO (paired by seed),
                     Holm-adjusted over the variants; r as above; n = number of seeds with different results.
Convergence speed: nfe_opt = NFE*, the evaluation at which HV(FA) first equals the exact optimum HV* (written by
ssdsp); Reach = number of runs that reached HV*; NFE* = median (25th-75th percentile) of nfe_opt over the runs
that reached HV*. Speed is tested like the HV (paired by seed, Holm), with runs that never reach HV* counted as
the full budget (right-censored at E_max); for speed, r > 0 means the reference needs fewer evaluations.
Every run's final archive is re-read from the *_front.csv files: HV is recomputed (and checked against the
value written by ssdsp), the number of exact Pareto points found is counted, and the HV is re-evaluated for
other reference points of the performance metric ((1.01,1.01), (1.05,1.05), (1.2,1.2), (1.5,1.5); Section 4.5).
Outputs: <root>/report.md, <root>/tables/*.csv
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stats_util import (ALG_LABEL, ALG_ORDER, block_matrix, compare_to_reference, exact_points_found,  # noqa: E402
                        fmt_p, holm, hv_from_points, load_exact, wilcoxon_paired)

HERE = pathlib.Path(__file__).resolve().parent
METRIC_REFS = [1.01, 1.05, 1.1, 1.2, 1.5]
ABL_ORDER = ["full", "rule_A", "swarm_leader", "rule_A_swarm", "gbest_leader", "layered_leader", "hv_pbest",
             "with_mutation"]
ABL_LABEL = {"full": "MOPSO (rule B + archive leader)", "rule_A": "Rule A + archive leader",
             "swarm_leader": "Rule B + swarm leader", "rule_A_swarm": "Rule A + swarm leader",
             "gbest_leader": "Rule B + best-HV_pt leader (gbest)",
             "layered_leader": "Rule B + layered leaders (Q = 10)",
             "hv_pbest": "HV_pt-based personal best", "with_mutation": "With swap mutation (p_m = 0.2)"}


def read_runs_and_fronts(prefix, exact, f1max, f2max):
    """Reads <prefix>_runs.csv and <prefix>_front.csv; adds exact_found and HV at several metric references."""
    runs = pd.read_csv(f"{prefix}_runs.csv")
    if (runs["nfe"] != runs["budget"]).any():
        raise SystemExit(f"{prefix}: some runs did not use the full budget")
    fr = pd.read_csv(f"{prefix}_front.csv")
    extra = []
    for seed, g in fr.groupby("seed"):
        row = {"seed": seed, "exact_found": exact_points_found(g["cost"], g["carbon"], exact)}
        for ref in METRIC_REFS:
            row[f"HV_{ref:g}"] = hv_from_points(g["cost"], g["carbon"], f1max, f2max, ref)
        extra.append(row)
    runs = runs.merge(pd.DataFrame(extra), on="seed", how="left")
    if runs["exact_found"].isna().any():
        raise SystemExit(f"{prefix}: front data missing for some seeds")
    if ((runs["HV"] - runs["HV_1.1"]).abs() > 1e-8).any():
        raise SystemExit(f"{prefix}: HV recomputed from the fronts differs from ssdsp output")
    return runs


def describe(x):
    return {"Best": x.max(), "Mean": x.mean(), "Std": x.std(ddof=1), "Worst": x.min()}


def speed(runs):
    """Reach count and median / quartiles of nfe_opt over the runs that reached HV*."""
    n = runs["nfe_opt"].to_numpy()
    hit = n[n > 0]
    if len(hit) == 0:
        return {"reach": 0, "nfe_med": np.nan, "nfe_q1": np.nan, "nfe_q3": np.nan}
    q1, med, q3 = np.percentile(hit, [25, 50, 75])
    return {"reach": int(len(hit)), "nfe_med": med, "nfe_q1": q1, "nfe_q3": q3}


def nfe_censored(runs):
    """nfe_opt per run, unreached runs counted as the budget (sorted by seed)."""
    r = runs.sort_values("seed")
    return np.where(r["nfe_opt"] > 0, r["nfe_opt"], r["budget"]).astype(float)


def speed_tests(ref_runs, others):
    """Wilcoxon (paired by seed) of the censored NFE of each other group vs the reference, Holm-adjusted.
    Returns {name: (p_holm, r)} with r > 0 when the reference is faster."""
    x0 = nfe_censored(ref_runs)
    res, ps = {}, []
    for k, rr in others.items():
        _, p, r, _ = wilcoxon_paired(nfe_censored(rr), x0)
        res[k] = [p, r]
        ps.append(p)
    for (k, v), pa in zip(res.items(), holm(np.array(ps)) if ps else []):
        v[0] = pa
    return res


def fmt_nfe(s):
    return "---" if s["reach"] == 0 else f"{s['nfe_med']:,.0f} ({s['nfe_q1']:,.0f}-{s['nfe_q3']:,.0f})"


# --------------------------------------------------------------------------------------------- ablation
def ablation(root, exact, hv_star, f1max, f2max, rep, tabdir):
    d = root / "ablation"
    tags = [t for t in ABL_ORDER if (d / f"abl_fan_{t}_runs.csv").exists()]
    if "full" not in tags:
        print("[skip] ablation: no results")
        return
    data = {t: read_runs_and_fronts(d / f"abl_fan_{t}", exact, f1max, f2max) for t in tags}
    budget = int(data["full"]["budget"].iloc[0])
    seeds = data["full"]["seed"]
    allr = pd.concat([data[t].assign(variant=t) for t in tags])
    piv = block_matrix(allr, "variant", tags)
    rows, raw = [], []
    for t in tags:
        r = {"variant": t, **describe(data[t]["HV"]), **speed(data[t]), "exact_found": data[t]["exact_found"].mean()}
        if t != "full":
            W, p, rr, n = wilcoxon_paired(piv["full"], piv[t])
            r.update({"W": W, "p_raw": p, "r_rb": rr, "n_diff": n})
            raw.append(p)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.loc[tab["variant"] != "full", "p_holm"] = holm(np.array(raw)) if raw else []
    sp = speed_tests(data["full"], {t: data[t] for t in tags if t != "full"})
    tab["p_nfe"] = tab["variant"].map(lambda t: sp[t][0] if t in sp else np.nan)
    tab["r_nfe"] = tab["variant"].map(lambda t: sp[t][1] if t in sp else np.nan)
    tab.to_csv(tabdir / "ablation.csv", index=False)
    rep += [f"## Ablation (ceiling fan, E_max = {budget:,}, 30 runs, seeds {seeds.min()}-{seeds.max()})\n",
            f"Exact optimum HV* = {hv_star:.6f} ({len(exact)} Pareto points). Reach = runs that reached HV*; "
            "NFE* = median (25th-75th percentile) evaluations needed to reach HV*; Found = mean number of exact "
            "Pareto points in the final archive. p: Wilcoxon signed-rank test on the final HV vs the full MOPSO "
            "(two-sided, paired by seed), Holm-adjusted; r > 0 means the full MOPSO attains higher HV; n = seeds "
            "with different final HV. p_NFE / r_NFE: the same test on the evaluations needed to reach HV* "
            "(unreached = budget); r_NFE > 0 means the full MOPSO is faster.\n",
            "| Variant | Mean | Std | Worst | Reach | NFE* | Found | p | r | n | p_NFE | r_NFE |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        isf = r["variant"] == "full"
        ps = "---" if isf else fmt_p(r["p_holm"])
        rs = "---" if isf else f"{r['r_rb']:.2f}"
        ns = "---" if isf else f"{int(r['n_diff'])}"
        pn = "---" if isf else fmt_p(r["p_nfe"])
        rn = "---" if isf else f"{r['r_nfe']:.2f}"
        rep.append(f"| {ABL_LABEL[r['variant']]} | {r['Mean']:.6f} | {r['Std']:.6f} | {r['Worst']:.6f} | "
                   f"{int(r['reach'])} | {fmt_nfe(r)} | {r['exact_found']:.1f} | {ps} | {rs} | {ns} | {pn} | {rn} |")
    rep.append("")


# --------------------------------------------------------------------------------------------- rho
def rho_sensitivity(root, exact, hv_star, f1max, f2max, rep, tabdir):
    d = root / "rho"
    files = sorted(d.glob("rho_fan_*_runs.csv"))
    if not files:
        print("[skip] rho: no results")
        return
    allr = pd.concat([read_runs_and_fronts(str(f)[: -len("_runs.csv")], exact, f1max, f2max) for f in files])
    allr["level"] = allr["rho"].map(lambda v: f"{v:g}")
    levels = [f"{v:g}" for v in sorted(allr["rho"].unique())]
    if "1.1" not in levels:
        raise SystemExit("rho results must include rho = 1.1")
    budget = int(allr["budget"].iloc[0])
    piv = block_matrix(allr, "level", levels)
    summ, post = compare_to_reference(piv, "1.1")
    allr["nfe_c"] = np.where(allr["nfe_opt"] > 0, allr["nfe_opt"], allr["budget"]).astype(float)
    pivn = block_matrix(allr, "level", levels, value="nfe_c")
    summ_n, post_n = compare_to_reference(-pivn, "1.1")     # negated: fewer evaluations = better
    rows = []
    for lv in levels:
        g = allr[allr["level"] == lv]
        r = {"rho": float(lv), **describe(g["HV"]), **speed(g), "exact_found": g["exact_found"].mean(),
             "mean_rank": summ["mean_rank"][lv]}
        pr = post[post["group"] == lv]
        r.update({"p_holm": pr["p_holm"].iloc[0] if len(pr) else np.nan, "r_rb": pr["r_rb"].iloc[0] if len(pr) else np.nan})
        pn = post_n[post_n["group"] == lv]
        r.update({"rank_nfe": summ_n["mean_rank"][lv], "p_nfe": pn["p_holm"].iloc[0] if len(pn) else np.nan,
                  "r_nfe": pn["r_rb"].iloc[0] if len(pn) else np.nan})
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(tabdir / "rho_sensitivity.csv", index=False)
    seeds = allr["seed"]
    rep += [f"## Sensitivity to the HV_pt reference coordinate rho (ceiling fan, E_max = {budget:,}, seeds "
            f"{seeds.min()}-{seeds.max()})\n",
            "rho only selects the representative stopping point used in the personal-best comparison; the HV "
            "performance metric uses the reference point (1.1, 1.1) throughout, and under rule B the exact "
            f"Pareto front (HV* = {hv_star:.6f}) does not depend on rho.\n",
            ("Final HV: identical for all rho levels in every run; no test.\n" if summ["identical"] else
             f"Final HV: Friedman chi2 = {summ['chi2']:.3f}, p {fmt_p(summ['p_friedman'])}"
             + ("" if summ["post_hoc"] else " (no post-hoc tests)") + "\n"),
            f"Evaluations to reach HV* (unreached = budget): Friedman chi2 = {summ_n['chi2']:.3f}, p "
            f"{fmt_p(summ_n['p_friedman'])}" + ("" if summ_n["post_hoc"] else " (no post-hoc tests)") + "\n",
            "| rho | Mean | Std | Worst | Reach | NFE* | Found | Rank (HV) | p (HV) | Rank (NFE) | p (NFE) | r (NFE) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        isref = abs(r["rho"] - 1.1) < 1e-12
        p = "---" if isref or pd.isna(r["p_holm"]) else fmt_p(r["p_holm"])
        pn = "---" if isref or pd.isna(r["p_nfe"]) else fmt_p(r["p_nfe"])
        rn = "---" if isref or pd.isna(r["r_nfe"]) else f"{r['r_nfe']:.2f}"
        rep.append(f"| {r['rho']:g} | {r['Mean']:.6f} | {r['Std']:.6f} | {r['Worst']:.6f} | {int(r['reach'])} | "
                   f"{fmt_nfe(r)} | {r['exact_found']:.1f} | {r['mean_rank']:.2f} | {p} | {r['rank_nfe']:.2f} | {pn} | {rn} |")
    rep.append("\nr (NFE) > 0 means rho = 1.1 needs fewer evaluations than the compared level.\n")


# --------------------------------------------------------------------------------------------- comparison
def comparison(root, rep, tabdir):
    d = root / "compare"
    cases = [c for c in ("stapler", "fan") if list(d.glob(f"cmp_{c}_*_MOPSO_runs.csv"))]
    if not cases:
        print("[skip] comparison: no results")
        return
    rep += ["## Formal comparison (30 runs per method and budget)\n",
            "Best / Mean / Std / Worst: final HV of the 30 runs; Gap = HV* - Mean; Reach = runs that reached HV*; "
            "NFE* = median (25th-75th percentile) evaluations needed to reach HV*; Found = mean number of exact "
            "Pareto points in the final archive; Rank = Friedman mean rank (1 = best); p: Wilcoxon signed-rank test "
            "of MOPSO vs the method on the final HV (two-sided, paired by seed), Holm-adjusted, computed only when "
            "the Friedman test is significant (p < 0.05); r > 0 favours MOPSO. p_NFE / r_NFE: Wilcoxon signed-rank "
            "test of MOPSO vs the method on the evaluations needed to reach HV* (unreached = budget), Holm-adjusted; "
            "r_NFE > 0 means MOPSO is faster.\n"]
    all_tab, refsens = [], []
    for case in cases:
        exact, f1max, f2max = load_exact(HERE, case)
        hv_star = hv_from_points(exact["cost"], exact["carbon"], f1max, f2max)
        budgets = sorted({int(f.name.split("_")[2]) for f in d.glob(f"cmp_{case}_*_runs.csv")})
        for B in budgets:
            algs = [a for a in ALG_ORDER if (d / f"cmp_{case}_{B}_{a}_runs.csv").exists()]
            data = {a: read_runs_and_fronts(d / f"cmp_{case}_{B}_{a}", exact, f1max, f2max) for a in algs}
            allr = pd.concat([data[a].assign(method=a) for a in algs])
            seeds = allr["seed"]
            piv = block_matrix(allr, "method", algs)
            summ, post = compare_to_reference(piv, "MOPSO")
            sp_t = speed_tests(data["MOPSO"], {a: data[a] for a in algs if a != "MOPSO"})
            head = (f"### {case}, E_max = {B:,} (seeds {seeds.min()}-{seeds.max()})\n\n" +
                    ("All methods obtained identical results in every run; no test.\n" if summ["identical"]
                     else f"Friedman: chi2 = {summ['chi2']:.3f}, p {fmt_p(summ['p_friedman'])}\n"))
            rep += [head, "| Method | Best | Mean | Std | Worst | Gap | Reach | NFE* | Found | Rank | p | r | p_NFE | r_NFE |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for a in algs:
                x = data[a]["HV"]
                pr = post[post["group"] == a]
                p = pr["p_holm"].iloc[0] if len(pr) else np.nan
                rr = pr["r_rb"].iloc[0] if len(pr) else np.nan
                sp = speed(data[a])
                row = {"case": case, "budget": B, "method": a, **describe(x), "gap": hv_star - x.mean(), **sp,
                       "exact_found": data[a]["exact_found"].mean(), "mean_rank": summ["mean_rank"][a],
                       "p_holm": p, "r_rb": rr, "chi2": summ["chi2"], "p_friedman": summ["p_friedman"],
                       "p_nfe": sp_t[a][0] if a in sp_t else np.nan, "r_nfe": sp_t[a][1] if a in sp_t else np.nan}
                all_tab.append(row)
                ps = "---" if a == "MOPSO" or np.isnan(p) else fmt_p(p)
                rs = "---" if a == "MOPSO" or np.isnan(rr) else f"{rr:.2f}"
                pn = "---" if a == "MOPSO" else fmt_p(row["p_nfe"])
                rn = "---" if a == "MOPSO" else f"{row['r_nfe']:.2f}"
                rep.append(f"| {ALG_LABEL[a]} | {row['Best']:.6f} | {row['Mean']:.6f} | {row['Std']:.6f} | "
                           f"{row['Worst']:.6f} | {max(row['gap'], 0.0):.6f} | {sp['reach']} | {fmt_nfe(sp)} | "
                           f"{row['exact_found']:.1f} | {row['mean_rank']:.2f} | {ps} | {rs} | {pn} | {rn} |")
            rep.append("")
            for ref in METRIC_REFS:   # Friedman mean ranks with other reference points of the HV metric
                pv = block_matrix(allr, "method", algs, value=f"HV_{ref:g}")
                s2, _ = compare_to_reference(pv, "MOPSO")
                for a in algs:
                    refsens.append({"case": case, "budget": B, "ref": ref, "method": a, "mean_HV": pv[a].mean(),
                                    "mean_rank": s2["mean_rank"][a], "p_friedman": s2["p_friedman"]})
            curves = {a: pd.read_csv(d / f"cmp_{case}_{B}_{a}_trace.csv").groupby("nfe")["HV"].mean() for a in algs}
            pd.DataFrame(curves).to_csv(tabdir / f"convergence_{case}_{B}.csv")
    pd.DataFrame(all_tab).to_csv(tabdir / "comparison.csv", index=False)
    rs = pd.DataFrame(refsens)
    rs.to_csv(tabdir / "metric_reference_sensitivity.csv", index=False)
    rep += ["## Sensitivity of the comparison to the HV metric reference point\n",
            "Friedman mean rank of each method when the final archives are re-evaluated with reference point "
            "(ref, ref).\n"]
    for (case, B), g in rs.groupby(["case", "budget"], sort=False):
        algs = [a for a in ALG_ORDER if a in set(g["method"])]
        rep.append(f"**{case}, E_max = {B:,}**\n")
        rep.append("| ref | " + " | ".join(ALG_LABEL[a] for a in algs) + " | Friedman p |")
        rep.append("|---|" + "---|" * len(algs) + "---|")
        for ref, h in g.groupby("ref"):
            mr = h.set_index("method")["mean_rank"]
            rep.append(f"| {ref:g} | " + " | ".join(f"{mr[a]:.2f}" for a in algs) + f" | {fmt_p(h['p_friedman'].iloc[0])} |")
        rep.append("")


def main(root):
    root = pathlib.Path(root)
    tabdir = root / "tables"
    tabdir.mkdir(parents=True, exist_ok=True)
    exact, f1max, f2max = load_exact(HERE, "fan")
    hv_star = hv_from_points(exact["cost"], exact["carbon"], f1max, f2max)
    rep = ["# Results report\n"]
    f = root / "grid" / "tuning_table.md"
    if f.exists():
        rep.append(f.read_text(encoding="utf-8"))
    ablation(root, exact, hv_star, f1max, f2max, rep, tabdir)
    rho_sensitivity(root, exact, hv_star, f1max, f2max, rep, tabdir)
    comparison(root, rep, tabdir)
    (root / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"report written to {root / 'report.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v5")
