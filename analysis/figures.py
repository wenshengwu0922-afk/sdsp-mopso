"""Figures of Section 4 (golden ratio, ticks inside on the bottom and left axes, no grid, framed legend inside the
top of the axes, STIX font, PNG 1000 dpi + vector PDF).

usage: python figures.py results_v5        (needs matplotlib; run after analyze_grid.py and analyze_results.py)

  convergence_<case>_<budget>  mean HV(FA) of the 30 runs vs NFE (five algorithms; Fig. 4: stapler 1,500, Fig. 5: fan 20,000)
  heatmap_<ALG>_c1c2           mean HV over c1 x c2 at the selected w (MOPSO, PSO); bold frame = selected set (supplementary)
  ablation_hv                  final HV of every run for the rule-B ablation variants (Reach annotated; Fig. 6)
  ablation_nfe                 evaluations needed to reach HV* (runs that reached it; Fig. 7)
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    print("[info] matplotlib not available: figures skipped")
    sys.exit(0)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stats_util import hv_from_points, load_exact  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})
GOLD = (6.0, 6.0 / 1.618)
STYLE = {  # label, colour, linestyle, linewidth
    "MOPSO": ("MOPSO (proposed)", "#1F5AA6", "-", 2.0),
    "PSO": ("PSO", "#D55E00", "--", 1.4),
    "KG_GA": ("K&G-GA", "#009E73", "-.", 1.4),
    "NSGA2": ("NSGA-II", "#E69F00", ":", 1.8),
    "TBGA": ("TBGA", "#CC79A7", (0, (5, 1, 1, 1, 1, 1)), 1.4),
    "RS": ("Random search", "#555555", (0, (8, 2)), 1.2),
}
THOUSANDS = FuncFormatter(lambda v, p: f"{int(round(v)):,}")


def style_ax(ax, labelsize=11):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.9)
    ax.tick_params(direction="in", top=False, right=False, labelsize=labelsize, length=4)
    ax.grid(False); ax.set_facecolor("white")


def framed_legend(ax, handles=None, labels=None, ncol=3):
    kw = dict(loc="upper center", ncol=ncol, fontsize=10, frameon=True, fancybox=False, edgecolor="black",
              framealpha=1.0)
    leg = ax.legend(handles, labels, **kw) if handles else ax.legend(**kw)
    leg.get_frame().set_linewidth(0.8)


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=1000, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def hv_star(case):
    e, f1max, f2max = load_exact(HERE, case)
    return hv_from_points(e["cost"], e["carbon"], f1max, f2max)


def convergence(tabdir, figdir):
    for f in sorted(tabdir.glob("convergence_*_*.csv")):
        _, case, budget = f.stem.split("_")
        budget = int(budget)
        cv = pd.read_csv(f, index_col=0)
        hs = hv_star(case)
        fig, ax = plt.subplots(figsize=GOLD)
        style_ax(ax)
        algs = [a for a in ["PSO", "KG_GA", "NSGA2", "TBGA", "MOPSO"] if a in cv.columns]   # random search not drawn
        for a in algs:
            lab, col, ls, lw = STYLE[a]
            ax.plot(cv.index, cv[a], color=col, ls=ls, lw=lw, label=lab, zorder=3 if a == "MOPSO" else 2)
        ax.axhline(hs, color="#7F7F7F", lw=0.8, ls="-", label="Exact optimum", zorder=1)
        lo, span = cv.min().min(), hs - cv.min().min()
        lo = min(cv[a].min() for a in algs)
        span = hs - lo
        ax.set_ylim(lo - 0.03 * span, hs + 0.40 * span)   # top band for the 2-row legend
        ax.set_xlim(0, budget)
        ax.xaxis.set_major_formatter(THOUSANDS)
        ax.set_xlabel("Number of function evaluations (NFE)", fontsize=12)
        ax.set_ylabel("Mean HV of the archive", fontsize=12)
        order = [STYLE[a][0] for a in ["MOPSO", "PSO", "KG_GA", "NSGA2", "TBGA"] if a in algs] + ["Exact optimum"]
        h, l = ax.get_legend_handles_labels()
        framed_legend(ax, [h[l.index(x)] for x in order], order)
        fig.tight_layout()
        save(fig, figdir / f"convergence_{case}_{budget}")


def heatmap(griddir, figdir, alg, best):
    f = griddir / f"grid_summary_{alg}.csv"
    if not f.exists() or alg not in best:
        return
    d = pd.read_csv(f)
    d = d[np.isclose(d["w"], best[alg]["w"])]
    piv = d.pivot_table(index="c2", columns="c1", values="mean").sort_index()
    fig, ax = plt.subplots(figsize=GOLD)
    im = ax.imshow(piv.values, cmap="coolwarm", origin="lower", aspect="auto")
    ncell = piv.size
    if ncell <= 80:
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", fontsize=9.5)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{v:g}" for v in piv.columns], fontsize=9)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{v:g}" for v in piv.index], fontsize=9)
    ax.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, piv.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(direction="in", top=False, right=False, length=4)
    for s in ax.spines.values():
        s.set_visible(False)
    j = list(piv.columns).index(best[alg]["c1"]); i = list(piv.index).index(best[alg]["c2"])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8, zorder=5))
    ax.set_xlabel(r"Cognitive coefficient $c_1$", fontsize=12)
    ax.set_ylabel(r"Social coefficient $c_2$", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction="in", labelsize=10)
    cb.set_label("Mean HV (30 runs)", fontsize=11)
    fig.tight_layout()
    save(fig, figdir / f"heatmap_{alg}_c1c2")


def ablation(root, figdir):
    d = root / "ablation"
    tags = ["full", "swarm_leader", "gbest_leader", "layered_leader", "hv_pbest", "with_mutation"]
    names = {"full": "MOPSO", "swarm_leader": "Swarm\nleader", "gbest_leader": "gbest\nleader",
             "layered_leader": "Layered\nleaders", "hv_pbest": "HV$_{pt}$\npbest", "with_mutation": "With\nmutation"}
    tags = [t for t in tags if (d / f"abl_fan_{t}_runs.csv").exists()]
    if not tags:
        return
    runs = {t: pd.read_csv(d / f"abl_fan_{t}_runs.csv") for t in tags}
    hs = hv_star("fan")
    rng = np.random.default_rng(0)
    # final HV
    fig, ax = plt.subplots(figsize=GOLD)
    style_ax(ax)
    allv = np.concatenate([runs[t]["HV"].to_numpy() for t in tags])
    lo = allv.min()
    for k, t in enumerate(tags):
        v = runs[t]["HV"].to_numpy()
        reach = v >= hs - 1e-9
        x = k + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x[~reach], v[~reach], s=14, facecolors="white", edgecolors="#1F5AA6", linewidths=0.8, zorder=3)
        ax.scatter(x[reach], v[reach], s=14, color="#D55E00", zorder=4)
        ax.text(k, hs + 0.04 * (hs - lo), f"{int(reach.sum())}/{len(v)}", ha="center", va="bottom", fontsize=10)
    ax.axhline(hs, color="#7F7F7F", lw=0.8, zorder=1)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([names[t] for t in tags], fontsize=10)
    ax.set_ylim(lo - 0.05 * (hs - lo), hs + 0.45 * (hs - lo))
    ax.set_ylabel("Final HV", fontsize=12)
    ax.scatter([], [], s=14, color="#D55E00", label="Reached exact optimum")
    ax.scatter([], [], s=14, facecolors="white", edgecolors="#1F5AA6", linewidths=0.8, label="Not reached")
    framed_legend(ax, ncol=2)
    fig.tight_layout()
    save(fig, figdir / "ablation_hv")
    # evaluations needed to reach the exact optimum
    fig, ax = plt.subplots(figsize=GOLD)
    style_ax(ax)
    B = int(runs[tags[0]]["budget"].iloc[0])
    for k, t in enumerate(tags):
        n_all = runs[t]["nfe_opt"].to_numpy()
        n = n_all[n_all > 0]
        if len(n):
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(n)), n, s=14, color="#1F5AA6", zorder=3)
            ax.plot([k - 0.28, k + 0.28], [np.median(n)] * 2, color="black", lw=1.2, zorder=4)
        ax.text(k, B * 1.02, f"{len(n)}/{len(n_all)}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([names[t] for t in tags], fontsize=10)
    ax.set_ylim(0, B * 1.12)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("NFE to reach the exact optimum", fontsize=12)
    fig.tight_layout()
    save(fig, figdir / "ablation_nfe")


def main(root):
    root = pathlib.Path(root)
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    convergence(root / "tables", figdir)
    bp = root / "grid" / "best_params.json"
    if bp.exists():
        best = json.loads(bp.read_text())
        for alg in ("MOPSO", "PSO"):
            heatmap(root / "grid", figdir, alg, best)
    ablation(root, figdir)
    print(f"figures written to {figdir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v5")
