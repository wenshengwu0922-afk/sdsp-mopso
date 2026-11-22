"""Section 4.6, Fig. 8: (a) exact Pareto front of the ceiling fan vs the final archive of the median-HV run of each
baseline (E_max = 20,000); (b) maximum carbon recovery attainable under a disassembly-cost budget."""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})
res, exact_csv, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
E = pd.read_csv(exact_csv)
ST = {"PSO": ("PSO", "#D55E00", "s"), "KG_GA": ("K&G-GA", "#009E73", "^"), "NSGA2": ("NSGA-II", "#E69F00", "D"),
      "TBGA": ("TBGA", "#CC79A7", "v")}
def style(ax):
    for s in ax.spines.values(): s.set_visible(True); s.set_color("black"); s.set_linewidth(0.9)
    ax.tick_params(direction="in", top=False, right=False, labelsize=11, length=4); ax.grid(False)
fronts, info = {}, {}
for a in ST:
    r = pd.read_csv(res / f"cmp_fan_20000_{a}_runs.csv").sort_values(["HV", "seed"]).reset_index(drop=True)
    med = r.iloc[(len(r) - 1) // 2]                      # lower median run
    f = pd.read_csv(res / f"cmp_fan_20000_{a}_front.csv")
    fronts[a] = f[f.seed == med.seed].sort_values("cost")
    info[a] = (int(med.seed), float(med.HV), len(fronts[a]))
def best_under(df, b):
    v = df[df.cost <= b + 1e-12].carbon
    return v.max() if len(v) else np.nan
# (a) fronts
fig, ax = plt.subplots(figsize=(6.0, 6.0 / 1.618)); style(ax)
ax.plot(E.cost, E.carbon, "o", ms=6, mfc="white", mec="#1F5AA6", mew=1.2, label=f"Exact front = MOPSO ({len(E)} pts)", zorder=3)
for a, (lab, col, mk) in ST.items():
    f = fronts[a]
    ax.plot(f.cost, f.carbon, mk, ms=4, color=col, label=f"{lab} ({len(f)} pts)", zorder=4)
ax.set_xlabel("Disassembly cost $f_1$ (NT\\$)", fontsize=12); ax.set_ylabel("Carbon recovery $f_2$ (kgCO$_2$e)", fontsize=12)
lo, hi = E.carbon.min(), E.carbon.max(); ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))
leg = ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
leg.get_frame().set_linewidth(0.8); fig.tight_layout()
fig.savefig(out / "pareto_fan_a.png", dpi=1000, facecolor="white"); fig.savefig(out / "pareto_fan_a.pdf", facecolor="white"); plt.close(fig)
# (b) best carbon recovery under budget
B = np.linspace(0, E.cost.max(), 2000)
fig, ax = plt.subplots(figsize=(6.0, 6.0 / 1.618)); style(ax)
ax.step(B, [best_under(E, b) for b in B], where="post", color="#1F5AA6", lw=2.0, label="Exact front = MOPSO")
for a, (lab, col, mk) in ST.items():
    ls = {"PSO": "--", "KG_GA": "-.", "NSGA2": ":", "TBGA": (0, (5, 1, 1, 1, 1, 1))}[a]
    ax.step(B, [best_under(fronts[a], b) for b in B], where="post", color=col, lw=1.4, ls=ls, label=lab)
ax.set_xlabel("Disassembly-cost budget (NT\\$)", fontsize=12); ax.set_ylabel("Maximum carbon recovery (kgCO$_2$e)", fontsize=12)
ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.40 * (hi - lo))
leg = ax.legend(loc="upper center", ncol=3, fontsize=10, frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
leg.get_frame().set_linewidth(0.8); fig.tight_layout()
fig.savefig(out / "pareto_fan_b.png", dpi=1000, facecolor="white"); fig.savefig(out / "pareto_fan_b.pdf", facecolor="white"); plt.close(fig)
for a in ST: print(a, "median run seed %d HV %.6f pts %d" % info[a])
for b in (1.0, 2.0, 3.0, 5.0):
    print("budget", b, "exact %.3f" % best_under(E, b), " ".join(f"{a} {best_under(fronts[a], b):.3f}" for a in ST))
