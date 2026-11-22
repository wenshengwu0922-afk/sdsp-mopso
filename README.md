# Stopping-point selective disassembly with an archive-guided MOPSO

Source code and experiment scripts for the manuscript

> H.-E. Tseng, W.-P. Wang, W.-S. Wu, J.-J. Zhang. *A multi-objective particle swarm optimization approach for stopping-point selective disassembly sequence planning with carbon recovery.* Submitted to *The International Journal of Advanced Manufacturing Technology* (JAMT-D-26-05111, under revision).

The code solves selective disassembly sequence planning (SDSP) with a stopping point under two objectives, disassembly cost (min) and carbon recovery (max). Every stopping point of each decoded sequence is evaluated, and all non-dominated (sequence, stopping point) pairs enter an external archive. That archive is both the output Pareto front and the leader pool of the MOPSO (stopping-point Pareto evaluation, "rule B").


## Repository layout

| Path | Content | Manuscript |
|---|---|---|
| `src/problem.hpp` | Instance data (stapler, ceiling fan), carbon and cost model | Sec. 2, Tables 2–5, 7, 9, 10; Eqs. (1)–(14) |
| `src/evaluator.hpp` | Sequence evaluation at all stopping points, external archive, budget counter, HV | Algorithm 1; Eqs. (15)–(17) |
| `src/mopso.hpp` | Archive-guided MOPSO and ablation switches | Sec. 3, Algorithm 2; Eqs. (18)–(20) |
| `src/pso.hpp`, `kg_ga.hpp`, `nsga2.hpp`, `tbga.hpp`, `rs.hpp` | Baselines (PSO, K&G-GA, NSGA-II, TBGA, random search), all using the same evaluation and archive | Sec. 4.1 |
| `src/exact.hpp` | Exact Pareto front by enumerating all order ideals (feasible stopping states) | Table 8 |
| `src/main.cpp` | Command-line driver (multi-threaded) | — |
| `*.ps1` | Main experiment pipeline (Windows PowerShell) | Secs. 4.2–4.6 |
| `analysis/` | Statistics, tables and figures; exact fronts (`exact_*.csv`) | Tables 11–17, Figs. 4–8 |
| `sensitivity/` | Carbon-parameter scenarios, Monte Carlo, and component damage experiments, with result tables | Secs. 4.7–4.8, Tables 18–21 |

## Code and manuscript notation

Comments in the source cite the equations, tables, sections and algorithm lines of the revised manuscript.

| Manuscript (Table 6) | Code |
|---|---|
| $N$, part $i$, sequence $S=(s_1,\ldots,s_N)$, stopping point $k$ | `P.N`, part id, `std::vector<int> S` / `seq`, `k` |
| $\tau_l$ = rating × 4 s, $W_b$ = 196 NT\$/h, $C_l$ (Eq. 4) | `rating * BASE_TIME`, `WAGE`, `adj[i] = (v_l, C_l)` |
| $OCF_m$, $RCF_m$, $PCF_q$ (Tables 2–3), $\eta_m$ (Table 7) | `OCF`, `RCF`, `PCF`, `ETA` |
| $g_i$ (Eq. 8), $g_i^{bulk}$ (Eq. 9) | `P.g[i]`, `P.gbulk[i]` |
| $f_1$, $f_2$ (Eqs. 2–3), ${f_1}_{\max}$, ${f_2}_{\max}$ (Eqs. 10–11) | `f1`, `f2`, `P.f1max`, `P.f2max` |
| $\sigma_1$, $\sigma_2$, $\rho$, $HV_{pt}$ (Eqs. 12–14) | `P.sigma1()`, `P.sigma2()`, `P.rho`, `P.hv_point()` |
| $k^{rep}(S)$, $K^{ND}(S)$, $FA$ (Eqs. 15–17) | `Eval::k_rep`, rule B loop in `Evaluator::evaluate`, `Evaluator::archive()` |
| precedence matrix $A$ | `P.prec` (transitive closure of the priority graph) |
| $X_p$, $V_p$, $pBest_p$, $leader_p$, $X^a$ (Eqs. 18–20) | `Particle::x`, `v`, `pbest`, `Lbuf`, `encode()` |
| $\omega$, $c_1$, $c_2$, $\xi_1$, $\xi_2$, $P$ | `w`, `c1`, `c2`, `xi1`, `xi2`, `SW = 100` |
| $E_{\max}$, $e$ | `--budget`, `Evaluator::evaluations()` |
| $HV(FA)$, $HV^*$, Reach, $NFE^*$ | `Evaluator::hv()`, exact front HV, runs with `nfe_opt > 0`, median `nfe_opt` |
| rule B (the stopping-point Pareto evaluation of the manuscript) / rule A (only the representative stopping point enters the archive; used in the response to the reviewers, not in the manuscript) | `--rule B` (default) / `--rule A` |

## Requirements

- g++ with C++17 (GCC ≥ 9). No external C++ libraries.
- Python ≥ 3.9 with numpy, pandas, scipy, matplotlib (analysis and figures only).

## Build and quick check

```bash
make            # builds ./ssdsp
make test       # exact fronts: stapler HV* = 0.831917, fan HV* = 0.859667; 3 MOPSO runs on the fan
```

On Windows, `00_build.ps1` builds `ssdsp.exe` and runs the same self-tests.

## Command-line usage

```text
ssdsp --alg MOPSO|PSO|KG_GA|NSGA2|TBGA|RS --case stapler|fan --budget 20000 --seeds 5001-5030
      [--set "w=0,c1=3.5,c2=4.5"] [--grid "w=0,0.1;c1=1,2"]
      [--rule B|A] [--leader archive|swarm_nd|layer_hv] [--pbest dominance|hv] [--rho 1.1]
      [--trace 100] [--front] [--threads 8] --out results/name
ssdsp --exact --case fan --out analysis/exact_fan
```

One evaluation is one sequence scanned at all N stopping points. All population-based methods use a population of 100. Outputs:

- `<out>_runs.csv`: one row per run, including the final HV and `nfe_opt`, the evaluation at which the archive first reached HV* (−1 = never).
- `<out>_trace.csv`: written with `--trace`.
- `<out>_front.csv`: final archives, written with `--front`.

Results do not depend on the number of threads.

### Final parameters (Table 11)

| Method | Setting |
|---|---|
| MOPSO | `w=0,c1=3.5,c2=4.5` |
| PSO | `w=0,c1=12,c2=1.5` |
| K&G-GA | `pc=0.75,pm=0.95` |
| NSGA-II | `pc=0,pm=1` |
| TBGA | `team_ratio=0.08,delta=0.05,nm_ratio=6` |

## Reproducing the experiments

### Sections 4.2–4.6 (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_night.ps1      # resumable; outputs in results_v5\
```

| Script | Experiment | Seeds |
|---|---|---|
| `01_grid.ps1` | Full-factorial parameter grids and ANOVA (fan, E_max = 20,000) — Tables 11–12 | 1001–1030 |
| `02_ablation.ps1` | Ablation, 8 variants — Table 15, Figs. 6–7 | 2001–2030 |
| `02b_rho.ps1` | Sensitivity to ρ — Table 16 | 4001–4030 |
| `03_compare.ps1` | Comparison of 6 methods, stapler 500/1,000/1,500, fan 20,000/50,000/100,000 — Tables 13–14 | 5001–5030 |
| `04_analyze.ps1` | Friedman / Wilcoxon–Holm statistics, tables and figures (`analysis/`) — Figs. 4–5 | — |

`config.ps1` holds the whole design (budgets, grids, seeds). Fig. 8 and Table 17:
`python analysis/fig_pareto.py results_v5/compare analysis/exact_fan.csv <outdir>`.

The same runs can be issued on Linux by calling `./ssdsp` with the arguments listed in the scripts.

### Sections 4.7–4.8 (bash, Linux)

```bash
make
cd sensitivity
bash run_sens.sh           # 8 carbon-parameter scenarios x 6 methods x 2 instances (seeds 6001-6030)
python3 ana.py && python3 combine.py   # -> results/sens_algs.csv, results/sens_table.csv (Table 18)
python3 mc.py              # Monte Carlo, 500 samples, U(0.8,1.2) -> results/mc.json (Table 19)
python3 destr.py           # component damage, fan, 175 scenarios x 30 MOPSO runs (seeds 7001-7030) 
python3 destr_summary.py   # -> results/destr_summary.csv (Table 21)
```

The sensitivity runs override the model data through environment variables. Without them the program uses the manuscript values.

| Variable | Overrides |
|---|---|
| `SSDSP_OCF` | Virgin-material carbon factors |
| `SSDSP_RCF` | Recycled-material carbon factors |
| `SSDSP_PCF` | Process carbon factors |
| `SSDSP_ETA` | Bulk-recycling efficiencies |
| `SSDSP_DESTR` | Damaged part ids |

- `SSDSP_OCF`, `SSDSP_RCF` and `SSDSP_ETA` take 5 comma-separated values, ordered iron, steel, aluminium, stainless steel, copper.
- `SSDSP_PCF` takes 3 values, ordered stamping, extrusion, machining.
- A damaged part is downgraded from RU/RM to RC. Its liaisons use the destructive ratings of Das & Naik (2002), Table 5.

`sensitivity/results/` contains the result tables used in the manuscript; `ana.py`, `combine.py`, `mc.py` and `destr_summary.py` reproduce them exactly from the raw runs. Raw run files (`sensitivity/out/`) are not versioned; the scripts regenerate them.

## Exact fronts

| Instance | N | Feasible stopping states | Exact Pareto points | HV* |
|---|---|---|---|---|
| Stapler | 18 | 415 | 18 | 0.831917 |
| Ceiling fan | 26 | 24,839 | 31 | 0.859667 |

HV is computed in the normalized objective space (Eqs. (12)–(13)) with the reference point (1.1, 1.1).

## Citation

Please cite the manuscript above (citation details will be added on publication).

## License

Released under the MIT License; see [LICENSE](LICENSE).
